import sys
import numpy as np
import pandas as pd
from collections import deque
# from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import pchip_interpolate
from scipy.stats import skew, kurtosis


class Partition(object):
    def __init__(self, Fs=2):        
        self.Fs = Fs

    def data_partition(self, R, duration, sqi, stage_path, window=9, bidirectional=True):
        
        STAGE = pd.read_csv(stage_path, index_col=0, encoding='unicode_escape')
        
        # annotation per 30s
        annotation = [STAGE.iloc[:,0].values[i] for i in range(0, STAGE.shape[0], 30)]

        # annotation length
        length = len(annotation)

        if length != len(sqi):
            sys.exit('Warning: annotation length must equal to sqi length!')     

        if bidirectional:
            pad = int(window/2)
        else:
            pad = 0

        labels, data, start = [], [], 0
        features = deque()

        for i in range(length-pad):

            lower, upper = i/length, (i+1)/length
            current_signal = R[(R >= lower*duration) & (R <= upper*duration)]
            if current_signal.shape[0] > 5:
                current_features = self.TimeSeries_feature_extraction(current_signal)
                current_features = np.array(current_features)
            else:
                current_features = np.zeros(31)
            
            current_features = np.append(current_features, sqi[i])   # append sqi information to current features vector
            features.append(current_features)

            if len(features) == window:
                lower = start/length
                global_signal = R[ (R >= lower*duration) & (R <= upper*duration)]
                start += 1
                
                # Ignore if signal is too short
                if len(global_signal) < 30:
                    features.popleft()
                    continue
                global_features = self.Global_feature_extraction(global_signal)
                global_features = np.tile(global_features, (window, 1))
                data_i = np.concatenate((np.array(features), global_features), axis=1)
                data.append(data_i)
                labels.append(annotation[i-pad])
                features.popleft()
        return np.array(data), np.array(labels)


    def TimeSeries_feature_extraction(self, signal):
        RRi = np.diff(signal)
        HRV = 1/RRi
        
        # Time domain
        dRRi = np.diff(RRi)
        
        RRi = RRi[~np.isnan(RRi)]
        dRRi = dRRi[~np.isnan(dRRi)]
        
        RR50 = np.median(RRi)
        RRmean = np.mean(RRi)
        
        HR50 = np.median(HRV)
        HRmean = np.mean(HRV)
        
        SDNN = np.std(RRi, ddof=1)
        
        # AHRR is RR range
        AHRR = np.quantile(RRi, 0.995) - np.quantile(RRi, 0.005)
        pNN50 = (1000*dRRi>50).sum()/ len(dRRi)
        RMSSD = np.sqrt((dRRi**2).sum()/len(dRRi))
        SDSD = np.std(dRRi, ddof=1)
        MAD = abs(RRi - np.mean(RRi)).mean()
        
        HR05 = np.quantile(HRV, 0.05) 
        HR10 = np.quantile(HRV, 0.10) 
        HR25 = np.quantile(HRV, 0.25) 
        HR50 = np.quantile(HRV, 0.50) 
        HR75 = np.quantile(HRV, 0.75) 
        HR90 = np.quantile(HRV, 0.90) 
        HR95 = np.quantile(HRV, 0.95) 
        
        
        RR05 = np.quantile(RRi, 0.05) 
        RR10 = np.quantile(RRi, 0.10) 
        RR25 = np.quantile(RRi, 0.25) 
        RR50 = np.quantile(RRi, 0.50) 
        RR75 = np.quantile(RRi, 0.75) 
        RR90 = np.quantile(RRi, 0.90) 
        RR95 = np.quantile(RRi, 0.95) 
        
        pNN20 = len(dRRi[dRRi*1000>20]) / len(dRRi)
        IQRNN = RR75 - RR25
        hist, bin_edges = np.histogram(RRi, range=(float('-inf'), float('inf')), bins=np.linspace(0, 3, 256))
        HTI = hist.max()/len(RRi)
        skew_v = skew(RRi)
        kurt_v = kurtosis(RRi)
        SD1 = SDSD/ np.sqrt(2)
        SD2 = np.sqrt(2*SDNN**2 - SDSD**2/2)
        return RRi.mean(), np.median(RRi), HRV.mean(), np.median(HRV), SDNN, AHRR, pNN50, RMSSD, SDSD, MAD, HR05, HR10, HR25, HR50, HR75, HR90, HR95, RR05, RR10, RR25, RR50, RR75, RR90, RR95, pNN20, IQRNN, HTI, skew_v, kurt_v, SD1, SD2

    def Global_feature_extraction(self, signal):
        RRi = np.diff(signal)
        time = np.cumsum(RRi)

        time_resample = np.arange(time[0], time[-1]+1, 1/self.Fs)
        RRI_resample = pchip_interpolate(time, RRi, time_resample)
        
        if len(RRI_resample) % 2:
            RRI_resample = RRI_resample[:-1]
            time_resample = time_resample[:-1]
        
        tmpX = np.array([np.ones(len(time_resample)), time_resample])
        
        m1 = np.dot(RRI_resample, tmpX.T)
        m2 = np.linalg.inv(np.dot(tmpX, tmpX.T))
        betahat = np.dot(m1, m2)
        
        RRI_resample = RRI_resample - np.dot(betahat, tmpX)
        
        xi = self.Fs * np.arange(1, (len(RRI_resample)/2)+1) / len(RRI_resample)
        
        RRIhat = np.fft.fft(RRI_resample)
        
        P1 = np.square(abs(RRIhat[1:int((len(RRIhat)/2)+1)]))
        
        TOTAL = np.trapz(P1, xi)
        
        # HF power, 0.15 to 0.50 Hz
        down, up = 0.15, 0.5
        HF = np.log(np.trapz(P1[(xi >= down) & (xi <= up)], xi[(xi >= down) & (xi <= up)]) / TOTAL)
        
        # LF power, 0.04 to 0.15 Hz
        down, up = 0.04, 0.15
        LF = np.log(np.trapz(P1[(xi >= down) & (xi <= up)], xi[(xi >= down) & (xi <= up)]) / TOTAL)
        
        LHR = np.log(LF/HF)
        
        down, up = 0.003, 0.04
        if len(xi[(xi >= down) & (xi <= up)]) > 2:
            VLF = np.log(np.trapz(P1[(xi >= down) & (xi <= up)], xi[(xi >= down) & (xi <= up)]) / TOTAL)
        else:
            VLF = np.nan
        return VLF, LF, HF, LHR
        
