import time
import os
import xlrd
import numpy as np 
from pyedflib import highlevel
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import re

class DataPreprocess(object):
    def __init__(self, SLIDE=2, CSA=-1, NOR=0, OSA=1, HYP=2, MSA=3, fs=100, fsDown=4, psg=0, hyp=1, osa=2, csa=3, msa=4, nor=5, lens=6, AHI=7):
        #the sliding window has 0.5 sec overlap
        self.SLIDE = SLIDE #/SLIDE(sec)

        #stage assignment
        self.CSA = CSA
        self.NOR = NOR
        self.OSA = OSA
        self.HYP = HYP
        self.MSA = MSA

        #sampling rate / downsampling
        self.fs = fs
        self.fsDown = fsDown
        self.downRate = np.fix(fs/fsDown)

        #state
        self.psg = psg
        self.hyp = hyp
        self.osa = osa
        self.csa = csa
        self.msa = msa
        self.nor = nor
        self.lens = lens
        self.AHI = AHI
        
    def Back2noromal(self, b2n_signal):

        # b2n_signal = loadmat(b2n_signal_mat_path).get('signal').flatten()
        # b2n_signal = [1 , 2, 3, 4, 5, 6, 0 ,0 ,1 ,0 ,1 ,2]

        length = len(b2n_signal)
        zero = "0" 

        for i in range(len(b2n_signal)):
            if b2n_signal[i] == 0:
                zero += "1"
            else:
                zero += "0"

        pattern = re.compile(r'1+')
        slice_part = pattern.findall(zero)
        
        M=[]
        for i in slice_part:
            M.append(len(i))

        find_slice_part_index = re.finditer(r'1+',zero)

        start=[]
        for i in find_slice_part_index:
            start.append(i.start())

        idx = []
        for i in range(len(M)):
            s = max(start[i] - 6*self.fsDown, 1)
            e = min(start[i] + M[i] + 6*self.fsDown, length)+1
            p = range(s,e)
            for j in p:
                idx.append(j)

        idx_uni = np.unique(idx)
        Tem = b2n_signal

        for i in idx_uni:
            Tem[i] = np.nan            # Remove the zeros to prevent the bias of mean

        mu = np.nanmean(Tem)
        for i in idx_uni:
            b2n_signal[i] = mu
        b2n_signal_n = b2n_signal - mu         #Let the mean of signal back to 0.
        return b2n_signal_n

    def prepocessPSG(self, s):
        patient_type, name, lightoff_time, startRecord_time = s
        lightoff_time, startRecord_time = float(lightoff_time), float(startRecord_time)
        recordOffSet = (lightoff_time - startRecord_time)*24*60*60
        THO, ABD, CFlow, SpO2, STAGE, T_LEN = self.InputTestDataLK_PSG(patient_type, name, recordOffSet)
        THO = self.Back2noromal(THO)
        ABD = self.Back2noromal(ABD)
        CFlow = self.Back2noromal(CFlow)
        state = self.InputTestDataLK_Event(patient_type, name, lightoff_time, T_LEN)
        Channels =[THO,ABD,STAGE,CFlow,SpO2]
        return state, Channels

    def InputTestData(self, pnumLK_path, lightoff_path):
        '''
        Input:
            pnumLK_path: str
            lightoff_path: str
        Output:
            pnum: int
            state: 
            Channels: 
        '''
        wb_pnumLK = xlrd.open_workbook(pnumLK_path)
        sheet_pnumLK = wb_pnumLK.sheet_by_index(0)
        pnumLK_df = pd.DataFrame([sheet_pnumLK.row_values(i) for i in range(sheet_pnumLK.nrows )], columns=['type', 'name'])
        
        #open excel file lightoff_path ="./database/LK/lightoff_for_python.xlsx"
        wb_lightoff = xlrd.open_workbook(lightoff_path)
        sheet_lightoff = wb_lightoff.sheet_by_index(0)
        lightoff_df = pd.DataFrame([sheet_lightoff.row_values(i) for i in range(sheet_lightoff.nrows )], columns=['name', 'lightoff_time', 'startRecord_time'])
        
        df = pnumLK_df.merge(lightoff_df, on='name')
        res = df.apply(self.prepocessPSG, axis=1)
        res = pd.DataFrame(res.to_list(), columns=['state', 'Channels'])
        
        pnum = df.shape[0]
        state = res['state'].values.tolist()
        Channels = res['Channels'].values.tolist()
        return pnum, state, Channels

    def trigger_by_cloudfunction(self, ticket_path):
        '''
        Input:
            ticket_path: str
        Output:
            state: 
            Channels: 
        '''
        return self.prepocessPSG(pd.read_csv(ticket_path, header=None)[0])
        # patient_type, name, lightoff_time, startRecord_time = pd.read_csv(ticket_path)
        # return self.prepocessPSG([patient_type, name, lightoff_time, startRecord_time]) 

    def InputTestDataLK_PSG(self, ptype, pnum, recordOffSet):
        
        folder_path ="./database/LK/"+ ptype +"/"+pnum
        STAGE_path = folder_path + "/STAGE.csv"
        items = os.listdir(folder_path)
        edf_list = []
        for names in items:
            if names.endswith(".edf"):
                edf_list.append(names)
        edf_path = folder_path +"/"+edf_list[0]
        # read an edf file
        PSG_signals, signal_headers, header = highlevel.read_edf(edf_path)
        #'EEG C3-A2', 'EEG C4-A1', 'EEG O1-A2', 'EEG O2-A1', 'EEG A1-A2', 'EOG Left', 'EOG Right', 'EMG Chin', 'ECG I', 'RR', 'ECG II', 'Snore', 'SpO2', 'Flow Patient', 'Flow Patient', 'Effort Tho', 'Effort Abd', 'Body', 'Pleth', 'Leg RLEG', 'Leg LLEG', 'Imp']
        data = pd.read_csv(STAGE_path, encoding= 'unicode_escape')
        STAGE = data.values[:,1].tolist()
        CFlow = PSG_signals[13]
        THO = PSG_signals[15]
        ABD = PSG_signals[16]
        SpO2 = PSG_signals[12]
        #SpO2 = [round(num, 2) for num in signals[12]]
        
        STAGE_Len = len(STAGE)  # Total second of this patient
        StartTime = int(np.round(recordOffSet))

        #skip before StartTime*fs & after STAGE_Len*self.fs 
        CFlow = CFlow[StartTime*self.fs:STAGE_Len*self.fs]
        THO = THO[StartTime*self.fs:STAGE_Len*self.fs]
        ABD = ABD[StartTime*self.fs:STAGE_Len*self.fs]
        SpO2 = SpO2[StartTime:STAGE_Len]
        STAGE = STAGE[StartTime:]

        #resample 
        CFlow = signal.resample_poly(CFlow, self.fsDown, self.fs).tolist()
        THO = signal.resample_poly(THO, self.fsDown, self.fs).tolist()
        ABD = signal.resample_poly(ABD, self.fsDown, self.fs).tolist()

        return THO, ABD, CFlow, SpO2, STAGE, STAGE_Len
    
    def StatusRecoder(self, statePSG, stateNOR, state, LENS, time, duration, event):
        for ii, t in enumerate(time):
            start = int(np.fix(t*self.SLIDE))
            end = int(np.fix((t+duration[ii])*self.SLIDE))
            for jj in range(start,end):
                statePSG[jj] = event
                state[jj] = 1
                stateNOR[jj] = 0
            LENS = max(LENS, jj+1)
        return statePSG, stateNOR, state, LENS

    def InputTestDataLK_Event(self, ptype, pnum, t_lightoff, T_LEN):
        
        Dc, Do, Dm, Dh = [], [], [], []             # duration of each event (record in second)
        t_csa, t_osa, t_msa, t_hyp = [], [], [], [] # start time of each event after light off (record in second)
        numerator = 0

        STAGE_path ="./database/LK/"+ ptype +"/"+pnum+"/STAGE.csv"
        data = pd.read_csv(STAGE_path, encoding= 'unicode_escape')
        STAGE = data.values.tolist()
        Eventlist_path ="./database/LK/"+ ptype +"/"+pnum+"/Eventlist.xlsx"

        #open excel file
        wb = xlrd.open_workbook(Eventlist_path)
        sheet = wb.sheet_by_index(0)
        for index in range(sheet.nrows):
            # if type(sheet.row_values(index)[0]) is not str:
            if not isinstance(sheet.row_values(index)[0], str):
                # Program to extract a particular row value
                time = sheet.row_values(index)[0] # time(convert percentage of day format)
                if time < 0.375:
                    time += 1
                event = sheet.row_values(index)[2] # event(Central apnea/Obstructive apnea....)
                duration = sheet.row_values(index)[3]
                subtration = round((time - t_lightoff)*24*60*60,3)# time - t_lightoff

                if event == "Central apnea":
                    Dc.append(duration)
                    t_csa.append(subtration)
                    numerator += 1
                elif event == "Obstructive apnea":
                    Do.append(duration)
                    t_osa.append(subtration)
                    numerator += 1
                elif event == "Mixed apnea":
                    Dm.append(duration)
                    t_msa.append(subtration)
                    numerator += 1
                elif event == "Hypopnea":
                    Dh.append(duration)
                    t_hyp.append(subtration)
                    numerator += 1

        statePSG = [0]*(T_LEN*self.SLIDE+1)
        stateNOR = [1]*(T_LEN*self.SLIDE+1)  

        LENS = 0   #LEN is the last time that apean event occur
        statePSG, stateNOR, stateOSA, LENS = self.StatusRecoder(statePSG, stateNOR, [0]*(T_LEN*self.SLIDE+1), LENS, t_osa, Do, self.OSA)
        statePSG, stateNOR, stateCSA, LENS = self.StatusRecoder(statePSG, stateNOR, [0]*(T_LEN*self.SLIDE+1), LENS, t_csa, Dc, self.CSA)
        statePSG, stateNOR, stateMSA, LENS = self.StatusRecoder(statePSG, stateNOR, [0]*(T_LEN*self.SLIDE+1), LENS, t_msa, Dm, self.MSA)
        statePSG, stateNOR, stateHYP, LENS = self.StatusRecoder(statePSG, stateNOR, [0]*(T_LEN*self.SLIDE+1), LENS, t_hyp, Dh, self.HYP)
        
        total = 0
        for row in STAGE:
            if row[1] != 11:
                total += 1
        denominator = np.round(total/3600,1)
        return [statePSG[:LENS], stateHYP[:LENS], stateOSA[:LENS], stateCSA[:LENS], stateMSA[:LENS], stateNOR[:LENS], LENS, numerator/denominator]
        ## return = [psg, hyp, osa, csa, msa, nor, lens, AHI]


def RollingFucntion(signal, Fs, func):
    # reproduce matlab moving funcion (eg: movmean, movsum) via Pandas
    Fs = int(Fs)
    if func == 'mean':
        return pd.DataFrame(signal).rolling(window=Fs, min_periods=1, center=True).mean()[0].values
    elif func == 'sum':
        return pd.DataFrame(signal).rolling(window=Fs, min_periods=1, center=True).sum()[0].values
    else:
        print ('Parameter "func" is required')

def RpeakUltraLong(ecg, Fs):
    # Parameters
    NN = len(ecg)
    beta = 0.08
    QRS_length = round(0.097 * Fs)
    signal_length = min(5 * Fs, NN)

    # bandpass filter
    [b, a] = signal.butter(3, [8*2/Fs , 20*2/Fs], 'bandpass');
    y = signal.filtfilt(b, a, ecg)

    # QRS isolation 
    u = np.square(y)
    z = RollingFucntion(u, QRS_length, 'mean')

    # stft parameters
    hlength = 5 * Fs + 1 - Fs % 2; # window length, must be odd
    Lh = (hlength - 1) / 2
    h = np.hanning(hlength) # hann window
    hop = Fs # estimate heart rate every one second
    n = 2 * Fs + 1 # resolution of frequency axis, must be odd
    N = 2 * (n - 1) # number of fft points
    hf = 6 # 360 bpm
    lf = 0.5 # 30 bpm
    t = np.array([i for i in range(hop, NN+1, hop)], ndmin=2)  # samples at which to take the FT
    tcol = t.shape[1] # number of frequency estimates

    # signal to take FT
    x = z

    # non-negative frequency axis, cropped
    fr = Fs / 2 * np.linspace(0, 1, n) 
    eta = np.logical_and(fr >= lf, fr <= hf)
    fr = fr[np.logical_and(fr >= lf, fr <= hf)]

    # DFT matrix
    w = 2*np.pi*1j / np.array(N) * (np.argwhere(eta) + np.array(1)) 
    D = np.exp(-w*np.arange(hlength))

    # STFT, serial
    f = np.zeros((tcol, 1))
    l = .01
    rSig = np.zeros(len(h))
    for icol in range(tcol):
        ti = t[0]
        ti = t[:,icol][0]
        tau = np.arange(-min(int(Lh), ti-1), min(int(Lh), NN - ti)+1).reshape(1, -1)
        
        rSig[int(Lh) + tau] = x[ti+tau-1]
        rSig = rSig - np.mean(rSig)  # - mean(x(ti + tau)); % remove low-frequency content
        tfr = np.dot(D, rSig*h)
        if icol == 0:
            i = np.argmax(abs(tfr**2))
        else:
            tfr = abs(tfr**2)
            tfr = tfr / sum(tfr)
            i = np.argmax(tfr - (l*(fr - fr[i])**2))   # penalty for jumping
        f[icol] = fr[i]

    # time-varying threshold
    win = np.round(Fs*0.611/np.sqrt(f)) # Bazett's formula
    g, index = np.unique(win, return_inverse=True)
    interpolate = interp1d(x=t.reshape(-1), y=index, kind='nearest', fill_value='extrapolate')
    index = interpolate(list(range(1, NN+1)))
    V = np.zeros_like(u)
    for i in range(len(g)):
        s = index == i
        v = RollingFucntion(u, g[i], 'mean')
        V[s] = v[s]

    # alpha
    alpha = RollingFucntion(u, signal_length, 'mean') * beta;

    # QRS detection
    r = z > V + alpha;

    # ensure detected segments are long enough
    QRS = RollingFucntion(r, QRS_length, 'sum') == QRS_length;

    # R peak detection
    string = "".join(QRS.astype('int').astype('str'))
    c, d = [], []
    for obj in re.finditer('1+', string):
        c.append(obj.start())
        d.append(obj.end())
        
    R = np.empty((len(c), 1))
    R[:] = np.nan

    for i, (s, e) in enumerate(zip(c, d)):
        R[i] = np.argmax(z[s:e])
        R[i] += c[i]
    
    print ('Peak detection completed!')
    return R.squeeze()

def PPG_peakdetection(ppg, Fs):
    
    def RpeakIndex(s, e):
        if e - s > QRS_length:
            idx = np.argmax(ppg[s:e])
            return s+idx
        else:
            return 0

    def QpointIndex(e):
        s = max(1, int(e - 0.2*Fs))
        idx = np.argmin(ppg[s:e])
        return s+idx

    ## Fill Nan
    ppg[np.isnan(ppg)] = 0

    # parameters
    threshold = 0.02
    QRS_length = np.fix(0.11 * Fs)
    beat_length = np.fix(0.66 * Fs)

    # QRS isolation
    [b, a] = signal.butter(2, [0.5*2/Fs , 8*2/Fs], 'bandpass');
    filtered = signal.filtfilt(b, a, ppg, padlen=3*(max(len(b),len(a))-1))
    filtered[filtered<0] = 0
    u = filtered**2
    V = signal.filtfilt(np.ones(int(beat_length))/beat_length, 1, u , padlen=int(3*(beat_length-1)))
    indicator = signal.filtfilt(np.ones(int(QRS_length))/QRS_length, 1, u, padlen=int(3*(QRS_length-1)))

    mu = RollingFucntion(u, Fs*30, 'mean')

    try:
        threshold = signal.filtfilt(np.ones(5*Fs)/(5*Fs), 1, u, padlen=5*Fs-1) * threshold
    except:
        threshold *= mu
        
    t = indicator > V + threshold

    # QRS detection
    string = "".join(t.astype('int').astype('str'))
    starts, ends = [], []
    for obj in re.finditer('1+', string):
        starts.append(obj.start())
        ends.append(obj.end())

    vfunc_Rpeak = np.vectorize(RpeakIndex)
    R = vfunc_Rpeak(starts, ends)
    R = R[R>0]

    # df = pd.DataFrame({'start': R[:-1], 'end':R[1:]})
    # df['status'] = df.apply(lambda x: True if x['end'] - x['start'] < np.ceil(0.3*Fs) else False, axis=1)
    # idx = df[df['status'] == True].index+1
    diff = np.diff(R)
    idx = np.where(diff < np.ceil(0.3*Fs))[0]+1
    R = np.delete(R, idx)

    vfunc_Qpoint = np.vectorize(QpointIndex)
    Q = vfunc_Qpoint(R)
    return R, Q