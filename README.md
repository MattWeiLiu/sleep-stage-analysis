# Sleep stage analysis

[GitHub]

### 1.R peak detection
```python
from pyedflib import highlevel
from source.utils import PPG_peakdetection

# read edf
Fs = 200
edf_path = 'path_to_edf'
signals, signal_headers, header = highlevel.read_edf(edf_path)
ppg = signals[18]

# R peak detection
R, Q = PPG_peakdetection(ppg, Fs)
```


### 2. Data Partition
```python
from source.DataPartition import DataPartition

# data partition
stage_path = 'path_to_STAGE.csv'
# len(ppg): Original length of PPG signal
data, label = F.data_partition(R, len(ppg), sqi, stage_path)
```

**Tips**: **R peak detection** & **Data Partition** process must be used together and the usage  can refer to Sleep_FeatureExtraction.ipynb

### 3. Model training

#### Steps
1. edit YAML file in **config**
2. command line to training

    ```bash
    python trainer.py example.yaml > save/log/example.log
    ```
3. The result will be stored in **save**, classified according to the yaml file name
