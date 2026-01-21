import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

subject = 1
runs = [6]   # 先测试一个 run

files = eegbci.load_data(subject, runs)
print("Downloaded files:", files)

raw = read_raw_edf(files[0], preload=False, verbose=False)
print(raw)
print("Sampling rate:", raw.info["sfreq"])

