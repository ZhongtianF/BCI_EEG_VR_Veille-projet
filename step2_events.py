import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws

subject = 1
runs = [6, 10]

files = eegbci.load_data(subject, runs)
raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
raw = concatenate_raws(raws)

# ✅ 关键：清理通道名（去掉末尾的点/多余符号）
raw.rename_channels(lambda ch: ch.strip().replace('.', ''))

# ✅ 可选：只保留 EEG 通道（更干净）
raw.pick_types(eeg=True)

# ✅ 关键：montage 设置为 ignore missing，避免再报错
raw.set_montage("standard_1020", on_missing="ignore")

events, event_id = mne.events_from_annotations(raw)

print("event_id =", event_id)
print("events shape =", events.shape)
print("channel examples =", raw.ch_names[:10])
