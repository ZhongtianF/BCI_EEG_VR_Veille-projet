import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from mne.decoding import CSP

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1) Load data
subject = 1
runs = [6, 10]
files = eegbci.load_data(subject, runs)
raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
raw = concatenate_raws(raws)

# 2) Clean channel names + keep EEG
raw.rename_channels(lambda ch: ch.strip().replace('.', ''))
raw.pick_types(eeg=True)

# 3) Basic preprocessing
raw.filter(8., 30., fir_design="firwin", verbose=False)  # MI band
raw.set_montage("standard_1020", on_missing="ignore")

# 4) Events: keep only T1 vs T2
events, event_id = mne.events_from_annotations(raw)
event_id_use = {"T1": event_id["T1"], "T2": event_id["T2"]}

# 5) Epoching (time window)
# 这个窗口可以后面再调，先用 0~4s 跑通
tmin, tmax = 0.0, 4.0
picks = mne.pick_types(raw.info, eeg=True, exclude="bads")

epochs = mne.Epochs(
    raw, events, event_id=event_id_use,
    tmin=tmin, tmax=tmax,
    picks=picks, baseline=None, preload=True, verbose=False
)

X = epochs.get_data()                # (n_trials, n_channels, n_times)
y = epochs.events[:, -1]             # labels are numeric (2 or 3)
print("X shape:", X.shape, "y shape:", y.shape)
print("Classes in y:", np.unique(y))

# 6) CSP + LDA pipeline
csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
lda = LDA()
clf = Pipeline([("CSP", csp), ("LDA", lda)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
print("Accuracy (5-fold):", scores)
print("Mean accuracy:", scores.mean(), "+/-", scores.std())

# 7) Confusion matrix
y_pred = cross_val_predict(clf, X, y, cv=cv)
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix (CSP + LDA)")
plt.show()
