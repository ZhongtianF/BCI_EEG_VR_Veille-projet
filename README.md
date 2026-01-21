## Method 1 – Motor Imagery (CSP + LDA)

This method implements a classical EEG-based BCI pipeline using:
- Motor Imagery paradigm (left vs right hand)
- CSP for spatial feature extraction
- LDA for binary classification

Steps:
1. Download EEG Motor Imagery dataset (PhysioNet)
2. Extract events and epochs (T1 / T2)
3. Train and evaluate CSP + LDA classifier

Run:
```bash
python step1_download_eeg.py
python step2_events.py
python step3_csp_lda.py
