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
```

## Method 2 – SSVEP Frequency Recognition (CCA)

This method implements a classical SSVEP-based BCI pipeline using:
- Steady-State Visual Evoked Potentials (SSVEP)
- Canonical Correlation Analysis (CCA) for frequency recognition
- Multi-subject and multi-block EEG evaluation

The method estimates the stimulus frequency by computing canonical correlations
between recorded EEG signals and sinusoidal reference signals at predefined target
frequencies and their harmonics.

### Dataset
The EEG data used in this project come from the following open-access dataset:

**An open dataset for human SSVEPs in the frequency range of 1–60 Hz**  
Available at:  
https://springernature.figshare.com/collections/An_open_dataset_for_human_SSVEPs_in_the_frequency_range_of_1-60_Hz/6752910/1

The dataset includes multi-subject SSVEP recordings with multiple stimulus
frequencies and repeated trials, suitable for frequency recognition experiments.

### Steps
1. Load multi-subject SSVEP EEG data
2. Select stimulus condition, blocks, and target frequencies
3. Generate sinusoidal reference signals (fundamental and harmonics)
4. Apply CCA to compute correlation scores
5. Predict stimulus frequency using maximum correlation
6. Evaluate performance using accuracy and confusion matrix

### Run
```bash
python SSVEP-CCA/cca_TrueVsPredicted.py
```