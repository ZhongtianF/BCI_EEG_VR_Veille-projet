import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

from sklearn.cross_decomposition import CCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ============================================================
# PARAMÈTRES GLOBAUX
# ============================================================

subjects = [
    ('data_s19_64.npy', 'data_s19_64.mat'),
    ('data_s20_64.npy', 'data_s20_64.mat'),
    ('data_s21_64.npy', 'data_s21_64.mat'),
    ('data_s22_64.npy', 'data_s22_64.mat'),
]

sample_rate = 1000
num_harmonics = 2

condition_index = 0      # Low-Depth
blocks = range(12)       # 12 répétitions

# Fréquences testées (comme ton code qui marche)
tested_frequencies = [8, 10, 12, 14, 16]
frequency_indices  = [7, 9, 11, 13, 15]

# ============================================================
# GÉNÉRATION DES SIGNAUX DE RÉFÉRENCE
# ============================================================

def generate_reference_signals(frequencies, num_samples, sample_rate, num_harmonics=1):
    t = np.arange(num_samples) / sample_rate
    references = []

    for f in frequencies:
        for n in range(1, num_harmonics + 1):
            references.append(np.sin(2 * np.pi * n * f * t))
            references.append(np.cos(2 * np.pi * n * f * t))

    return np.array(references).T


# ============================================================
# PRÉDICTION CCA (INCHANGÉE)
# ============================================================

def predict_frequency_cca(X_reshaped, frequencies, num_samples, sample_rate, num_harmonics=2):
    correlations = []

    for freq in frequencies:
        ref = generate_reference_signals(
            [freq], num_samples, sample_rate, num_harmonics
        )

        cca = CCA(n_components=1)
        cca.fit(ref, X_reshaped)
        X_c, Y_c = cca.transform(ref, X_reshaped)

        corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
        correlations.append(corr)

    predicted_freq = frequencies[np.argmax(correlations)]
    return predicted_freq, correlations


# ============================================================
# TRUE VS PREDICTED ÉLARGI
# ============================================================

y_true = []
y_pred = []

print("\n===== CLASSIFICATION CCA (MULTI-SUJETS / MULTI-BLOCS) =====\n")

for npy_file, mat_file in subjects:

    print(f"Chargement : {mat_file}")

    if os.path.exists(npy_file):
        data = np.load(npy_file)
    else:
        with h5py.File(mat_file, 'r') as f:
            data = f['datas'][:]

    # data shape utilisée ici :
    # (block, frequency, time, channel, condition)

    for block_index in blocks:
        for true_freq, freq_idx in zip(tested_frequencies, frequency_indices):

            # Extraction EEG
            X = data[block_index, freq_idx, :, :, condition_index]
            num_samples = X.shape[0]

            # Reshape identique à ton code
            X_reshaped = X.reshape(-1, X.shape[-1])

            # Prédiction
            pred_freq, corr_scores = predict_frequency_cca(
                X_reshaped,
                tested_frequencies,
                num_samples,
                sample_rate,
                num_harmonics
            )

            y_true.append(true_freq)
            y_pred.append(pred_freq)

# ============================================================
# MÉTRIQUES
# ============================================================

accuracy = accuracy_score(y_true, y_pred)
print(f"\nAccuracy globale CCA : {accuracy*100:.2f} %")
print(f"Nombre total d'échantillons : {len(y_true)}")

# ============================================================
# MATRICE DE CONFUSION
# ============================================================

cm = confusion_matrix(y_true, y_pred, labels=tested_frequencies)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=tested_frequencies
)

#plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("True vs Predicted Frequencies (SSVEP-CCA)")
plt.grid(False)
plt.show()
