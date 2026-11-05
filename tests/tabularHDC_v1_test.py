# synthetic_tabular_hdc_test.py
# Synthetic check only. Not evidence about the GAL recordings.
# Uses the TabularHDCClassifier as written, plus one labelled alternative.

import numpy as np
from sklearn.metrics import roc_auc_score

from tabular_hdc_user import TabularHDCClassifier   # your class, saved as this file


# 1. Make data: 32 channels of noise, 5% of frames "on" per event.
rng = np.random.default_rng(0)
N, F, E = 3000, 32, 6
X = rng.normal(0, 1, (N, F))
y = (rng.random((N, E)) < 0.05).astype(int)

# 2. Give each event a signal: its own group of 4 channels shifts by 1.5 when the event is on.
for k in range(E):
    X[y[:, k] == 1, 4 * k:4 * k + 4] += 1.5

# 3. Train on the first 2000 frames, test on the rest.
split = 2000
Xtr, Xte = X[:split], X[split:]
ytr, yte = y[:split], y[split:]


# 4. Your code as written.
model = TabularHDCClassifier(D=4000, num_features=F)
model.fit(Xtr, ytr)
probs = model.predict_proba(Xte)
auc_as_written = [roc_auc_score(yte[:, k], probs[:, k]) for k in range(E)]

# How different are the six prototypes? 0 = identical.
P = model.prototypes.astype(int)
diff = [[(P[a] != P[b]).mean() for b in range(E)] for a in range(E)]
off_diag = [diff[a][b] for a in range(E) for b in range(E) if a < b]


# 5. Alternative (not textbook HDC): compare the mean encoded vector of "on" and "off" frames.
Htr = model.encode_dataset(Xtr)
Hte = model.encode_dataset(Xte)
Btr = np.where(Htr == 1, 1, -1).astype(float)     # bipolar, for the mean only
Bte = np.where(Hte == 1, 1, -1).astype(float)
auc_alt = []
for k in range(E):
    on = ytr[:, k] == 1
    proto = np.sign(Btr[on].mean(axis=0) - Btr[~on].mean(axis=0))
    auc_alt.append(roc_auc_score(yte[:, k], Bte @ proto))


# 6. Print.
print("Prototype difference (share of differing bits), min to max:",
      f"{min(off_diag):.3f} to {max(off_diag):.3f}")
print("AUC as written, per event:", [round(a, 3) for a in auc_as_written])
print("AUC mean-difference variant, per event:", [round(a, 3) for a in auc_alt])