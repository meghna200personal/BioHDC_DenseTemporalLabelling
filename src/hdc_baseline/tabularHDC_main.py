# =========================================================================
# TABULAR HDC ON GAL, through the shared pipeline (config, data_loading, preprocess, metrics).
# Replaces the commented-out "EXAMPLE USAGE" block at the bottom of tabular_hdc.py.
# Within-subject: train on series 1-6, validate on series 7-8, one subject at a time.
# Frame-wise model: each prediction uses only its own frame, so causality is preserved.
# =========================================================================
import sys
sys.path.insert(0, r"C:\Users\MEGHNA\Documents\GitHub\ResearchProject_1\BioHDC_DenseTemporalLabelling\src")

import numpy as np
from scipy.signal import lfilter
import config
from data_loading import load_series
from preprocess import bandpass
from metrics import dense_metrics
from tabular_hdc_perceptron import TabularHDCPerceptron
from tabular_hdc import TabularHDCClassifier
SUBJECTS_TO_RUN = list(range(1, 13))#        # start with one subject; use list(range(1, 13)) for all
MODEL = "perceptron_tiebreak"           # "original" (your fit) or "perceptron" (tabular_hdc_perceptron.py) or "perceptron_tiebreak" (tabular_hdc_tiebreak.py)
FEATURE = "power"       # DEFAULT for the tabular baseline: bandpassed amplitude, one frame at a time
                            # "power" ADDS temporal context (0.25 s causal average). Use it only as a
                            # labelled extra row, never as the tabular baseline.
TRAIN_STRIDE = 50            # keep every 50th training frame; the full set does not fit in memory
PREDICT_CHUNK = 20000        # frames per prediction chunk
D = 10000
NUM_LEVELS = 21


def make_model():
    if MODEL == "perceptron":
        return TabularHDCPerceptron(D=D, num_features=config.N_CHANNELS, num_levels=NUM_LEVELS)
    if MODEL == "perceptron_tiebreak":            # ONE change: random tie-break in encoding
        from tabular_hdc_tiebreak import TabularHDCPerceptronTieBreak
        return TabularHDCPerceptronTieBreak(D=D, num_features=config.N_CHANNELS, num_levels=NUM_LEVELS)
    else:
        return TabularHDCClassifier(D=D, num_features=config.N_CHANNELS, num_levels=NUM_LEVELS)


def causal_series(subject, series):
    """Features (N, 32) and labels (N, 6) for one recording. Every step is causal."""
    X, Y = load_series(subject, series)
    # shared toggle: config.USE_BANDPASS (defaults to True if the setting is missing)
    # print("Bandpass filter applied" if getattr(config, 'USE_BANDPASS', True) else "Bandpass filter skipped")
    B = bandpass(X) if getattr(config, 'USE_BANDPASS', True) else X
    if FEATURE == "power":
        P = lfilter(np.ones(125) / 125, 1.0, B ** 2, axis=1)  # causal 0.25 s average of power
        B = np.log(P + 1e-6)
    return B.T, Y.T


def run_tabular(subject):
    # 1. training series, filtered one series at a time so no filter output crosses a join
    parts = [causal_series(subject, s) for s in config.TRAIN_SERIES]
    X_all = np.concatenate([p[0] for p in parts])
    Y_all = np.concatenate([p[1] for p in parts])

    # 2. standardise each channel with TRAINING statistics only (the quantiser expects this)
    mu, sd = X_all.mean(axis=0), X_all.std(axis=0) + 1e-8
    X_tr = ((X_all - mu) / sd)[::TRAIN_STRIDE]
    Y_tr = Y_all[::TRAIN_STRIDE]

    # 3. fit
    hdc = make_model()
    print(hdc)
    hdc.fit(X_tr, Y_tr)
    print("fitted class:", type(hdc).__module__, type(hdc).__name__,
      "| epochs_used:", getattr(hdc, "epochs_used", "none"))
    # 4. validation series, standardised with the same training statistics, predicted in chunks
    y_true, y_score = [], []
    for s in config.VAL_SERIES:
        Xv, Yv = causal_series(subject, s)
        Xv = (Xv - mu) / sd
        p = np.vstack([hdc.predict_proba(Xv[i:i + PREDICT_CHUNK])
                       for i in range(0, len(Xv), PREDICT_CHUNK)])
        y_true.append(Yv)
        y_score.append(p)
    return dense_metrics(np.vstack(y_true), np.vstack(y_score)), len(X_tr)


# if __name__ == "__main__":
print(f"model: {MODEL} | feature: {FEATURE} | D={D} | train stride {TRAIN_STRIDE} | "
        f"train series {config.TRAIN_SERIES} | validation series {config.VAL_SERIES}")
results = []
for subject in SUBJECTS_TO_RUN:
    metrics, n_train = run_tabular(subject)
    results.append(metrics)
    print(f"\nSubject {subject} (training frames used: {n_train})")
    for name in config.EVENTS:
        print(f"  {name:20s} ROC-AUC {metrics['auc/' + name]:.3f}   AP {metrics['ap/' + name]:.3f}")
    print(f"  mean ROC-AUC {metrics['auc_macro']:.3f} | mAP {metrics['mAP']:.3f}")
if len(results) > 1:
    print("\nMean over subjects: ROC-AUC %.3f | mAP %.3f" % (
        np.mean([r["auc_macro"] for r in results]), np.mean([r["mAP"] for r in results])))
print("\nUAR is not reported here: it needs a threshold chosen on training data.")