# src/models/logreg_csp_7class.py
"""7-Class Baseline: causal bandpass -> CSP -> smoothed log power -> 7 logistic regressions (6 events + 1 Background).

Interface used by main_7class.py:
    run(subject, train_series, val_series) -> (preds, info)
    preds: {series: array (N, 7)} of event probabilities at every sample (500 Hz)
"""
import numpy as np
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression
from mne.decoding import CSP

import config
from data_loading import load_series
from preprocess import bandpass, bandpass_macs

N_FILTERS = 4
SMOOTH_WIN = 250                 # 0.5 s causal window at 500 Hz
SUBSAMPLE = 10                   # train on every 10th sample
EPOCH_LEN = 2 * config.FS        # CSP epochs: 2 s after / before HandStart


def _features(Xf, W):
    """Xf: filtered (32, N). Returns (4, N): log of causally smoothed CSP power."""
    power = (W @ Xf) ** 2
    smooth = lfilter(np.ones(SMOOTH_WIN), 1.0, power, axis=1)
    return np.log(np.maximum(smooth, 1e-30))


def _csp_epochs(Xf, Y):
    onsets = np.where(np.diff(Y[0], prepend=0) == 1)[0]     # HandStart rising edges
    during, rest = [], []
    for o in onsets:
        if o - EPOCH_LEN >= 0 and o + EPOCH_LEN <= Xf.shape[1]:
            during.append(Xf[:, o:o + EPOCH_LEN])
            rest.append(Xf[:, o - EPOCH_LEN:o])
    return during, rest


def check_causality(T0=12000, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 50, (config.N_CHANNELS, n)) * 1e-6
    W = rng.normal(size=(N_FILTERS, config.N_CHANNELS))
    X2 = X.copy()
    X2[:, T0:] += rng.normal(0, 500e-6, (config.N_CHANNELS, n - T0))
    f1 = _features(bandpass(X), W)
    f2 = _features(bandpass(X2), W)
    before_same = np.allclose(f1[:, :T0], f2[:, :T0])
    after_changed = not np.allclose(f1[:, T0:], f2[:, T0:])
    ok = bool(before_same and after_changed)
    print(f"[causality] logreg_csp_7class: {'PASS' if ok else 'FAIL'} "
          f"(before T0 unchanged: {before_same}, after T0 changed: {after_changed})")
    return ok


def run(subject, train_series, val_series):
    train, during, rest = [], [], []
    for s in train_series:
        X, Y6 = load_series(subject, s)
        Xf = bandpass(X * 1e-6)
        d, r = _csp_epochs(Xf, Y6)
        during += d
        rest += r
        
        # Explicit 7th target: Background = 1 where all 6 events are 0
        bg = (Y6.sum(axis=0) == 0).astype(np.int8)[None, :]
        Y7 = np.vstack([Y6, bg])
        train.append((Xf, Y7))

    X_ep = np.stack(during + rest)
    y_ep = np.array([1] * len(during) + [-1] * len(rest))
    csp = CSP(n_components=N_FILTERS, reg="oas")
    csp.fit(X_ep, y_ep)
    W = csp.filters_[:N_FILTERS]

    F = np.concatenate([_features(Xf, W) for Xf, _ in train], axis=1)
    L = np.concatenate([Y7 for _, Y7 in train], axis=1)
    
    clfs = []
    for i in range(7):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(F[:, ::SUBSAMPLE].T, L[i, ::SUBSAMPLE])
        clfs.append(clf)

    preds = {}
    for s in val_series:
        X, _ = load_series(subject, s)
        Fv = _features(bandpass(X * 1e-6), W)
        preds[s] = np.stack([c.predict_proba(Fv.T)[:, 1] for c in clfs], axis=1) # (N, 7)

    params = N_FILTERS * config.N_CHANNELS + 7 * (N_FILTERS + 1)
    macs_per_sample = (config.N_CHANNELS * bandpass_macs() + N_FILTERS * config.N_CHANNELS + 7 * N_FILTERS)
    info = {"params": params,
            "stored_bytes": params * 4,
            "ops_per_second": macs_per_sample * config.FS}
    return preds, info