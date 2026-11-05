# src/models/light_tcn_7class.py
"""7-Class Baseline: small causal temporal CNN outputting 7 event logits at every time step.

Interface used by main_7class.py:
    run(subject, train_series, val_series) -> (preds, info)
    preds: {series: array (N, 7)} of event probabilities at every sample (500 Hz)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from data_loading import load_series
from preprocess import bandpass, bandpass_macs

DECIM = 4                        # 500 Hz -> 125 Hz
HIDDEN = 8
KERNEL = 5
DILATIONS = (1, 2, 4, 8, 16)


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(cin, cout, k, dilation=d)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class LightTCN7Class(nn.Module):
    def __init__(self, n_ch=config.N_CHANNELS, hidden=HIDDEN, n_out=7, k=KERNEL, dilations=DILATIONS):
        super().__init__()
        self.mix = nn.Conv1d(n_ch, hidden, 1)
        self.blocks = nn.ModuleList([CausalConv(hidden, hidden, k, d) for d in dilations])
        self.head = nn.Conv1d(hidden, n_out, 1) # Outputs 7 logits

    def forward(self, x):
        h = self.mix(x)
        for blk in self.blocks:
            h = h + F.relu(blk(h))
        return self.head(h)


def _filtered_7class(subject, series):
    X, Y6 = load_series(subject, series)
    bg = (Y6.sum(axis=0) == 0).astype(np.int8)[None, :]
    Y7 = np.vstack([Y6, bg])
    return bandpass(X), Y7


def _make_batches(Xs, Ys, rng, length=1024, batch=16):
    xb, yb = [], []
    for _ in range(batch):
        i = rng.integers(len(Xs))
        s = rng.integers(0, Xs[i].shape[1] - length)
        xb.append(Xs[i][:, s:s + length])
        yb.append(Ys[i][:, s:s + length])
    return torch.from_numpy(np.stack(xb)), torch.from_numpy(np.stack(yb))


def _predict_probs(model, xf, mean, std, n_raw, device):
    xd = ((xf - mean) / std)[:, ::DECIM].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(xd[None]).to(device)))[0].cpu().numpy()
    return np.repeat(p, DECIM, axis=1)[:, :n_raw].T        # Shape (N, 7)


def check_causality(T0=3000, n=6000, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LightTCN7Class().eval()
    mean, std = np.zeros((config.N_CHANNELS, 1)), np.ones((config.N_CHANNELS, 1))
    X = rng.normal(0, 50, (config.N_CHANNELS, n))
    X2 = X.copy()
    X2[:, T0:] += rng.normal(0, 500, (config.N_CHANNELS, n - T0))
    p1 = _predict_probs(model, bandpass(X), mean, std, n, "cpu")
    p2 = _predict_probs(model, bandpass(X2), mean, std, n, "cpu")
    before_same = np.allclose(p1[:T0], p2[:T0], atol=1e-6)
    after_changed = not np.allclose(p1[T0:], p2[T0:], atol=1e-6)
    ok = bool(before_same and after_changed)
    print(f"[causality] light_tcn_7class: {'PASS' if ok else 'FAIL'} "
          f"(before T0 unchanged: {before_same}, after T0 changed: {after_changed})")
    return ok


def run(subject, train_series, val_series, steps=3000, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = [_filtered_7class(subject, s) for s in train_series]
    cat = np.concatenate([x for x, _ in raw], axis=1)
    mean, std = cat.mean(axis=1, keepdims=True), cat.std(axis=1, keepdims=True) + 1e-8
    Xs = [((x - mean) / std)[:, ::DECIM].astype(np.float32) for x, _ in raw]
    Ys = [y[:, ::DECIM].astype(np.float32) for _, y in raw]

    rate = np.clip(np.concatenate(Ys, axis=1).mean(axis=1), 1e-4, None)
    pos_w = torch.tensor((1 - rate) / rate, dtype=torch.float32).view(1, -1, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    model = LightTCN7Class().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(steps):
        xb, yb = _make_batches(Xs, Ys, rng)
        loss = loss_fn(model(xb.to(device)), yb.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  light_tcn_7class subj {subject} step {step} loss {loss.item():.4f}")

    model.eval()
    preds = {}
    for s in val_series:
        x, _ = _filtered_7class(subject, s)
        preds[s] = _predict_probs(model, x, mean, std, x.shape[1], device)

    params = sum(p.numel() for p in model.parameters())
    macs_per_step = (config.N_CHANNELS * HIDDEN
                     + len(DILATIONS) * HIDDEN * HIDDEN * KERNEL
                     + HIDDEN * 7)
    macs_per_second = (macs_per_step * config.FS // DECIM + config.N_CHANNELS * bandpass_macs() * config.FS)
    info = {"params": params,
            "stored_bytes": params * 4,
            "ops_per_second": macs_per_second}
    return preds, info