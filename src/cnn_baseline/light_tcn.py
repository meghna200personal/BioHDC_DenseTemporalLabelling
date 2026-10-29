"""Baseline: small causal temporal CNN that outputs 6 event logits at every time step.

Interface used by main.py:
    run(subject, train_series, val_series) -> (preds, info)
    preds: {series: array (N, 6)} of event probabilities at every sample (500 Hz)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from data_loading import load_series
from preprocess import bandpass, bandpass_macs

DECIM = 4                        # 500 Hz -> 125 Hz; the 7-30 Hz bandpass removes what would alias
HIDDEN = 8
KERNEL = 5
DILATIONS = (1, 2, 4, 8, 16)     # receptive field = 1 + 4 * 31 = 125 steps = 1 s at 125 Hz


class CausalConv(nn.Module): # sliding window causal convolution with dilation
    def __init__(self, cin, cout, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(cin, cout, k, dilation=d)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))          # pad the past only


class LightTCN(nn.Module):
    def __init__(self, n_ch=config.N_CHANNELS, hidden=HIDDEN, n_out=6, k=KERNEL, dilations=DILATIONS):
        super().__init__()
        self.mix = nn.Conv1d(n_ch, hidden, 1)
        self.blocks = nn.ModuleList([CausalConv(hidden, hidden, k, d) for d in dilations])
        self.head = nn.Conv1d(hidden, n_out, 1)

    def forward(self, x):                                   # (batch, 32, time)
        h = self.mix(x)
        for blk in self.blocks:
            h = h + F.relu(blk(h))
        return self.head(h)                                 # (batch, 6, time) logits


def _filtered(subject, series):
    X, Y = load_series(subject, series)
    return bandpass(X), Y


def _make_batches(Xs, Ys, rng, length=1024, batch=16):
    xb, yb = [], []
    for _ in range(batch):
        i = rng.integers(len(Xs))
        s = rng.integers(0, Xs[i].shape[1] - length)
        xb.append(Xs[i][:, s:s + length])
        yb.append(Ys[i][:, s:s + length])
    return torch.from_numpy(np.stack(xb)), torch.from_numpy(np.stack(yb))


def run(subject, train_series, val_series, steps=3000, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = [_filtered(subject, s) for s in train_series]
    cat = np.concatenate([x for x, _ in raw], axis=1)
    mean, std = cat.mean(axis=1, keepdims=True), cat.std(axis=1, keepdims=True) + 1e-8
    Xs = [((x - mean) / std)[:, ::DECIM].astype(np.float32) for x, _ in raw]
    Ys = [y[:, ::DECIM].astype(np.float32) for _, y in raw]

    rate = np.clip(np.concatenate(Ys, axis=1).mean(axis=1), 1e-4, None)
    pos_w = torch.tensor((1 - rate) / rate, dtype=torch.float32).view(1, -1, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    model = LightTCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(steps):
        xb, yb = _make_batches(Xs, Ys, rng)
        loss = loss_fn(model(xb.to(device)), yb.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  light_tcn subj {subject} step {step} loss {loss.item():.4f}")

    model.eval()
    preds = {}
    with torch.no_grad():
        for s in val_series:
            x, _ = _filtered(subject, s)
            n = x.shape[1]
            xd = ((x - mean) / std)[:, ::DECIM].astype(np.float32)
            p = torch.sigmoid(model(torch.from_numpy(xd[None]).to(device)))[0].cpu().numpy()
            preds[s] = np.repeat(p, DECIM, axis=1)[:, :n].T     # back to 500 Hz, shape (N, 6)

    params = sum(p.numel() for p in model.parameters())
    n_out = len(config.EVENTS)
    macs_per_step = (config.N_CHANNELS * HIDDEN
                     + len(DILATIONS) * HIDDEN * HIDDEN * KERNEL
                     + HIDDEN * n_out)
    macs_per_second = (macs_per_step * config.FS // DECIM
                       + config.N_CHANNELS * bandpass_macs() * config.FS)   # network + bandpass
    info = {"params": params,
            "stored_bytes": params * 4,                         # float32
            "ops_per_second": macs_per_second}
    return preds, info