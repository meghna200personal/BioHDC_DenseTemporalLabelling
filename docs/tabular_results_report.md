# Tabular HDC baseline: results, checks, and what they mean

Status: real GAL data, subjects 1 to 12, within-subject protocol (train series 1-6, validate series 7-8).
Model: the original `TabularHDCClassifier`, unchanged. Feature: bandpassed amplitude, one frame at a time.

---

## 1. Summary

- The tabular baseline is weak on GAL. Mean validation ROC-AUC is 0.478 across 12 subjects, below chance.
- The pipeline is not broken. Check 3 (random prototypes) scores 0.497 to 0.509 for every subject.
- The model learns the event direction backwards. On the frames it was trained on, 11 of 12 subjects score below 0.5 (mean 0.447).
- The model adds nothing beyond a single number: how large the signal is in each frame. Its validation scores track that energy score closely for every subject.
- This is a measured limitation of frame-wise amplitude, which is the motivation for the temporal phases.

---

## 2. The pipeline, step by step

Every number in this report came from one recipe. The steps are in `tabular_gal_block.py` (`causal_series`, `run_tabular`) and `tabular_checks_all_subjects.py` (`run_subject`).

### 2.1 Load and filter each series (causal)

```python
X, Y = load_series(subject, series)            # X: (32, N) raw EEG, Y: (6, N) labels
B = bandpass(X) if config.USE_BANDPASS else X  # causal 7-30 Hz filter, uses past samples only
return B.T, Y.T                                 # (N, 32) features, (N, 6) labels
```

- Each series is filtered on its own, so no filter output crosses a join between recordings.
- The bandpass is a shared toggle. The results here were run with it on.
- Labels are not changed.

### 2.2 Standardise with training statistics only

```python
X_all = np.concatenate([p[0] for p in parts])          # series 1-6 for the subject
mu, sd = X_all.mean(axis=0), X_all.std(axis=0) + 1e-8
X_tr = ((X_all - mu) / sd)[::TRAIN_STRIDE]              # stride 50
Xv = np.vstack([(v[0] - mu) / sd for v in val])         # validation uses the SAME mu, sd
```

- The validation series never contribute to `mu` or `sd`. That keeps the causality rule.
- Standardisation matters because the quantiser assumes values near -4 to +4.

### 2.3 Fit the original model

```python
model = TabularHDCClassifier(D=10000, num_features=32, num_levels=21)
model.fit(X_tr, Y_tr)
```

Inside `fit` (your code):
- Each frame is encoded to a 10,000-bit fingerprint (`encode_row`): each channel's ID is XORed with its level vector, the 32 results are bundled by majority vote.
- For each event, every frame is **added** to a running total if the event is on and **subtracted** if off.
- The prototype is the sign of that total.

### 2.4 Score frames

Your class does this inside `predict_proba` (unchanged, from `tabular_hdc.py`):

```python
def predict_proba(self, X_test):
    H = self.encode_dataset(X_test)
    out = np.zeros((H.shape[0], self.prototypes.shape[0]))
    for i in range(H.shape[0]):
        for k in range(self.prototypes.shape[0]):
            out[i, k] = 1.0 - np.sum(H[i] != self.prototypes[k]) / self.D
    return out
```

The check scripts (`tabular_checks_all_subjects.py`) need the same scores for several prototype sets at once, so they use a vectorised copy of the same formula:

```python
def similarity(H, P):
    out = np.empty((len(H), len(P)))
    for k in range(len(P)):
        out[:, k] = 1.0 - np.sum(H != P[k], axis=1) / D
    return out
```

We confirmed the two agree exactly (maximum difference 0.0) on test data, so the checks score frames the same way your class does.

- Similarity is 1 for identical fingerprints and about 0.5 for unrelated ones.
- Scoring runs in chunks of 20,000 frames in the GAL block and the checks, to limit memory.

### 2.5 The metric

```python
roc_auc_score(y_event, scores)   # per event; 0.5 is chance, 1.0 is perfect
```

ROC-AUC is the chance that a random "on" frame scores above a random "off" frame. It needs no threshold, so it is the fair number to report here.

---

## 3. Results

### 3.1 Validation performance (v1 base run)

Source: `v1_base_results.txt`. The per-subject means match the check 1 validation column below, which confirms the two runs used the same recipe.

| Event | Mean ROC-AUC across 12 subjects | Mean AP |
|---|---|---|
| HandStart | 0.537 (validation, check run) | about 0.04 |
| FirstDigitTouch | 0.487 | about 0.04 |
| BothStartLoadPhase | 0.495 | about 0.04 |
| LiftOff | 0.477 | about 0.04 |
| Replace | 0.424 | about 0.03 |
| BothReleased | 0.450 | about 0.03 |
| **Mean** | **0.478** | **0.037** |

Chance for AP is the share of "on" frames, about 0.04 per event, so an AP near 0.04 is chance-level.

### 3.2 The three checks, all subjects

| Subject | Check 1 train | Check 1 validation | Check 2 energy | Check 3 random |
|---|---|---|---|---|
| 1 | 0.482 | 0.490 | 0.491 | 0.501 |
| 2 | 0.410 | 0.424 | 0.424 | 0.508 |
| 3 | 0.403 | 0.455 | 0.458 | 0.503 |
| 4 | 0.419 | 0.472 | 0.471 | 0.502 |
| 5 | 0.435 | 0.473 | 0.474 | 0.502 |
| 6 | 0.486 | 0.491 | 0.489 | 0.499 |
| 7 | 0.417 | 0.441 | 0.444 | 0.509 |
| 8 | 0.444 | 0.475 | 0.475 | 0.503 |
| 9 | 0.504 | 0.544 | 0.540 | 0.497 |
| 10 | 0.432 | 0.484 | 0.484 | 0.499 |
| 11 | 0.483 | 0.513 | 0.512 | 0.500 |
| 12 | 0.444 | 0.479 | 0.478 | 0.503 |
| **Mean** | **0.447** | **0.478** | **0.478** | **0.502** |
| Subjects on the wrong side of 0.5 | 11 of 12 | 10 of 12 | 10 of 12 | 0 outside 0.45-0.55 |

### 3.3 Per event, averaged across subjects

| Event | Check 1 train | Check 1 validation | Check 2 energy |
|---|---|---|---|
| HandStart | 0.490 | 0.537 | 0.538 |
| FirstDigitTouch | 0.456 | 0.487 | 0.485 |
| BothStartLoadPhase | 0.460 | 0.495 | 0.492 |
| LiftOff | 0.449 | 0.477 | 0.477 |
| Replace | 0.400 | 0.424 | 0.428 |
| BothReleased | 0.424 | 0.450 | 0.452 |

---

## 4. What each check means

### Check 3: random prototypes (the pipeline test)

```python
random_sets = [rng.integers(0, 2, size=real.shape).astype(np.uint8) for _ in range(5)]
s_val = score_all(model, Xv, [real] + random_sets)
```

Random prototypes carry no information, so they should score 0.5. They score 0.497 to 0.509 for every subject. That confirms:
- the labels line up with the frames,
- the scoring code and the train/validation split are correct.

Had this check failed, every other number would be suspect. It did not fail.

### Check 1: training AUC (is the rule learning anything?)

```python
(s_train,) = score_all(model, X_tr, [real])
auc_per_event(Y_tr, s_train)
```

Scoring the frames the prototype was built from should give at least 0.5 if the prototype points the right way. It averages 0.447. Most subjects are below 0.5 on the very frames they were fitted to. So the problem is not generalisation to new series. The prototype itself points the wrong way.

### Check 2: energy baseline (is the model adding anything?)

```python
energy = np.mean(Xv ** 2, axis=1)     # mean squared standardised amplitude per frame
```

This needs no HDC at all. It scores 0.478 on average, almost identical to the HDC validation score (0.478), and the two agree subject by subject to within about 0.01. The HDC model is, in effect, computing "how large is the signal in this frame".

Energy is also below 0.5, which tells us the direction of the effect in the features themselves: event frames have *smaller* squared amplitude than rest frames on average, not larger. That is why any score based on frame-level size points the wrong way.

---

## 5. Insights

### 5.1 Why the prototype points the wrong way (mechanism)

With about 4 to 5 percent of frames "on", the original `fit` adds about 1 frame for every 20 it subtracts. The total then reflects the average frame, not the event. Similarity to a prototype built this way measures how typical a frame is. If event frames are less typical than rest, similarity ranks them low, which is what check 1 shows.

This mechanism explains why training is also inverted. It does not require any new idea.

### 5.2 Why the energy direction is inverted (hypothesis, to test)

The energy check points the same way, so the features themselves carry the inversion. One known motor-physiology effect fits this pattern: during movement, the mu and beta rhythms over sensorimotor cortex are *suppressed*, called event-related desynchronisation (ERD). Power drops during movement, so event frames would be quieter than average.

This is a hypothesis. It predicts that a *smoothed* power measure should point the right way, because ERD is a sustained drop. Testing it requires a temporal feature, so it belongs to the next phase, not the tabular one.

### 5.3 The HDC model adds no information beyond energy

The HDC validation scores track the energy scores within about 0.01 for each subject. A frame-wise model that cannot beat the energy of the frame is not using the encoding for anything. This is the key result for the thesis motivation: the 10,000-bit encoding is not yet buying discriminative power on this feature.

### 5.4 Per event: HandStart is the exception

HandStart is the only event above 0.5 on validation (0.537). It is also the event with the earliest onset, with the gap to the next event about 350 frames (earlier analysis). This is consistent with the causal-lag observation: a frame-wise model sees the hand start when it happens, while later events are preceded by a window in which the signal has already changed. The pattern supports the temporal phases, but this is an interpretation to check, not a result.

### 5.5 Subject 9 is the only consistent subject

Subject 9 is above 0.5 on every check (0.504 train, 0.544 validation). Subjects 2, 3 and 7 are the weakest on training. Subject differences are a question for the leave-one-subject-out phase, not for this baseline.

---

## 6. Things that were wrong or changed along the way

- **Frame counts.** I earlier estimated about 720k frames for six training series. The real total is 1,185,498 frames, with series ranging from 116k to 272k. The 23,710 training frames after the stride are correct.
- **A temporal feature was briefly the default.** The `power` feature averages over 0.25 s, which is temporal context. It was switched off as the tabular default. The synthetic power results (0.69 and 0.78) are not tabular results and should not be quoted as one.
- **The perceptron result.** The synthetic gain from perceptron training (0.699 to 0.753) was measured on synthetic data, not on GAL. It has not been run on real subjects.
- **Speed.** The encoder and scorer loop over frames in Python. A 12-subject run takes a long time for this reason. Vectorising is a separate change.

---

## 7. What is not yet tested (so not claimed here)

- Whether the inversion comes from the bandpass. Not run with the bandpass off.
- Whether the perceptron training rule changes the result on real subjects. Not run on GAL.
- Whether a smoothed power measure points the right way (the ERD hypothesis). Not run.
- Tie handling. Ties still go to 0, which biases every fingerprint toward 0 (measured 44% ones instead of 50%). Not corrected.
- The quantiser range. ±4 standard deviations is a fixed choice. Not varied.
- UAR. Needs a threshold chosen on training data. Not computed.

---

## 8. Next steps, and why each one follows

| Order | Step | Why it follows from the results |
|---|---|---|
| 1 | Rerun subject 1 with `USE_BANDPASS = False` | Separates the filter from the feature. If the inversion survives without the filter, it is in the raw amplitude. |
| 2 | Rerun subject 1 with `MODEL = "perceptron"` | Tests whether a training fix can rescue the feature. Prediction from the energy result: it should not, since the features point the wrong way. A null result here is informative. |
| 3 | A causal smoothed-power energy score, no HDC (temporal) | Tests the ERD hypothesis directly. If smoothed power points the right way, the frame-wise amplitude is the problem, and temporal context is the fix. This is the motivation for the n-gram phase. |
| 4 | Vectorise the encoder and scorer | Needed before any larger experiment. A separate change, so the results stay comparable. |
| 5 | Random tie-break in encoding | Removes the 0-bias, as one separate, measurable change. |
| 6 | N-gram HDC (temporal phase) | Success criterion: beats both the tabular validation score and the causal energy baseline on the same subjects, at a measured cost in memory and operations. |

Each step changes one thing, so the effect of each can be read directly.

---

## 9. Code map, for tracing numbers back

| Number | File | Function |
|---|---|---|
| Frame counts | `tabular_checks_all_subjects.py` | `run_subject` (prints `frames:`) |
| Check 1 training AUC | `tabular_checks_all_subjects.py` | `run_subject`, `check1_train` |
| Check 2 energy | `tabular_checks_all_subjects.py` | `run_subject`, `check2_energy` |
| Check 3 random prototypes | `tabular_checks_all_subjects.py` | `run_subject`, `check3_random` |
| Validation AUC | `tabular_gal_block.py` | `run_tabular`, `dense_metrics` |
| Encoding | `tabular_hdc.py` (your class) | `encode_row`, `encode_dataset` |
| Training | `tabular_hdc.py` | `fit` |
| Scoring | `tabular_hdc.py` | `predict_proba` (checks use `similarity` in `tabular_checks_all_subjects.py`, verified identical) |
| Causal filter | `preprocess.py` | `bandpass` |
| Metric | `metrics.py` | `dense_metrics` |
