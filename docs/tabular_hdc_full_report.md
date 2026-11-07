# Tabular HDC: full report (results, bugs, insights, next steps, what to study)

Written in plain language. Read top to bottom once. Each section says where a number came from.

---

## 1. The short version

- The frame-by-frame amplitude model is weak on the GAL data. Averaged over 12 subjects, its score is 0.478, below chance.
- The pipeline is sound. A test with random prototypes scores 0.50 for every subject.
- The model learns the event direction the wrong way round. On the training frames, 11 of 12 subjects score below 0.5.
- The model adds nothing beyond a simple "how strong is the signal" number.
- Changing the training rule (original vs perceptron) does not change the result for amplitude, because the features give the rule no usable direction.
- Using a smoothed power feature removes the inversion and brings the average to about chance (0.508). But power looks back in time, so it is not a tabular result. It belongs to the next phase.

---

## 2. Settings used

| Run | Model | Feature | Bandpass | Train stride | Train series | Validation series |
|---|---|---|---|---|---|---|
| v1 base | original | amplitude | on | 50 | 1-6 | 7-8 |
| checks (12 subjects) | original | amplitude | on | 50 | 1-6 | 7-8 |
| perceptron, amplitude | perceptron | amplitude | on | 50 | 1-6 | 7-8 |
| perceptron, power | perceptron | power | on | 50 | 1-6 | 7-8 |

All runs use D = 10,000 bits, 21 levels, and the same causal filter. Every run is within-subject (one model per subject).

---

## 3. Results

### 3.1 Original model, amplitude, validation (v1 base run)

Each number is the ROC-AUC on validation series 7-8. 0.5 is chance, 1.0 is perfect.

| Event | Mean over 12 subjects |
|---|---|
| HandStart | 0.537 |
| FirstDigitTouch | 0.487 |
| BothStartLoadPhase | 0.495 |
| LiftOff | 0.477 |
| Replace | 0.424 |
| BothReleased | 0.450 |
| **Overall** | **0.478** |

Average precision (AP) is about 0.04 for every event. That is the chance level for these events, because only about 4% of frames are "on".

### 3.2 The three checks (original model, amplitude, all 12 subjects)

| Subject | Check 1: training | Check 2: energy only | Check 3: random prototypes |
|---|---|---|---|
| 1 | 0.482 | 0.491 | 0.501 |
| 2 | 0.410 | 0.424 | 0.508 |
| 3 | 0.403 | 0.458 | 0.503 |
| 4 | 0.419 | 0.471 | 0.502 |
| 5 | 0.435 | 0.474 | 0.502 |
| 6 | 0.486 | 0.489 | 0.499 |
| 7 | 0.417 | 0.444 | 0.509 |
| 8 | 0.444 | 0.475 | 0.503 |
| 9 | 0.504 | 0.540 | 0.497 |
| 10 | 0.432 | 0.484 | 0.499 |
| 11 | 0.483 | 0.512 | 0.500 |
| 12 | 0.444 | 0.478 | 0.503 |
| **Mean** | **0.447** | **0.478** | **0.502** |

What each check asks:
- **Check 1 (training):** does the model score the frames it learned from above chance? It mostly does not.
- **Check 2 (energy):** can one simple number, the average squared signal per frame, do as well? It nearly does.
- **Check 3 (random):** does a random model score 0.5? It does, which proves the pipeline is not broken.

### 3.3 Perceptron training rule, amplitude

The perceptron run gave the same numbers as the original, to three decimals, for every subject. We checked why:
- The class did run. It used all 5 training rounds for every event (`epochs_used: [5 5 5 5 5 5]`).
- On data of the same size and rarity, the two rules produce prototypes that differ in only 0.2% of bits. Their scores differ by at most 0.003.
- So with a weak amplitude signal, the rule barely changes the result. The features are the bottleneck, not the rule.

### 3.4 Perceptron training rule, power feature (temporal, not tabular)

Per event, mean over 12 subjects, with the amplitude original alongside:

| Event | Power + perceptron | Amplitude + original |
|---|---|---|
| HandStart | 0.509 | 0.537 |
| FirstDigitTouch | 0.515 | 0.487 |
| BothStartLoadPhase | 0.514 | 0.495 |
| LiftOff | 0.496 | 0.477 |
| Replace | 0.507 | 0.424 |
| BothReleased | 0.505 | 0.450 |
| **Overall** | **0.508** | **0.478** |

- 7 of 12 subjects are above 0.5 with power. Subject 8 (0.542) and subject 2 (0.534) are the best. Subject 4 is the lowest (0.485).
- The big change is on Replace and BothReleased, which moved from about 0.43 to about 0.51. These are the events that were most inverted with amplitude.
- HandStart is the one event that did better with amplitude (0.537 against 0.509). A plausible reason is that power is an average over the past, so it lags the onset. This is a hypothesis to check.
- The gains are small. Averages around 0.51 are only slightly better than chance.

---

## 4. Insights

1. **Frame-level amplitude carries almost no event information.** The energy score (one number per frame) matches the HDC model. The HDC encoding adds nothing on top of it.

2. **The direction is inverted for amplitude.** Event frames have smaller squared amplitude than rest frames. One possible reason is a known movement effect: the mu and beta rhythms over the motor cortex weaken during movement. This is a hypothesis, not a result.

3. **Smoothing over time changes the direction.** Using power averaged over 0.25 s removes most of the inversion. This suggests the event information is in the rhythm's strength over a short period, not in any single frame. That is the key reason to build a temporal model.

4. **Smoothing has a lag.** A causal average cannot rise until after the event starts. HandStart is the earliest event, so it suffers most. This is a property of looking only at the past, not a code error.

5. **The training rule matters only when the signal is strong.** With weak signal, the original and perceptron rules give the same answer. We have not yet found the signal strength where they separate. That is the purpose of the planned signal-strength sweep.

6. **Random prototypes score 0.5.** This is evidence the labels, the alignment, and the split are correct.

7. **Subjects differ.** Subject 9 is above 0.5 on every check with amplitude. Subject 8 is the best with power. Differences between people are a question for the leave-one-subject-out phase.

---

## 5. Bugs, mistakes, and corrections (all found along the way)

| # | What | Status |
|---|---|---|
| 1 | Early frame-count estimate of about 720k was wrong. The real total is 1,185,498 training frames across series 1-6. | Corrected |
| 2 | The power feature was briefly the tabular default. Power averages over the past, so it is temporal. | Fixed: default is amplitude |
| 3 | The "similarity" code in the report was shown as if it were in the class. It is a vectorised copy of your `predict_proba`. | Fixed: verified identical (difference 0.0) |
| 4 | I once said the perceptron results were identical because of a wiring bug. They are not. The class runs; the rule just barely changes the prototypes with weak signal. | Corrected |
| 5 | A synthetic test with bandpass "on" used a constant offset, which the bandpass removes. A 10 Hz burst was used instead. | Fixed in later tests |
| 6 | Ties in encoding go to 0. Fourteen percent of bits are ties, so about 44% of bits are 1 instead of 50%. | Open: fix as a separate run |
| 7 | The ±4 range assumes standardised input. Min/max or standardise-and-clip are alternatives. | Open |
| 8 | The perceptron hit the 5-round limit for every event. It never converged. | Open: choose the limit on validation data |
| 9 | Your original `fit` builds a large int64 matrix. At stride 50 this is about 1.2 GB. Stride 10 would need about 6 GB. | Known limit |
| 10 | The encoder and scorer loop over frames in Python. A 12-subject run takes a long time. | Open: vectorise (separate change) |
| 11 | The causal bandpass prints "Bandpass filter applied" for every series. Harmless, but noisy. | Cosmetic |
| 12 | Notebook issues: the early-stop test `np.any(accs_log) > 0.99` tests a boolean; `best_class_prototypes` can be undefined; cell 40 passes a list to `evaluate_HDC`; cell 43 saves the wrong prototypes; a function is duplicated. | Noted, not yet fixed |
| 13 | UAR is not computed. It needs a threshold chosen on training data. | Open |

---

## 6. Points we discussed (decisions and guidance)

From Peter:
- Keep the baselines and HDC in a similar accuracy and cost range.
- Use leave-one-subject-out as the main protocol later, with within-subject as a reference.
- Inference must be causal: only the present and past.
- Report overall multi-class performance, not just per-event numbers, including a "no event" class.
- Treat bandpass and CSP as domain knowledge; log which steps need it.
- Use a two-phase plan: compare existing methods first, then build a targeted method.
- Measure accuracy, memory, inference operations, and streaming behaviour.

From our own work:
- Two kinds of cost: bit operations (for HDC) and multiply-adds (for the neural models). Report them separately.
- Each model shares the same split and metrics: ROC-AUC, mAP, and UAR later.
- The bandpass is a shared toggle (`config.USE_BANDPASS`). The split is shared.
- Labels overlap. About 26.5% of "on" frames have two or more events in one series. Six independent outputs are the right design.
- The tabular model is frame-wise by definition. Any temporal feature is an extra variant and must be labelled that way.

---

## 7. Next steps (in order, one change per run)

1. **Vectorise the encoder and scorer.** Check that the new version gives identical vectors on a small sample first. This is speed only.
2. **Random tie-break in encoding.** One change: ties go to a random bit instead of 0. Compare with the current result.
3. **Bandpass off, amplitude, original.** Tests whether the inversion comes from the filter.
4. **Causal energy score with smoothing, no HDC.** Tests whether the smoothed direction is correct without any encoding. Label as temporal.
5. **Signal-strength sweep.** Plant signals of different strengths in fake data. Find where the original and perceptron rules start to differ.
6. **Quantiser range.** Compare ±4 with standardise-and-clip, using training statistics only.
7. **Run the three checks for the power feature**, to see whether the random floor and energy comparison hold there too.
8. **Start the n-gram (temporal) phase.** Success measure: beats the tabular amplitude model and the causal energy baseline, on the same subjects, at a measured cost.

Each step should keep everything else fixed, so each result can be read on its own.

---

## 8. What to study or think about

Concepts:
- **Mu and beta rhythms, and event-related desynchronisation (ERD).** Why does movement make these rhythms weaker, and how is that visible in power?
- **Causal filters and lag.** Why does a past-only average arrive late, and how does that affect early events?
- **One-vs-rest training.** Why do the negatives drag a prototype toward the average frame?
- **Hamming similarity and BSC vs MAP.** What does a similarity of 0.5 mean, and how do ties change it?
- **ROC-AUC, mAP, and UAR.** Why is AP at chance near 0.04? When is each metric the right one?

Questions to ponder:
- If the energy score matches the HDC score, what does the HDC encoding add? Is it supposed to add anything at the frame level?
- Is the event information in the signal's strength over time, or in its shape? What would the answer change in the design?
- At what signal strength do the two training rules separate? Does the real data sit below or above that point?
- Overlapping events: does the representation fail to tell them apart, or does the decision rule fail to separate them? Each needs a different fix.
- Which features should a tabular model see if it must stay frame-wise?
- For the thesis: is the research question about HDC's encoding, its training, or its cost? The answer decides which runs matter most.
- How should a "no event" class enter the evaluation, given that most frames are quiet?

Reading:
- Papers on event-related desynchronisation and motor imagery in EEG.
- Multi-label event detection in sensor streams, to see how overlap is handled elsewhere.
- Non-linear and lighter encoders for HDC, if the kernel route is kept as an option.

---

## 9. Where each file is

| File | What it holds |
|---|---|
| `docs/tabular_hdc_explained.md` | Plain-language explanation of all tabular code |
| `docs/tabular_results_report.md` | Earlier report on the amplitude results and checks |
| `docs/tabular_hdc_comparison.md` | Original vs perceptron rules, side by side |
| `docs/tabular_hdc_full_report.md` | This document |
| `code/tabular_gal_block.py` | Runs the tabular model on GAL through the shared pipeline |
| `code/tabular_hdc_perceptron.py` | Perceptron-style subclass |
| `code/tabular_checks.py` | The three checks, one subject |
| `code/tabular_checks_all_subjects.py` | The three checks, all subjects, with a summary |
| `v1_base_results.txt` | Original model results, amplitude, 12 subjects |
