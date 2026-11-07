# TabularHDC: original prototype rule vs perceptron-style rule

Status: synthetic test only. Not run on GAL. Compare both on the same within-subject split before drawing conclusions.

## 1. What is being compared

Two classes that share one encoder. Only the prototype rule differs, so any difference in results comes from training alone.

| Piece | Original `TabularHDCClassifier` | `TabularHDCPerceptron` |
|---|---|---|
| Feature IDs, level memory (CIM) | as written | inherited, unchanged |
| Quantisation (fixed ±4, 21 levels) | as written | inherited, unchanged |
| Binding (XOR) and bundling (sum > 16) | as written, ties go to 0 | inherited, unchanged |
| Prototype start | sum over all frames: +1 when event on, -1 when off | sum over positive frames only |
| Prototype updates | one pass, no updates after that | up to 5 epochs; add mistaken positives, subtract mistaken negatives |
| Stopping | none | stop early when there are no mistakes, or after `max_epochs` |
| Prototype sign | `> 0`, so zero becomes 0 | `sign` with a seeded random tie break |
| Similarity | 1 minus normalised Hamming | inherited, unchanged |
| Threshold | none in the class | `training_thresholds()` (Youden, training data only), for UAR and confusion matrices |

## 2. Why the two rules behave differently

The original rule subtracts every negative frame, even those the prototype already rejects. With about 5% positives, the negatives dominate the sum, so the prototype drifts toward the average frame.

The perceptron-style rule starts from the positives and updates only on mistakes. A correctly rejected negative causes no update. The prototype is changed only by frames it gets wrong, so the negatives cannot dominate it.

Caveat on the threshold: the mistake test uses a fixed line at 0. Moving that line changes which frames count as mistakes, so the result depends on the threshold. The class fixes it at 0 for training and uses a separate training-data threshold only for decisions.

## 3. Synthetic results (single seed, D = 4000, 2000 train / 1000 test frames)

| Model | AUC per event | Mean AUC |
|---|---|---|
| Original (subtract all) | 0.762, 0.681, 0.711, 0.638, 0.680, 0.718 | 0.699 |
| Perceptron-style | 0.800, 0.748, 0.740, 0.714, 0.732, 0.785 | 0.753 |

These are not evidence about GAL. The synthetic events have a strong planted signal, so both rules are judged on an easy problem. A single seed can also move each number by a few hundredths.

## 4. Things the perceptron version still gets wrong or leaves open

1. **Epoch cap decides the result.** In the test, every event used all 5 epochs without reaching zero mistakes (`epochs_used` = 5 for all). The stopping rule did not fire. Choose `max_epochs` on a validation split, not on the test series.
2. **Threshold is fixed at 0 during training.** The mistake set depends on that threshold, and changing it changes the prototype. A threshold that is tuned during training needs its own validation.
3. **Cost.** The original fit is one pass. The perceptron version makes up to `max_epochs` passes, each a full dot product over the training frames. Training is offline, so this is acceptable by Peter's rule, but record it.
4. **Not textbook one-shot HDC.** It is retraining, which the notebook also does. Describe it that way in the thesis.
5. **Per-event rules are independent.** Overlapping events are handled naturally, since a frame can be a mistake for one event and not another.

## 5. Things deliberately left unchanged

These are real issues, but they are separate changes. Change one thing per run.

- **Tie handling in encoding.** The 32-channel sum ties 14% of bits, and they go to 0. Expected share of 1-bits is 0.43, and it was measured at 0.44. Fix as a separate run: random tie break in `encode_row`.
- **Quantiser range.** Fixed ±4 assumes standardised input. For EEG, standardise each channel with training statistics before encoding. Test ±3 and ±4 on training data first, and report the share clipped.
- **Memory.** Encoded frames are stored as unpacked bytes (10 KB each, against 1.25 KB packed).

## 6. How to run the comparison (one change at a time)

1. Save your original class as `tabular_hdc.py` next to `tabular_hdc_perceptron.py`.
2. Use the same split as the logistic regression and the CNN: train on series 1-6, validate on 7-8, one subject at a time.
3. Fit both models on the same training frames.
4. Report, per event and as a mean: ROC-AUC, mAP, and UAR (threshold from training data).
5. Record cost: `D`, encoding time, training time, prediction time, and stored bytes for both.
6. Choose `max_epochs` on series 5-6 (inside training), then score series 7-8 once.

## 7. Next runs, in order

1. Run this comparison on subject 1 with GAL data. Record epochs used per event.
2. Add the random tie break to the shared encoder, and rerun both. This measures the tie effect alone.
3. Standardise the input with training statistics, and rerun both.
