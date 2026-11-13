# Research direction notes

Prepared for the 4 pm meeting with Peter. Written in plain language.
Status: working notes. Nothing here is a finding unless it says "measured".

---

## 1. Where the thesis stands

- **Problem.** For every sample of 32-channel EEG, say which of six overlapping movement events is happening, if any. Events are short (about 150 ms) and rare (about 4% of frames each).
- **Constraint.** Predictions must be causal: only the present and past. Peter also wants a cost measure (bit operations for HDC, multiply-adds for the neural models).
- **Two-phase plan (Peter).** Phase 1 compares existing methods. Phase 2 builds a targeted method from what Phase 1 shows.
- **Measured so far.** A frame-wise binary HDC baseline scores about 0.48 mean ROC-AUC across 12 subjects. This is below chance. A simple energy score does as well, so the encoding adds nothing at frame level. Power, which is temporal, reaches about 0.51 on average.

---

## 2. Research questions we have posed

### 2.1 Phase 1 question (existing methods)

> How far can a frame-by-frame binary HDC model detect the six grasp-and-lift movement events, and how much does adding temporal context change that?

**What it asks.** For each sample, which event (if any) is happening. The answer is a table and a curve, not yes or no.

**Parts.**
- *The task.* Six overlapping events, short and rare.
- *The method.* Binary HDC, frame-wise (each sample alone) and temporal (with a history of past samples, the n-gram phase).
- *"How far."* ROC-AUC and mAP per event and on average, across subjects, with the failure cases named.
- *"How much does temporal context change that."* The gap between frame-wise and temporal results, on the same subjects, same split, same metrics.

**Status.** Frame-wise half measured. Temporal half is the plan.

**What "temporal context" means here.** Information from past samples, carried into the prediction. Frame-wise uses none. Power uses a 0.25 s average of the past, which is temporal. The n-gram phase uses a window of past frames, encoded with permutation.

### 2.2 Sub-questions we have discussed

**(a) Wave drop versus spike.**

> Does binary HDC encode a gradual drop in rhythm power differently from a sudden burst, and does that difference help separate movement from artifact?

- *Suggestion.* "Spike" is ambiguous. It can mean a sudden jump (often artifact, such as the F8/T8 bursts from our EDA) or a fast event. "Wave drop" matches what we see: power falls during movement (event-related desynchronisation, ERD).
- *What to define before the meeting.* What result counts as a difference, for example a clear gap in event scores or in how similar fingerprints stay as level changes to begin with, I think.

**(b) Robustness to noise.**

> To what extent, in terms of signal-to-noise ratio (SNR), can an HDC model handle noise without losing accuracy or raising computation cost?

- *Suggestion.* Define the compromise: for example, a drop in mean ROC-AUC of more than 0.02 from the clean signal.
- *Cost.* Work per frame does not change with noise, so report cost once, next to the baselines.
- *Test.* Add Gaussian noise at several SNR levels to validation frames, and plot ROC-AUC against SNR for HDC, logistic regression, and the TCN.
- *Fit.* Peter placed robustness in the later phase.

### 2.3 Main research question (still open)

The Phase 1 question is not the thesis's novelty. The main question has to say what you will build. Three candidates came from our discussion, each with a literature check still needed.

1. **Overlap-aware prototypes.** Keep overlapping events apart in the prototypes, with a training rule that does this. Overlap is common in our data (26.5% of "on" frames have two or more events).
2. **Encoding rhythm, not samples.** Encode the strength of each channel's rhythm over time. Frame-wise amplitude carries almost no event information. Rhythm power does, which targets the gap we measured.
3. **Streaming HDC with a cost guarantee.** Update the window incrementally, so each new sample costs a fixed amount. This matches Peter's real-time requirement.

*Caution.* A combination of TempHD and multi-centroid was in your proposal. Peter was cautious about hybrids, and there is no evidence for it yet. Keep it as a possible later step.

---

## 3. Your ideas from the premise discussion, and my suggestions

### 3.1 Window size

- *Idea.* Too small gives too little information. Too large blurs timing and merges events.
- *Suggestion.* Sweep the window length. Plot per-event timing error against ROC-AUC. This is measurable with the current pipeline.

### 3.2 Overlap inside a window

- *Idea.* Two events in one window cause ambiguity.
- *Data.* About 26.5% of "on" frames have two or more events. FirstDigitTouch and BothStartLoadPhase are 22 frames apart.
- *Suggestion.* A window covering both cannot separate them by content alone. This supports the overlap-aware direction.

### 3.3 Forgetting and silence

- *Idea.* Long quiet stretches can drown out short events in an incremental window.
- *Link.* This is the same mechanism as the prototype problem: quiet frames dominated the original training sum.
- *Suggestion.* A decay factor (how quickly old frames fade) is the natural fix. "How long should the memory last?" is testable.

### 3.4 The Shazam idea

**What Shazam does.** It listens to a noisy live stream, finds the strongest spectral peaks (frequency and time of each peak), and matches the pattern of peaks to a database. It ignores most of the signal and keeps only the peaks, which makes it robust to noise and to where in the song you start listening.

**Mapping to EEG.** Encode peak strength and timing of the filtered signal in each window, instead of raw samples.

**Pros.**
- Robust to noise, because weak, noisy parts are discarded.
- Less data per window, which can make the hypervectors more informative.
- Does not depend on the exact start time of the window.
- The "characteristics" idea is close to a feature we already tested (power), so the comparison is direct.

**Cons.**
- Shazam matches against a database of known songs. Dense labelling is a live classification problem, so only the feature choice carries over, not the whole method.
- Peaks in EEG are less established than band power for motor tasks. Band power has the stronger support in the literature (ERD).
- Peak detection adds parameters (thresholds, peak spacing) that need tuning.
- "Sparsity" needs a definition. HDC vectors are already dense (about half their bits are 1). Sparsity may refer to the input, where most frames are quiet, or to redundant information in the raw signal.

**Suggestion.** Test peaks against band power on the same windows. Peter will ask why peaks should beat power, so define the test before the meeting.

### 3.5 Onset detection and anticipation

**Question for Peter.** Is the goal to anticipate the next move, or to predict the current event?

**Anticipation.** Predict a movement before it starts. The model sees a window that ends before onset and must match the build-up to the upcoming event's prototype.

- *Labels.* Shift the event labels earlier by a fixed amount (for example 500 ms or 1 s). This is the design choice that turns detection into anticipation.
- *Shift.* The best value depends on how early the build-up becomes detectable. Test a few values.
- *Frequency band.* The pre-movement build-up (readiness potential) is slow, mostly below 7 Hz. Our 7-30 Hz bandpass removes it. Anticipation therefore needs a different filter, so the bandpass toggle is not neutral for this question.
- *Causality.* Still holds. The window ends before onset, and the label points to the future. The model sees only the past.
- *Difficulty.* The build-up for different movements may look alike. Test whether pre-onset windows for different events form separate clusters.

**Suggestion.** Treat anticipation as a separate experiment from detection, not as a small variant.

---

## 4. Candidate main questions from the discussion

1. *Does encoding rhythm characteristics in a streaming window detect overlapping movement events more robustly than encoding raw samples, at a lower data cost?*
   - Broadest. Matches the Shazam idea.
2. *How long should an incremental HDC window remember, before silence drowns out short movement events?*
   - Most specific. Ties window, forgetting and overlap together.
3. *Do rhythm characteristics transfer across people better than raw signal, given that each person's waves differ?*
   - A stretch for now. Links to the leave-one-subject-out phase.

**Suggestion.** Start from 2 if you want a narrow, testable question. Use 1 as the framing if you want to keep the Shazam idea.

---

## 5. Claims to check before the meeting

- **"N-grams are state of the art for temporal classification."** Not established in this discussion. Rahimi et al. use windowed HDC for biosignals, but the premise cites them as examples, not as a benchmark. Do not state this without a source.
- **"Denser HV matrices, tackling sparsity."** Clarify what sparsity means (see 3.4).
- **Novelty.** Overlap-aware prototypes, rhythm encoding, and streaming HDC each need a literature search before being called new.

---

## 6. Open choices to settle before building

| Choice | Options | Why it matters |
|---|---|---|
| "Characteristics" | Peak strength and timing (Shazam), band power, wave shape | Decides the feature to encode |
| Window length | Sweep | Balances information against overlap |
| Memory length | Decay factor, sweep | Controls forgetting and silence |
| Task | Detection, anticipation | Decides the labels (shifted or not) and the filter band |
| Label shift (anticipation only) | 500 ms, 1 s, others | Sets how early the model must act |

---

## 7. What to say at 4 pm (short version)

- The Phase 1 question is clear, and the frame-wise half is measured: it is below chance and the encoding adds nothing at frame level.
- The main question is still open. Three candidate directions came out of the discussion. The next step is a literature check to see which is still open.
- Two design choices are still open: the feature (peaks, band power, or shape) and the task (detection or anticipation).
- Ask Peter which of the three directions fits his view of the thesis.