# Action plan from the 3 November 2026 voice note (Meghna)

Source: a spoken, unedited note recorded late at night. Ideas below are mine and yours, sorted by what you can act on.
Items marked **[question]** are research questions hidden in the note. Items marked **[bling]** are ideas to park.

## 1. Actionable items (in order)

1. **Build the tabular HDC baseline** (phase 1 of the four). Record its size, its speed, and its limitations. Treat each frame as independent: no time context.
2. **Measure what tabular HDC cannot do.** List the failures (missed onsets, confusion between close events, behaviour on rare events) with examples from the data. These failures motivate phase 2.
3. **Measure the overlap in the data across all subjects and series.** Earlier you counted 26.5% of "on" frames with two or more events in one series. Check whether that holds everywhere, and whether the 150-frame window and the pair gaps are stable. This is already open item 1 in `STATUS.md`.
4. **Check Rahimi et al.'s overlap tolerance.** Find the exact claim (the note says about 30%) and what it was measured on. Do not quote 30% until it is checked.
5. **Build the n-gram HDC baseline** (phase 2). Measure how it handles the overlap from item 3, and whether it classifies correctly.
6. **Measure streaming behaviour** for tabular and n-gram HDC: feed small chunks, compare with whole-series output, and time each step.
7. **Measure memory per phase.** Tabular HDC encodes each frame as a 10,000-bit vector. Record the item memories and the per-frame cost. Note what the n-gram version adds.
8. **Look up literature on overlapping events** (for example, multi-label event detection in EEG or sensor streams) and on non-linear HDC encoders that are lighter than kernel encoding.
9. **Decide the research question** after items 1 to 5. Peter's advice is to let the problems you meet decide.
10. **Ask Peter** whether a null or in-between class belongs in the evaluation, and how to position multi-centroid if it is not about class imbalance.

## 2. Sub-research questions hidden in the note

**From the tabular baseline (phase 1)**
- **[question]** Which limits of a tabular encoder (no temporal order) cost the most accuracy on dense labelling?
- **[question]** How much of the tabular model's error comes from overlapping events versus from classes that look alike?

**From n-gram HDC (phase 2)**
- **[question]** Does temporal n-gram encoding separate overlapping events that tabular HDC cannot?
- **[question]** What is the largest n for which the window still fits the streaming budget, and how does accuracy change with n?
- **[question]** Does the n-gram model keep its accuracy when the stream arrives in small chunks?

**Overlap as a problem in itself**
- **[question]** When two events overlap, do their class prototypes get the same updates, so that they become similar? Can you measure the similarity between the prototypes for overlapping and non-overlapping pairs?
- **[question]** Is overlap a problem of the representation (the encoder cannot tell the two events apart) or of the decision rule (the prototypes differ, but the threshold cannot)?

**Class imbalance**
- **[question]** Does HDC's prototype-based classification handle the 80% "rest" majority better than the discriminative baselines?

**Multi-centroid**
- **[question]** Is multi-centroid needed for onset detection (timing) rather than for separation? What would show that, in your numbers?
- **[question]** If two rest-like states exist, do they form separate clusters in the encoded space?

**Non-linear encoding**
- **[question]** Is a kernel (non-linear) encoder needed, or can a lighter non-linear encoder do the job?
- **[question]** Does a between-state class (for example, "almost moving") help separate the classes, or only the onset timing?

**Spatial versus temporal**
- **[question]** What does the spatial part of the EEG representation add that the temporal n-gram misses? (Hint from the note: is it about order, or about which electrodes carry the signal?)

## 3. Ideas to park (bling)

- **[bling]** Two hand gestures could be separated by putting their prototypes in different dimensions. Check later whether this is a real method or a restatement of multi-centroid.
- **[bling]** Try a Markov-style model where the state carries the decay of past frames, since a decay-weighted n-gram may match the streaming need.
- **[bling]** The idea that the kernel encoder may be unnecessary: test first whether a simpler non-linear step (for example, a sign or a product of two channels) helps.
- **[bling]** Measure accuracy, streaming, and separation as three separate axes, and report each one.

## 4. Metrics to record (from the note, plus what we already have)

- Accuracy and ROC-AUC, mAP, UAR (existing).
- Streaming behaviour: output difference between chunked and whole-series runs; latency per step.
- Memory: item memories, prototypes, and per-step state, in bytes.
- Separation: distance between class prototypes, and how it changes with overlap.
- Cost in bit-operations and multiply-accumulates, reported separately.

## 5. Open points to raise with Peter
- The overlap problem: is it a representation problem or a decision problem? Which would he expect?
- Whether multi-centroid's role is onset timing rather than class balance.
- Whether a null or intermediate class belongs in the evaluation.
- Whether the streaming test (chunked versus whole-series) counts as the real-time requirement.
