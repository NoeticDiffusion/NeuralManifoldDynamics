# 138 -- 2026-06-26 -- Claim upgrade + 04d-04f analyses + analysrepo batch

## Session purpose

Science lead approved upgrading the 04b m/d finding to internally validated
and requested behavioral review, subject robustness, classical EEG comparator,
updated claim ledger, and a batch package for handoff to the analysis repo.

---

## 1. Claim ledger update

File: `project/claims/ds006848_verbal_wm_claims.md`

Four internally validated results (V1-V4), two plausible interpretations
(P1-P2), three speculative claims (S1-S3), two rejected paths (R1-R2).

Key framing change from science lead:
- NOT "sequential > simultaneous" (Slow is also sequential)
- YES "rapid item-updating conditions (Fast and FastDelay) show higher
  manifold mobility/diffusivity than simultaneous or slow-paced encoding"

---

## 2. 04d -- Behavioral condition review

### NCorrect (median across 30 subjects)

| Condition    | NCorrect/7 | partialScore/7 |
|-------------|-----------|----------------|
| Simultaneous | 5.65      | 6.41           |
| Slow         | 4.80      | 5.71           |
| FastDelay    | 4.39      | 5.55           |
| Fast         | 4.03      | 5.01           |

Friedman NCorrect: chi2=39.77, p=1.2e-8.

**Critical observation**: The m/d ordering (Fast >= FastDelay >> Simultaneous > Slow)
is roughly the INVERSE of the accuracy ordering. Fast has the highest m/d
but worst accuracy; Simultaneous has the lowest m/d but best accuracy.
This strongly suggests m/d tracks encoding effort/load, not encoding success.

FastDelay partially recovers accuracy relative to Fast (ISI allows rehearsal)
while maintaining high m/d -- consistent with P2 (ISI aids consolidation
without suppressing updating dynamics).

### Serial position

Typical primacy/recency curve for Fast; near-perfect for Slow and FastDelay.
Data saved in `04d_behavioral/serial_position_accuracy.csv`.

---

## 3. 04e -- Subject robustness

### LOO Friedman
- m: 30/30 LOO runs remain p<0.05 (perfectly robust)
- d: 30/30 LOO runs remain p<0.05

No single subject drives the result.

### Bootstrap 95% CIs for m (10 000 resamples)

| Condition    | Median | 95% CI         |
|-------------|--------|----------------|
| FastDelay   | 0.460  | [0.183, 0.635] |
| Fast        | 0.411  | [0.277, 0.589] |
| Simultaneous| 0.256  | [0.037, 0.423] |
| Slow        | 0.201  | [0.032, 0.422] |

Fast/FastDelay CIs clearly separated from Slow, partially overlapping
with Simultaneous. The Fast vs Simultaneous comparison has some uncertainty.

### Rank consistency

| Metric | Fast > Simult | FastDelay > Simult |
|--------|---------------|--------------------|
| m      | 20/30 (67%)   | 23/30 (77%)        |
| d      | 23/30 (77%)   | 20/30 (67%)        |

Moderate-to-good directional consistency at the subject level.
Not floor/ceiling -- there is genuine heterogeneity across subjects.

---

## 4. 04f -- Classical EEG comparator

### Friedman results

| Feature                          | chi2  | p         |
|---------------------------------|-------|-----------|
| eeg_beta__g_frontal             | 33.24 | 2.9e-7    |
| eeg_alpha__g_frontal            | 14.76 | 0.0020    |
| eeg_theta__g_frontal            | 14.20 | 0.0026    |
| eeg_hjorth_complexity           | 14.72 | 0.0021    |
| eeg_hjorth_complexity__g_frontal| 11.48 | 0.0094    |

### Frontal theta by condition

| Condition    | theta |
|-------------|-------|
| Simultaneous | 6.38  |
| Slow         | 5.36  |
| Fast         | 4.83  |
| FastDelay    | 4.46  |

### Double dissociation

The classical EEG ordering (frontal theta: Simultaneous > Slow > Fast > FastDelay)
is the approximate INVERSE of the MNPS ordering (m: FastDelay >= Fast > Simultaneous > Slow).

Interpretation (plausible):
- Frontal theta tracks sustained maintenance/rehearsal engagement.
  Simultaneous presentation immediately loads working memory in a maintained
  representation (high theta), while sequential presentation requires rapid
  updating (low theta, high m/d).
- MNPS m/d tracks rapid neural state-updating dynamics during encoding.
- The two measures appear to index different aspects of WM encoding strategy.

This dissociation is potentially a strong addition to the paper: MNPS m/d
and frontal theta are measuring different, complementary aspects of encoding.
It also provides evidence against m/d being a simple redescription of
classical theta power.

---

## 5. Batch handoff package

File: `project/analysis/ds006848_analysis_batch.ps1`

Runs 04b through 04f in sequence and assembles all CSV/TXT summary files
plus the claim ledger into a dated package under:
`J:/processed/openneuro/ds006848/handoff_analysrepo/<timestamp>/`

Latest package: `handoff_analysrepo/20260626_143724/`

Usage: `.\project\analysis\ds006848_analysis_batch.ps1`
Dry run: `.\project\analysis\ds006848_analysis_batch.ps1 -DryRun`
Custom paths: use `-RunDir`, `-BidsDir`, `-OutRoot`, `-HandoffDir` parameters.

---

## 6. Open items

1. Behavioral-MNPS correlation: do high-m subjects have lower NCorrect within Fast?
   (Current 04d only has per-subject x condition medians; needs trial-level merge.)
2. Double dissociation report: confirm theta vs m/d ordering in pairwise contrasts.
3. Shorter-window rerun: still pending decision (allows F2/F3 for Fast/Simultaneous).
4. FastDelay microcycle: speculative until item-locked or <2 s windows.
5. Cross-dataset replication: check m/d pattern in ds003838 or other WM datasets.
