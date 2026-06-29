# ds006848 Verbal Working-Memory — Claim Ledger
## Last updated: 2026-06-29

---

## INTERNAL VALIDATED RESULTS
*Supported by: code, reproducible pipeline, n=30 Friedman + BH-FDR pairwise,
purity filters F0/F1/F4 (04b + 04c).*

### V1 — Encoding-phase m and d differ across presentation modes
- **Metric**: MNPS m (manifold mobility) and d (diffusivity) during encoding.
- **Contrast**: Rapid item-updating conditions (Fast, FastDelay) > Simultaneous
  and Slow during encoding.
- **Statistics**: Friedman χ²(m)=14.52 p=0.0023, χ²(d)=18.68 p=0.0003, n=30.
  7 BH-FDR significant pairwise contrasts (q<0.05) for m and d.
- **Purity audit (04c)**: Pattern survives F0 (any ≥25% enc overlap), F1
  (window centre inside encoding), F4 (overlap-weighted). F2/F3 (≥50/70%
  enc overlap) are geometrically inapplicable for 2.8 s encoding with 8 s
  windows — not an empirical weakness.
- **Caveat**: 8 s MNPS windows. Encoding windows for Fast/Simultaneous (2.8 s)
  average only 26% inside encoding; retention contamination ≈35%. Item-level
  interpretation and F2/F3 purity filters require shorter (≤4 s) windows.
- **Framing**: "Rapid item-updating conditions (Fast and FastDelay) show higher
  manifold mobility (m) and diffusivity (d) during encoding than simultaneous
  or slow-paced encoding." Do NOT frame as "sequential > simultaneous" because
  Slow is also sequential but shows lowest m/d.

### V2 — Maintenance-window MNPS null across prior presentation modes
- **Contrast**: Digits_Retrieval MNPS (m, d, e) does not differ significantly
  across prior encoding modes.
- **Statistics**: Friedman p(m)=0.110, p(d)=0.430, p(e)=0.155, n=30.
- **Interpretation**: The encoding-phase effect is encoding-specific, not a
  carry-forward state separation into maintenance.

### V3 — ECG polarity correction (generic)
- 92.7% of ds006848 epochs were inverted QRS; automated polarity correction
  brought median HR from ~99 bpm to 76.2 bpm (RMSSD 40.7 ms).
- Generic infrastructure: `_apply_polarity_correction` in `ecg.py`, activated
  for all datasets.

### V4 — WM-phase HRV gated due to superwindow contamination
- 87.7% of verbal-WM 60 s HRV windows contain Digits_Retrieval signal.
- Only ~2% of verbal-WM windows pass the clean HRV gate (ecg_hrv_contains_
  excluded_label == False, ecg_hrv_dominant_stage_frac ≥ 0.60).
- Rest-task HRV is clean (85% pass gate) and suitable for physiological QC.
- **Consequence**: No WM-phase HRV or anchor-index claims from ds006848 until
  a trial-aligned short-window HRV approach is implemented.

---

## PLAUSIBLE INTERPRETATIONS
*Reasonable reading of validated evidence; not yet directly tested.*

### P1 — Rapid item-updating increases manifold mobility/diffusivity
- Fast sequential presentation demands rapid neural state updating, which
  manifests as higher manifold mobility (m) and diffusivity (d).
- The effect appears even in the first 2.8 s (Test B), ruling out a pure
  duration effect.
- Fast (poorest accuracy) and FastDelay (best accuracy) both show high m/d,
  suggesting m/d reflects encoding effort/load rather than success directly.
  Pending behavioral correlation analysis (04d).

### P2 — FastDelay ISI aids consolidation without suppressing encoding dynamics
- FastDelay achieves near-Slow accuracy (6.66 vs 6.56 correct) while
  maintaining Fast-level m/d. The inter-stimulus delay may allow rehearsal
  that recovers accuracy without dampening encoding dynamics.
  Pending formal test in 04d/04e.

---

## SPECULATIVE
*Creative possibilities not yet tested or supported by direct evidence.*

### S1 — FastDelay encode–rehearse microcycles
- The 600 ms ISI in FastDelay may produce rapid encode→rehearse→encode
  micro-cycles, visible as periodic m/d oscillations at the item timescale.
- Requires ≤2 s windows or item-locked analysis to test.
- **Do not claim from current 8 s window data.**

### S2 — Item-level MNPS dynamics
- Individual digit events (400 ms presentation) may produce distinct MNPS
  transitions.
- Not resolvable from 8 s windows.

### S3 — m/d as a neural efficiency marker for WM encoding
- If m/d correlates with encoding success across conditions and subjects,
  it may index neural efficiency rather than mere task engagement.
- Requires 04d behavioral correlation and cross-dataset replication.

---

## REJECTED / FALSIFIED
*Hypotheses tested and found not to hold under current assumptions.*

### R1 — T-wave double-detection as primary HRV artifact source
- Resolved: the primary ds006848 ECG artifact was inverted QRS polarity, not
  T-wave double-detection. Pan-Tompkins on inverted signals detected T-waves
  as R-peaks, producing spurious HR ≈100 bpm and RMSSD ≈178 ms.
  Fixed by polarity correction.

### R2 — LVCF over-expansion of discrete encoding events
- Science-lead concern raised and audited. For ds006848, all events.tsv
  durations are 0, so LVCF produces short intervals (~0.4-0.8 s per digit).
  Not over-expanded. Future datasets with explicit durations will prefer them.

---

## OPEN QUESTIONS

1. Does m/d correlate with trial-level NCorrect / partialScore within condition?
   (04d behavioral review -- completed; behavioral dissociation confirmed)
2. Is the m/d effect robust to leave-one-out subject removal? What are
   bootstrap CIs? (04e robustness -- completed)
3. Do classical EEG comparators (frontal theta, alpha, complexity) replicate
   the same condition ordering? (04f EEG comparator -- completed; double-dissociation found)
4. Would a 2-4 s window rerun pass F2/F3 purity filters for Fast/Simultaneous?
   (04c + 4s run: F2/F3 unachievable without raw-EEG re-extraction at shorter epochs)
5. Do the m/d patterns replicate in other cognitive datasets?
6. [NEW] Does ICA cleaning change the m/d ordering?
   Artifact balance audit (pr04) COMPLETED. KEY RESULT:
   Frontal 30-45 Hz power differs significantly across conditions (chi2=23.00,
   p=0.00004), but in the OPPOSITE direction to m/d: Simultaneous > Fast/FastDelay.
   Temporal muscle artifact NOT significant (p=0.61).
   INTERPRETATION: Condition-structured frontal artifact argues AGAINST, not FOR,
   the Fast/FastDelay > Simultaneous m/d ordering. ICA recommended before top-tier
   journal submission but the m/d pattern is not explained by artifact direction.

---

## EEG ARTIFACT STATUS (peer-review gating)

| step | status | notes |
|------|--------|-------|
| Band-pass filtering (1-45 Hz) | Done | Standard |
| Average re-referencing | Done | Standard |
| Epoch z-score rejection (z>3) | Done | Only artifact mitigation in current run |
| Condition-balanced artifact audit | DONE | pr04: frontal HF sig but OPPOSITE direction |
| RANSAC bad-channel detection | NOT RUN | Config added (disabled); needs ICA rerun |
| ICA (EOG via Fp1/Fp2 proxy) | NOT RUN | Config added (disabled); needs ICA rerun |

Gating rule: V1/V2 findings may be reported as internally validated with the
artifact balance result cited. For top-tier journals (eLife, NeuroImage),
ICA cleaning is still recommended. The current result provides a strong argument
against artifact confound.

---

## PIPELINE INFRASTRUCTURE VALIDATED
- ECG polarity auto-detection (generic, `ecg.py`)
- Direct BIDS events.tsv event-locking with `kind: bids_events`
- HRV superwindow contamination reporting with exclude_labels gate
- Event-locked stage-transition margin default = 0 for `bids_events` kind
