# 009 — 2026-05-08 — Metric Correction, Multi-Subject QC, Claim Revision

## Session context

The architect introduced an external LLM reference (Purcell et al. 2017, N=11 630) indicating that the correct calibration metric is **spindles/minute per channel per sleep stage**, not spindles/hour. This required recalculating all prior results and re-evaluating the DQF-001 claim.

---

## 1. Metric recalculation — critical finding

| Detector | Events (sub-1) | Rate /h | Rate /min |
|----------|---------------|---------|-----------|
| Default YASA, freq_sp=(11,16) | 288 | 86.8 | **1.45** |
| Default YASA, freq_sp=(12,15) | 184 | 55.5 | **0.92** |
| "Calibrated" (rms=3.0 + remove_outliers) | 54 | 16.3 | **0.27** |

**Reference** (Purcell 2017, N2, C3/C4): 1.88/min; acceptable range (PLOS One): 0.70–4.80/min.

**Conclusion**: The original default YASA detection (1.45/min or 0.92/min) was **within the reference range**. The 5–15/h target used in session 008 was the wrong reference — it appears to have been derived from older or more conservative literature using different units. The "calibrated" result (0.27/min) fell **below** the warning threshold (< 0.3–0.5/min), meaning we over-corrected.

**DQF-001 is retracted.** The original YASA default annotations are plausible as a starting point.

---

## 2. Multi-subject QC — sub-1 to sub-5, PSG_F3

Script: `multi_subject_spindle_qc.py`

### DEFAULT params (freq_sp=(12,15), all other YASA defaults)

| Subject | N2 (min) | N3 (min) | N2 spindles | N2 rate (/min) | Status |
|---------|----------|----------|-------------|----------------|--------|
| sub-1 | 199 | 86 | 184 | 0.92 | OK |
| sub-2 | 225 | 103 | 124 | 0.55 | LOW |
| sub-3 | 296 | 40 | 351 | 1.19 | OK |
| sub-4 | 327 | 66 | 249 | 0.76 | OK |
| sub-5 | 360 | 1 | 620 | 1.72 | OK |
| **Mean** | | | | **1.03** | **4/5 OK** |

- 4/5 subjects within the 0.70–4.80/min reference range.
- sub-2 at 0.55/min is marginally below the lower bound — may reflect a genuine population difference or a recording artifact.

### CALIBRATED params (rms=3.0, remove_outliers=True)

All 5 subjects: N2 mean = 0.35/min — **uniformly below range**. Confirmed over-corrected.

### N3 findings

N3 spindle counts are near-zero across all subjects with both parameter sets (0–3 spindles per subject in N3). This is unexpected given the reference (Purcell: 1.45/min N3). Possible explanations:
1. PSG_F3 frontal channel has low sensitivity to deep-sleep (N3) spindles compared to central C3/C4.
2. The `include=(3,)` YASA call requires sufficient spindle-band power in N3 segments; PSG delta dominance may suppress relative sigma power.
3. Short N3 periods in sub-5 (1 min) trivially yield 0.

**Action**: N3 detection is flagged as unresolved but does not block N2-based analysis.

---

## 3. Revised calibration recommendation

Going forward:

| Parameter set | N2 rate (sub-1) | N2 mean (5 subs) | Recommendation |
|--------------|----------------|-----------------|----------------|
| Default, freq_sp=(11,16) | 1.45/min | not tested | Plausible; slightly broad band |
| Default, freq_sp=(12,15) | 0.92/min | 1.03/min | **Recommended** — close to Purcell center |
| remove_outliers=True only | 0.30/min | — | Too strict |
| rms=3.0 + remove_outliers | 0.27/min | 0.35/min | Too strict, below range |

**Recommended canonical parameter set for next event-locked run**:
```python
# freq_sp=(12,15), all other YASA defaults + N2 only
yasa.spindles_detect(data, sf=sfreq, hypno=hypno_up, include=(2,),
                     freq_sp=(12,15), freq_broad=(1,30), duration=(0.5,3.0),
                     min_distance=500, verbose=False)
```
Expected: ~1.0/min N2 on this dataset. This is slightly below the Purcell center (1.88/min) but within range, and is robust across 4/5 subjects.

---

## 4. Claim ledger updates

- **DQF-001**: Retracted. Default YASA is plausible by the /min metric.
- **DQF-002**: Downgraded. Low F3/C3 overlap is channel non-homology, not a false-positive flag.
- **REJ-001**: Revised. Default YASA is no longer "unsuitable" — it is a plausible exploratory starting point.
- **EVR-001**: Updated to reflect the correct reference (Purcell 2017 /min metric).
- **IVR-006**: The "calibrated" params are now classified as over-calibrated. Superseded by the revised recommendation.

---

## 5. Open items

1. **Re-run event-locked export** with the revised recommended params (freq_sp=(12,15), defaults + N2) on sub-1. This is now the first event-locked run with a plausibly calibrated detector.
2. **Expand multi-subject QC** to sub-6 to sub-10 if needed to confirm 4/5 holds.
3. **N3 investigation**: Optional. Understand why N3 spindles are near-zero on PSG_F3.
4. **Subject 2 event-locked run**: After sub-1 is confirmed, run sub-2 to sub-5 for reproducibility.
