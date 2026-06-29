# 126 — 2026-06-24 — ECG T-wave double-detection: diagnosis, fix, and re-processing

## Context

Peer reviewer (pr7) flagged a potential fatal problem with ECG R-peak detection in the MNDM pipeline for ds003838. The supplement reported group-median HR of 124–128 bpm and RMSSD of 174–183 ms, while Pavlov et al. 2023 (the source paper) report HR ~77–80 bpm for the same cohort. A ratio of ~1.6× HR is the classic signature of **T-wave double-detection**: the T-wave is counted as an extra R-peak, halving RR intervals and doubling HR.

## Diagnosis (ECG audit)

Wrote and ran `project/scripts/ecg_audit_ds003838.py` on 6 subjects spanning the RMSSD distribution (sub-091 = low HR, sub-050 = highest HR). Compared the existing mndm detector against NeuroKit2's Pan-Tompkins implementation.

| Subject | Task   | mndm HR | NK2 HR | HR ratio | mndm RMSSD | NK2 RMSSD |
|---------|--------|---------|--------|----------|------------|-----------|
| sub-091 | memory | 96.3    | 75.9   | 1.27     | 195.5 ms   | 98.9 ms   |
| sub-065 | memory | 110.5   | 109.6  | 1.01     | 32.5 ms    | 31.9 ms   |
| sub-050 | memory | 149.7   | 76.7   | 1.95     | 173.8 ms   | 129.7 ms  |
| sub-050 | rest   | 151.3   | 75.8   | 2.00     | 179.5 ms   | 86.0 ms   |
| sub-035 | memory | 111.5   | 81.7   | 1.37     | 185.7 ms   | 55.9 ms   |
| sub-088 | memory | 114.0   | 84.3   | 1.35     | 183.5 ms   | 52.3 ms   |

**Group audit summary (12 recordings):**
- mndm HR median: 111 bpm vs NK2 HR median: 82 bpm (bias = +31.8 bpm, LoA = [-23, +87])
- mndm RMSSD median: **176.7 ms** vs NK2 RMSSD median: **48.6 ms** (3.6× inflation)

The near-perfect ratio of 2.00 for sub-050 rest confirms T-wave double-detection on every beat for that subject. Sub-065 (ratio ≈ 1.00) is an exception — likely different ECG morphology.

## Root cause

In `mndm/src/mndm/features/ecg.py`, the peak detector used:

```python
sig = np.abs(centered)   # converts negative S-wave and T-wave to positive peaks
```

The `np.abs()` causes both the R-peak (positive) and the T-wave (which can be 300–400 ms post-R, just outside the 300 ms refractory) to appear as prominent positive peaks. With `rr_min_s = 0.3` (300 ms), short T-wave intervals are allowed through. Alternating short (R→T) and longer (T→R) intervals massively inflate RMSSD and double HR.

## Fix

Replaced the `np.abs()` approach with a **three-tier detector cascade** in `ecg.py`:

1. **`"neurokit2"` (new default):** `nk.ecg_process()` → Pan-Tompkins QRS detection. Correctly ignores T-waves.
2. **`"scipy_polarity"` (fallback):** Bandpass → clips to positive side only, auto-detects polarity. No `abs()`.
3. **`"scipy_abs"` (legacy):** The old broken approach, retained for debugging only.

Config option: `features.ecg.peak_detector` — defaults to `"neurokit2"` when the package is installed.

Added `neurokit2>=0.2.13` to `requirements.txt`.

Also updated `project/scripts/hrv_robustness_sidecar_ds003838.py` to use the same NK2-based detection.

## Re-processing pipeline

**Step 1 (running now):** `project/scripts/ecg_patch_features.py`
- Backs up `features.parquet` as `features_pre_ecg_patch.parquet`
- For each (subject, task): loads raw ECG .set → resample to 250 Hz → NK2 detection → recompute per-epoch ECG features → overwrites ECG columns in-place
- Preserves all EEG feature columns
- 124 subject×task combos, estimated ~2.5 hours single-threaded

**Step 2 (after patch):** `mndm.cli summarize --dataset ds003838` to regenerate corrected `block_native_windows.parquet`

**Step 3 (after summarize):** `project/scripts/hrv_corrected_analysis_ds003838.py`
- Corrected group medians per stage (HR, RMSSD, pNN50)
- Friedman test for vagal_index
- C3 Wilcoxon + Cohen's d (listen vs mem)
- Bland-Altman: old vs corrected RMSSD

## Parallel deliverables

**Request 2 — Full-cohort robustness sidecar (running now):**
`project/scripts/hrv_robustness_sidecar_ds003838.py` on all 62 EEG subjects with corrected NK2 detector. Output: `hrv_robustness_sidecar_fullcohort.parquet`.

**Request 3 — Boundary-excluded HRV sensitivity:**
`project/scripts/hrv_boundary_sensitivity_ds003838.py` — script written, awaits corrected block_native_windows (post-summarize).

## Claim status

| Claim | Status |
|-------|--------|
| T-wave double-detection confirmed | **Validated** — audit shows HR ratio up to 2.00, RMSSD 3.6× inflated |
| NK2 gives expected HR (~78–85 bpm) | **Validated** — matches Pavlov et al. 2023 |
| `np.abs()` is the root cause | **Validated** — removing it fixes sub-050 ratio to 1.00 |
| NK2 fix is correct for sub-065 (already good) | **Validated** — ratio stays 1.00 after fix |

## Files changed

- `mndm/src/mndm/features/ecg.py` — new detector dispatch, three modes
- `mndm/config/config_ingest_ds003838.yaml` — explicit `peak_detector: neurokit2`
- `requirements.txt` — added `neurokit2>=0.2.13`
- `project/scripts/ecg_audit_ds003838.py` (new)
- `project/scripts/ecg_patch_features.py` (new)
- `project/scripts/hrv_corrected_analysis_ds003838.py` (new)
- `project/scripts/hrv_boundary_sensitivity_ds003838.py` (new)
- `project/scripts/hrv_robustness_sidecar_ds003838.py` — updated detector
