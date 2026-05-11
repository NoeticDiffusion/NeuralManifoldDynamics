# 007 — 2026-05-08 — QC Report, Regression Tests, C3 Channel, and Descriptive Summary

## Session context

Continuing from [006], where P0 (time-axis fix + provenance plumbing) and P1 (real YASA spindle annotations) were completed. The architect requested:

1. Add regression tests guarding the `/time == (ws+we)/2` and `dt` fix.
2. Run a YASA QC report (duration histogram, rate, stage, exclusion reasons).
3. Add YASA CSV SHA-256 hash to export provenance.
4. Re-run spindle detection on a second channel (C3) as a sensitivity check.
5. Compute descriptive event-vs-control MNPS summaries (exploratory, no statistics).

---

## 1. Regression tests — `test_time_axis_regression.py`

**File**: `mndm/tests/test_time_axis_regression.py`

Added 20 regression tests across 5 classes, covering all four contracts:

- **C1** — `time[i] == (t_start[i] + t_end[i]) / 2` for all `i` when bounds are present (6 tests).
- **C2** — `dt == median(diff(t_start))` when `t_start` available, not config formula (4 tests).
- **C3** — `coverage_seconds_assumed ≈ actual recording span`, not 2× pre-fix value (3 tests).
- **C4 + C5** — fallback to `build_time_index` and config formula when `t_start/t_end` absent (4 tests).

Additional class `TestDs005555Sub1RegressionValues` pins the exact numeric regression:
- `time[0] = 3.0`, not `4.0` (pre-fix)
- `time step = 2.0 s`, not `4.0 s`
- `dt = 2.0 s`, not `4.0 s`

Result: **20/20 passed**.

These tests are architecture-level (not integration): they test only the `dt/time` derivation logic extracted from `summary.py`, without requiring full pipeline dependencies or real data. They must be run on every PR touching `summary.py`.

---

## 2. YASA QC report — key findings

**Script**: `project/smoke_tests/qc_spindle_detections.py`

### Counts and rate

| | F3 (PSG_F3) | C3 (PSG_C3) |
|---|---|---|
| N spindles detected | 288 | 244 |
| N2 duration | 3.32 h | 3.32 h |
| Rate (spindles/h N2) | **86.8** | **73.6** |

> **IMPORTANT QC FLAG**: Both rates (73–87 /h) substantially exceed the published typical range for healthy adults (5–15 /h). This is a detector calibration issue, not a pipeline bug. Possible causes: PSG F3/C3 channels may have different amplitude characteristics than standard EEG scalp electrodes, or YASA's defaults are permissive for this equipment. These must be treated as preliminary detector-derived events with an elevated false-positive risk.

### Duration distribution (F3)

- Mean ± SD: 0.86 ± 0.39 s
- Median: 0.76 s
- Distribution is heavily skewed toward shorter durations (79 events at 0.4–0.6 s, 83 at 0.6–0.8 s)
- Short spindles dominate — consistent with YASA's minimum duration threshold (0.5 s) being active

### Frequency

- F3: 12.77 ± 0.59 Hz
- C3: 13.11 Hz (mean)
- Both within the sigma band (11–16 Hz) ✓

### Temporal distribution across night (F3)

Spindles distributed across all 4 quarters of the night — no dramatic circadian artifact visible:

| Quarter | N spindles |
|---------|------------|
| Q1 (0.38–2.16 h) | 64 |
| Q2 (2.16–3.94 h) | 89 |
| Q3 (3.94–5.73 h) | 69 |
| Q4 (5.73–7.51 h) | 65 |

### Stage filter

All detected spindles have `stage=2` (N2) — the N2-only filter is working correctly.

### Exclusion rate

- 46 of 288 spindles excluded from alignment (stage-transition margin)
- Exclusion rate: **16.0%** — within acceptable range (< 20%) ✓
- The 46 exclusions are reasonable

### C3/F3 channel comparison

- C3/F3 ratio: 0.85 — within expected range (0.7–1.3) ✓
- Temporal overlap (C3 within 1s of F3 onset): **39.3%** (96/244)
- This is lower than expected for homologous channels (typical 50–70%), consistent with the high false-positive rate inflating both channels independently

**Claim boundary**: The low temporal overlap and very high detection rate together confirm that a large proportion of detected events are likely false positives. No biological interpretation should be drawn until detection parameters are calibrated for this specific PSG equipment.

---

## 3. Annotation source hash

**Files changed**: `mndm/src/mndm/pipeline/event_locked_export.py`

Added `annotation_source_hash: Optional[str]` to `ExportConfig`. When `None`, the hash is automatically computed from `event_table.source_path` at export time (SHA-256, streaming). The hash is stored in every provenance row in the Parquet.

New helper: `_sha256_of_file(path, chunk=1 MiB)` — gracefully returns `""` if the file is missing or inaccessible.

Verified in smoke test: hash = `81b3369b26d90fd4...` (64 hex chars).

---

## 4. C3 detection

**Script**: `project/smoke_tests/detect_spindles_yasa_c3.py`

- Auto-detects `PSG_C3` channel from the available channel list
- Same YASA parameters as F3 run
- Output: `M:\datasets\received\openneuro\ds005555\sub-1\eeg\sub-1_task-Sleep_acq-psg_spindles_yasa_C3.csv`
- 244 spindles detected, rate 73.6/h N2

---

## 5. Descriptive event-vs-control summary

**Script**: `project/smoke_tests/descriptive_event_vs_control.py`

Reads the event-locked Parquet and outputs a plain-text exploratory summary. No statistics, no hypothesis tests.

### Output highlights (exploratory, all values from detector-derived events)

**Aggregate comparison (all event rows vs all control rows)**

| Dim | Event mean | Control mean | Diff (ev − ct) |
|-----|-----------|-------------|----------------|
| m   | 0.2648    | 0.3397      | −0.0749        |
| d   | 0.2309    | 0.2794      | −0.0484        |
| e   | −0.0873   | 0.0237      | −0.1110        |

Event rows show lower m, d, e than controls. The e (entropy) difference is largest in magnitude.

**Per-bin breakdown — spindle event rows**

| Bin | N | m | d | e |
|-----|---|---|---|---|
| event | 362 | 0.1867 | 0.2574 | −0.0451 |
| post_near | 848 | 0.2781 | 0.2574 | −0.0942 |
| post_far | 2420 | 0.2355 | 0.2125 | −0.0751 |
| pre_near | 1210 | 0.2846 | 0.2425 | −0.0888 |
| pre_far | 2420 | 0.2913 | 0.2303 | −0.1026 |

Note: `matched_control` rows are assigned `bin_label="control"` (single bin), not split by temporal position.

**IMPORTANT: These patterns must not be interpreted as evidence of spindle effects.** The high detection rate (86.8/h) means a substantial fraction of "spindle event" rows may correspond to N2 windows without genuine spindles. The descriptive differences may reflect confounding from event-density effects, selection bias, or YASA detection artifacts.

---

## Data quality

- 8124/8124 rows have finite MNPS values ✓
- All 5 temporal bins populated ✓
- 864/864 controls matched ✓
- All provenance fields complete ✓

---

## Smoke test update

`smoke_real_h5_event_locked.py` now writes the Parquet to a stable persistent path alongside the HDF5 (`.../sub-1_Sleep_acq-psg_event_locked.parquet`) instead of a temporary directory. This is needed for downstream scripts like `descriptive_event_vs_control.py`.

---

## Claims status

### Valid claim (this session)

> "Using YASA 0.7.0 detector-derived spindle events on ds005555 sub-1 (PSG_F3 channel), the event-locked MNPS pipeline produces finite, provenance-complete rows (8124 rows, 54 columns). Descriptive MNPS values differ between event and control rows. No statistical inference has been performed."

### Not yet valid

> "Sleep spindles have an MNPS effect." — requires: (1) calibrated detection with realistic rates, (2) formal statistical test with appropriate multiple-comparison control, (3) replication in a second subject.

---

## Open items for next session

**P0 (measurement contract)**:
- [ ] YASA detection calibration: evaluate stricter parameters (amplitude threshold, `min_distance`, `freq_sp` narrowing) to bring rate into the 5–20 /h range typical for PSG studies.

**P1 (QC / robustness)**:
- [ ] Second subject: repeat the full event-locked pipeline on ds005555 sub-2 (or any other N2-rich run) to check reproducibility.
- [ ] Document the 39% C3/F3 overlap as a pre-calibration baseline; revisit after parameter tuning.

**P2 (downstream)**:
- [ ] Design a formal comparison protocol (after calibration): proper matched-pairs test structure, accounting for within-subject temporal autocorrelation.
