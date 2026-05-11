# 006 – P0 fixes + real YASA spindles (2026-05-08)

## Session goal

Address architect's P0/P1 priorities after the ds005555 sub-1 real HDF5 smoke test:

1. **P0-A**: Audit `/time` vs `(window_start + window_end) / 2`.
2. **P0-B**: Fix summarization so `/time` follows feature-window timing.
3. **P0-C**: Wire `EventLockedProfile` into `ExportConfig` provenance.
4. **P1-A**: Inspect ds005555 `events.tsv` for spindle annotations; import real ones.
5. **P1-B**: Re-run event-locked export with real spindle annotations.

---

## P0-A: Time axis audit

**Finding (critical):**

| Field | Before fix | After fix |
|---|---|---|
| `window_start` step | 2 s (correct, from `t_start`) | 2 s (unchanged) |
| `window_end` step | 2 s (correct, from `t_end`) | 2 s (unchanged) |
| `/time` step | **4 s (WRONG)** | **2 s (CORRECT)** |
| `time[i]` formula | `window_sec/2 + step*i` where step=4 | `(t_start + t_end)/2` |

Root cause: `projection.build_time_index(n, window_sec=8.0, overlap=0.5)` gave
`step = 8*(1-0.5) = 4 s`, a value inherited from the base MNPS config. The feature
epochs have `step_s = 2 s` but the MNPS config was not updated for the overlay.

Impact:
- `rel_time_sec` in event alignment was computed from wrong window centers (off by
  up to several seconds per window for later windows).
- Stage-transition exclusion used wrong center times.
- Jacobian `j_dot_dt` was 4 s instead of 2 s (velocity scale error 2×).
- `coverage_seconds_assumed = n * dt = 13727 * 4 = 54908 s ≈ 15 h` instead of 7.63 h.

**Fix: `mndm/src/mndm/pipeline/summary.py`**

Added two changes:

1. **`dt` derivation** (early in `run()`): if `t_start` is present in features frame,
   compute `dt = t_start.diff().median()` and log if it differs from the config formula.
   Falls back to `mnps_cfg["window_sec"] * (1 - mnps_cfg["overlap"])` when `t_start`
   is absent.

2. **Time axis** (before `_extract_time_bounds`): if both `t_start` and `t_end` present,
   use `(t_start + t_end) / 2` as `/time` instead of `build_time_index`. The fallback
   formula remains for fMRI/legacy pipelines without `t_start/t_end`.

This fix is universally correct for all EEG pipelines (features always have `t_start/t_end`).
It is backward-safe for fMRI (no `t_start/t_end` → formula unchanged).

**Verification:**
```
=== Fixed time axis audit ===
  i  ws    we    center  time  diff
  0  0.0  6.0  3.0   3.0  +0.0
  ...
time == (ws+we)/2 everywhere: True
```

Log message during re-summarize:
```
INFO: Epoch step 2.000 s (from t_start) differs from mnps config formula 4.000 s
      (window_sec=8.0, overlap=0.5000). Using measured step for time axis and Jacobian dt.
```

---

## P0-B: Provenance plumbing

**Before:** `profile_name`, `window_length_s`, `window_step_s` were `None` in every
exported row because `ExportConfig` had no fields for them.

**After:**

1. `ExportConfig` (in `event_locked_export.py`) extended with:
   - `profile_name: Optional[str] = None`
   - `window_length_s: Optional[float] = None`
   - `window_step_s: Optional[float] = None`
   - `bins_json: Optional[str] = None`
   - `alignment_reference: Optional[str] = None`
   - `control_seed: Optional[int] = None`

2. `build_event_locked_table` provenance block updated to use these fields when
   present, falling back to alignment/control QC values.

3. New function `export_config_from_profile(profile, config, dataset_id)` in
   `event_locked_config.py` creates an `ExportConfig` fully wired to an
   `EventLockedProfile`, so all provenance fields are populated.

4. Smoke test updated to use `export_config_from_profile(PROFILE, _cfg, DATASET_ID)`.

**Verification (smoke test output):**
```
  profile_name: 'sleep_spindle_event_locked_v1'  [OK]
  window_length_s: 6.0  [OK]
  window_step_s: 2.0  [OK]
  control_seed: 42  [OK]
  dataset_id: 'ds005555'  [OK]
  subject_id: 'sub-1'  [OK]
  All provenance fields correct.
```

---

## P1-A: Real spindle annotations

`events.tsv` for ds005555 sub-1 contains only 30-second sleep staging annotations
(N2/N3/REM/Wake). No spindle-level events are present in the dataset.

**Decision:** use YASA automatic detection, clearly labeled as
`source = "detector:yasa-0.7.0"`, NOT ground truth.

YASA 0.7.0 was installed. Detection run on `PSG_F3` (first EEG channel) with
N2-only filter, sigma band 11–16 Hz, duration 0.5–3.0 s, min distance 500 ms.

Results saved to:
```
M:\datasets\received\openneuro\ds005555\sub-1\eeg\
  sub-1_task-Sleep_acq-psg_spindles_yasa.csv
```

**Detection summary:**
- N spindles: 288
- Duration: mean 0.86 s, range 0.50–2.84 s
- Onset range: 1369–27027 s (N2 periods, correct)
- Channel: PSG_F3

---

## P1-B: Real event-locked export

Re-ran `smoke_real_h5_event_locked.py` with YASA annotations against the fixed H5
(correct `/time` axis). Config driven by `EventLockedProfile` via `export_config_from_profile`.

| Metric | Value |
|---|---|
| Spindles loaded | 288 |
| Events aligned | 242 (46 excluded near stage transitions) |
| Controls matched | 864 / 864 (0 failed) |
| Total rows | 8 124 |
| Parquet size | 779 KB, 53 cols |
| Finite MNPS | 8 124 / 8 124 |

**Bin coverage:**

| Bin | Rows | % |
|---|---|---|
| pre_far | 2 420 | 29.8 |
| pre_near | 1 210 | 14.9 |
| event | 362 | 4.5 |
| post_near | 848 | 10.4 |
| post_far | 2 420 | 29.8 |
| control | 864 | 10.6 |

**MNPS mean at spindle-event windows:**
- m = 0.265, d = 0.231, e = −0.087 (N2-locked, YASA-detected, single channel)

---

## Claims ledger update

| Claim | Status | Evidence |
|---|---|---|
| `/time == (ws+we)/2` for 6 s/2 s epochs after fix | **Validated** | Audit script, allclose=True |
| Jacobian dt corrected from 4 s to 2 s | **Validated** | summary.py log + re-summarize |
| Provenance fields non-None when driven by EventLockedProfile | **Validated** | Smoke test output |
| 288 spindles detected by YASA on PSG_F3, N2 only | **Validated (detector, not GT)** | YASA output CSV |
| 242 of 288 spindles align to N2 MNPS windows | **Validated** | Alignment QC |
| All 5 bins populated with real spindle events | **Validated** | Smoke test |
| MNPS shows m=0.265 at spindle windows | **Plausible interpretation** | Single subject, single channel, YASA detector; no control contrast computed yet |

---

## Known limitations / open issues

1. **Single EEG channel**: YASA detection used only PSG_F3. Standard practice is to
   use C3–A2 or average of frontal/central channels. Rerunning on all EEG channels and
   taking the union (or selecting C3) is future work.
2. **No biological claim**: MNPS mean at spindle windows vs. matched controls has not
   been tested statistically. The difference (if any) is measurement, not interpretation.
3. **YASA confidence**: YASA 0.7.0 does not output per-spindle probability by default;
   `confidence = NaN` in all rows. Future: use `yasa.spindles_detect(..., verbose=False)`
   return value to extract probabilities if available.
4. **summary.py dt change**: existing HDF5 files produced before this fix have wrong
   `/time` values and wrong Jacobian velocity scale. They should be re-summarized.

---

## Next smallest steps

**Immediate (P0 clean-up):**
- Update `mnps.window_sec` / `mnps.overlap` in the overlay config to
  silence the INFO log (optional, cosmetic).
- Consider adding a `mnps.time_axis: feature_midpoints` config switch to
  explicitly opt into the new behavior (prevents surprise for other datasets).

**Short-term (P1 completion):**
- Re-run YASA on C3 or all EEG channels and compare spindle counts.
- Load the Parquet in a downstream analysis notebook; compute spindle vs.
  matched-control contrast in MNPS 3D (one subject, descriptive only).
- Run the same pipeline for sub-2 to verify no dataset-specific artifacts.

**Medium-term (Phase 2):**
- Integrate `detect_spindles_yasa.py` as an optional annotation module in `mndm`
  (with `source = "detector:yasa-<version>"` and explicit `confidence` column).
- Evaluate spindle detector against MODA or DREAMS expert annotations when available.
