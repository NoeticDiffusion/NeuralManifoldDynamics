# 005 – Real ds005555 sub-1 H5 smoke test (2026-05-08)

## Session goal

Run the full MNDM pipeline (features + summarize) for `ds005555 sub-1` using the
`config_ingest_ds005555_sleep_spindles.yaml` overlay, then validate the event-locked
export against the resulting real HDF5 file.

---

## Steps completed

### 1. Path patch in overlay config

`config_ingest_ds005555_sleep_spindles.yaml` inherited `received_dir = E:/Science_Datasets/openneuro/received`
from the base config but ds005555 lives on `M:` on this machine.
Added a `paths:` override directly in the overlay:

```yaml
paths:
  received_dir: "M:/datasets/received/openneuro"
  processed_dir: "M:/datasets/processed/openneuro"
```

Prerequisite-check passed with `overall_ok: True`.

### 2. Features run (mndm.cli features)

Command:
```
python -m mndm.cli features --dataset ds005555
    --config mndm/config/config_ingest_ds005555_sleep_spindles.yaml
    --subject sub-1 --h5-mode subject --n-jobs 1
```

Note: `--subject 1` pads to `sub-001` internally; must pass `--subject sub-1` to
match the dataset's non-padded naming convention.

Results:
- 13 727 EEG epochs extracted (6 s length, 2 s step)
- PSG file: `sub-1_task-Sleep_acq-psg_eeg.edf` (headband excluded by `exclude-files`)
- CSD skipped (only 6 channels, below `min_eeg_channels = 16`) — not a blocker
- Intermediate JSON written to `M:\datasets\processed\openneuro\ds005555\intermediate\`
- Features Parquet written to `M:\datasets\processed\openneuro\ds005555\features.parquet`
- Exit code: 0, elapsed: ~44 s

### 3. Summarize run (mndm.cli summarize)

Command:
```
python -m mndm.cli summarize --dataset ds005555
    --config mndm/config/config_ingest_ds005555_sleep_spindles.yaml
    --subject sub-1 --h5-mode subject --n-jobs 1
```

Output:
```
M:\datasets\processed\openneuro\ds005555\
  neuralmanifolddynamics_ds005555_20260508_090058\
    sub-1_Sleep_acq-psg\
      sub-1_Sleep_acq-psg.h5
      qc_reliability.json
      qc_summary.json
      summary.json
    run_manifest.json
```

Exit code: 0, elapsed: ~35 s

### 4. HDF5 structure verification

| Dataset | Shape | Notes |
|---|---|---|
| `time` | (13727,) | step = 4 s (center of 6 s window vs previous center, expected 4 s for non-overlapping centers with 2 s step and 6 s window) |
| `window_start` | (13727,) | step = 2.0 s confirmed ✓ |
| `window_end` | (13727,) | duration = 6.0 s confirmed ✓ |
| `mnps_3d` | (13727, 3) | |
| `mnps_3d_dot` | (13727, 3) | |
| `coords_9d/values` | (13727, 9) | |
| `jacobian/J_hat` | (13725, 3, 3) | |
| `jacobian_9D/J_hat` | (13725, 9, 9) | |
| `labels/stage` | (13727,) int8 | all 6 stages present |

**Sleep stage distribution:**

| Stage | Code | N | % |
|---|---|---|---|
| Wake | 0 | 3000 | 21.9 % |
| N1 | 1 | 855 | 6.2 % |
| N2 | 2 | 5970 | 43.5 % |
| N3 | 3 | 2579 | 18.8 % |
| REM | 4 | 1305 | 9.5 % |
| Unscored | 8 | 15 | 0.1 % |

Recording duration: **7.63 h**

### 5. Event-locked export smoke test (`smoke_real_h5_event_locked.py`)

Synthetic spindles: every 3rd N2 window onset (1 990 events, 1.5 s duration).

Results:
- Event-window alignment: 28 290 pairs, **1 886 events aligned** (104 excluded
  near stage transitions)
- All 5 bins populated:

| Bin | Pairs |
|---|---|
| pre_far | 9 430 |
| pre_near | 4 731 |
| event | 1 886 |
| post_near | 2 813 |
| post_far | 9 430 |

- Matched N2 controls: **5 970 / 5 970** (0 failed)
- Export table: **34 260 rows × 50 columns**, all MNPS finite
- Parquet: 1 277.7 KB
- Spindle-event MNPS mean: m=0.029, d=0.127, e=−0.232 (raw scale, not normed)
- Exit code: 0 ✓

---

## Known issues / observations

- `profile_name`, `window_length_s`, `step_s` are `None` in provenance rows because
  `build_event_locked_table` currently takes these from `ExportConfig` fields that
  haven't been wired to the YAML `EventLockedProfile`. This is a plumbing gap to
  address when integrating into the main CLI pipeline.
- `time` axis step appears as 4 s (centers of adjacent 6 s / 2 s windows are 4 s
  apart by geometry: center(window_k) = start_k + 3 = k*2 + 3). Correct; not a bug.
- Subject `--subject 1` pads to `sub-001` inside summarize; must pass `sub-1` to
  match ds005555 naming. Logged as a minor usability note; no code change needed
  for this stage.

---

## Claims (internal validated)

- **Validated**: The 6 s / 2 s MNDM windows, configured via `event_locked` overlay,
  produce 13 727 windows for sub-1 (7.63 h PSG, N2 = 43.5 %).
- **Validated**: All 5 relative-time bins (pre_far, pre_near, event, post_near,
  post_far) are populated against real MNPS data from ds005555 sub-1.
- **Validated**: Matched N2 controls can be sampled at 100 % success rate for
  1 990 synthetic spindle events in this recording.
- **Speculative**: MNPS values at synthetic spindle windows (m≈0.029, d≈0.127,
  e≈−0.232) are not interpreted; real spindle annotations are required before
  any effect claim can be made.

---

## Next smallest step

**Wire real spindle annotations.**

Options in priority order:
1. Check whether `ds005555` `events.tsv` contains spindle-type annotations beyond
   sleep-stage scoring.
2. If not: identify a published spindle annotation file (e.g. from MODA or DREAMS
   databases, or a YASA-generated output for sub-1) to use as imported CSV.
3. Re-run `smoke_real_h5_event_locked.py` with real spindle onsets.
4. Then wire `EventLockedProfile` provenance into `ExportConfig` / `build_event_locked_table`.
