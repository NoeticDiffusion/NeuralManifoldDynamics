# ds003838 full HRV block-native validation complete

## Research question

Did the full `ds003838` dataset run complete cleanly with the new raw `ecg_hrv_*`
surface exposed directly in `block_native` exports?

## Command

```text
python -m mndm.cli all --dataset ds003838 --config mndm/config/config_ingest_ds003838.yaml --out-dir H:/SourceRepo2/NeuralManifoldDynamics/.full_ds003838_hrv_blocknative --n-jobs 2
```

## Result

- The full run completed successfully with `exit_code: 0`.
- Runtime was approximately `29686411 ms` (~8.25 h).
- Output run directory:
  - `H:/SourceRepo2/NeuralManifoldDynamics/.full_ds003838_hrv_blocknative/ds003838/neuralmanifolddynamics_ds003838_20260607_111755`

## Full-cohort validation summary

### Run-level manifest

From `run_manifest.json`:

- `counts.h5 = 130`
- `counts.summary_json = 130`
- `capabilities.has_block_native_windows = true`
- `capabilities.counts.h5_with_block_native_windows = 130`
- `capabilities.anchor_state = true`
- `capabilities.anchor_quality = true`
- `capabilities.anchor_coupling = true`
- `extra.run_status = "completed"`
- `extra.run_errors.count = 0`

### Aggregate block-native QC

From `block_native_qc.json` / `run_manifest.extra.block_native_qc.aggregate`:

- `subjects_total = 130`
- `blocks_total = 6543`
- `windows_total = 27670`
- `source_window_match_fraction = 1.0`

Stage totals:

- blocks: `rest=65`, `listen=2171`, `mem5=1273`, `mem9=1469`, `mem13=1565`
- windows: `rest=3961`, `listen=7944`, `mem5=2905`, `mem9=5361`, `mem13=7499`

This confirms that the task-segment-driven `block_native` path scaled to the
full dataset without join failures or source-window mismatches.

### Direct sidecar confirmation

Checked a real exported table:

- `sub-098_digit_span/block_native_windows.csv`

The header includes the raw HRV columns directly:

- `ecg_hrv_hr_mean_bpm`
- `ecg_hrv_ibi_mean_ms`
- `ecg_hrv_sdnn_ms`
- `ecg_hrv_rmssd_ms`
- `ecg_hrv_pnn50`
- `ecg_hrv_nn_count`
- `ecg_hrv_artifact_fraction`
- `ecg_hrv_coverage_fraction`
- `ecg_hrv_quality_score`

The same sidecar also carries:

- anchor columns (`sympathetic_index`, `vagal_index`, `anchor_index`, etc.)
- anchor-quality columns
- `task_state_label`, `task_load_label`, `task_load_n`

### Subject-level contract confirmation

Checked `sub-098_digit_span/summary.json`:

- `anchor_hrv_v0_1.enabled = true`
- `anchor_state.source_features` prefers HRV v0.1 columns
- `block_native.status = "ok"`
- `block_native.derived_from = "task_state_label_segments"`
- `block_native.n_blocks = 96`
- `block_native.n_windows = 359`
- `block_native.source_window_match_count = 359`

## Conclusion

The full `ds003838` validation succeeded.

The new raw HRV v0.1 surface is now confirmed at dataset scale in the intended
analysis surface:

- raw `ecg_hrv_*` columns are present directly in `block_native` sidecars,
- block-native windows are derived from sustained task-state segments rather than
  one-second digit events,
- anchor-state and task-label joins remain intact,
- and the full export completed without run-level errors.
