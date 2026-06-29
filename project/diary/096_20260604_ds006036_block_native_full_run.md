# 096 — ds006036 block-native full run complete

## Research question

Does the first full `ds006036` production run complete successfully with the new
block-native window contract enabled, and do the resulting H5 files actually
contain the expected stage-blocking and block-native nodes?

## Command used

```powershell
python -m mndm.cli all --dataset ds006036 --config mndm/config/config_ingest_ds006036.yaml --n-jobs 6
```

## Documentation updates completed in the same session

Updated:

- `README.md`
- `mndm/README.md`
- `mndm/Output_variables_guide.md`

The docs now describe:

- block-native dataset config under `block_native.datasets.<id>`
- the new H5 groups `/blocks/*` and `/block_windows/*`
- run-manifest capability reporting for block-native outputs

## Run result

Run directory:

- `E:/Science_Datasets/openneuro/processed/ds006036/neuralmanifolddynamics_ds006036_20260604_115155`

The full pipeline completed successfully with `exit_code: 0`.

`run_manifest.json` reports:

- `counts.h5 = 88`
- `counts.summary_json = 88`
- `counts.qc_summary_json = 88`
- `counts.qc_reliability_json = 88`
- `capabilities.labels_stage = true`
- `capabilities.has_block_native_windows = true`
- `capabilities.counts.h5_with_block_native_windows = 88`
- `capabilities.counts.h5_with_stage = 88`

This confirms a complete subject-level run across the full `ds006036` cohort.

## New H5 nodes confirmed

Inspected example subject:

- `sub-010_eyes_open_photic_photomark/sub-010_eyes_open_photic_photomark.h5`

Confirmed top-level groups:

- `/events`
- `/event_windows`
- `/blocks`
- `/block_windows`

Confirmed stage-blocking-related event columns:

- `/events/inferred_block_id`
- `/events/is_stage_block_event`
- `/events/stage_block_frequency_hz`
- `/events/mapped_stage_code`
- `/events/mapping_mode`
- `/events/mapping_rule`

Confirmed block-native groups and shapes for `sub-010`:

- `/blocks/block_id` shape `(5,)`
- `/blocks/stage_code` shape `(5,)`
- `/blocks/start_sec` shape `(5,)`
- `/blocks/end_sec` shape `(5,)`
- `/blocks/duration_sec` shape `(5,)`
- `/block_windows/block_id` shape `(39,)`
- `/block_windows/window_id_within_block` shape `(39,)`
- `/block_windows/relative_time_in_block_sec` shape `(39,)`
- `/block_windows/distance_to_block_end_sec` shape `(39,)`
- `/block_windows/relative_pos_0_1` shape `(39,)`

Both groups carry:

- `_schema_version = "block_native_v1"`

## Per-subject validation example

For `sub-010`, summarize reported:

- `n_event_rows = 592`
- `n_event_rows_mapped = 592`
- `stage_blocking_enabled = true`
- `stage_frac_labeled = 0.6363636363636364`

Detected raw photic frequencies:

- `5`, `10`, `15`, `20`

Missing expected raw frequencies in this subject:

- `3`, `7`, `25`, `30`

Block-native injection for this subject reported:

- `5 blocks -> 39 windows`

Later subjects in the cohort commonly reported:

- `4 blocks -> 29-30 windows`

So the block-native contract is live, but block count is subject-dependent and
tracks the actual event stream rather than forcing a fixed expected count.

## Additional outputs confirmed

Run-level CSV sidecars were written:

- `regional_mnps_subjects_115155.csv`
- `regional_block_jacobians_subjects_115155.csv`
- `stratified_block_jacobians_subjects_115155.csv`

Run-level stage QC sidecar was also written:

- `stage_mapping_qc.json`

## Important runtime notes

1. `mnps_projection.export_contracts.cohort_anchored=true` was configured, but
   no external anchor was active, so cohort-anchored exports were skipped. This
   is expected and not a run failure.

2. FD censoring was skipped because `framewise_displacement` is not present for
   this EEG dataset. Also expected.

3. There is no built-in CLI `--clean` flag for deleting feature intermediates at
   run time. Cleanup remains a manual post-run action if desired.

## Outcome

The first full `ds006036` block-native production run succeeded end-to-end.
The new stage-blocking-aware event columns and the new block-native H5 groups
are present in real subject outputs, and the run manifest now correctly exposes
block-native capability flags at cohort level.
