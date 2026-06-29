# HRV block-native export and ds003838 validation

## Research question

Expose the raw HRV v0.1 surface directly in block-native sidecars so downstream reviewer analyses can work from block-level tables without first joining back through `features.csv`.

## What changed

### Block-native raw HRV export

- Extended `mndm/src/mndm/pipeline/block_native_export.py` to export selected raw feature columns directly from `MNPSPayload.features_raw_*`.
- The block-native sidecar now includes:
  - `ecg_hrv_hr_mean_bpm`
  - `ecg_hrv_ibi_mean_ms`
  - `ecg_hrv_sdnn_ms`
  - `ecg_hrv_rmssd_ms`
  - `ecg_hrv_pnn50`
  - `ecg_hrv_nn_count`
  - `ecg_hrv_artifact_fraction`
  - `ecg_hrv_coverage_fraction`
  - `ecg_hrv_quality_score`
  - `qc_ok_ecg_hrv`
- Added `raw_hrv_feature_columns` to the block-native manifest entry.

### ds003838 block-native source

- Added a `block_native.datasets.ds003838` config block in `mndm/config/config_ingest_ds003838.yaml`.
- For this dataset, block-native is driven by **derived task-state segments**, not the raw one-second digit events.
- Implemented this by extending:
  - `mndm/src/mndm/pipeline/block_native_config.py` with `use_derived_task_state_segments`
  - `mndm/src/mndm/pipeline/summary.py` so block-native can consume `task_state_label` segment intervals as synthetic duration-events.

This yields block-native windows for `rest`, `listen`, `mem5`, `mem9`, and `mem13`, which is the correct sustained-timescale surface for HRV in `ds003838`.

## Tests

- Updated `mndm/tests/test_block_native_export.py` so the block-native export test now verifies raw `ecg_hrv_*` columns are present in rows.
- Focused regression passed:

```text
python -m pytest mndm/tests/test_block_native_export.py mndm/tests/test_anchor_features.py mndm/tests/test_anchor_state.py -q
```

## Smoke validation

Successful targeted smoke run:

```text
python -m mndm.cli all --dataset ds003838 --subject 032 --config mndm/config/config_ingest_ds003838.yaml --out-dir H:/SourceRepo2/NeuralManifoldDynamics/.smoke_hrv_blocknative --n-jobs 1
```

### Confirmed outputs

- `sub-032_digit_span/block_native_windows.csv`
- `sub-032_digit_span/block_native_windows.parquet`
- `sub-032_rest/block_native_windows.csv`
- `sub-032_rest/block_native_windows.parquet`
- `block_native_qc.json`

### Confirmed behavior

- `digit_span` produced `96` derived task-state blocks and `343` block-native windows.
- `rest` produced `1` block and `55` block-native windows.
- The `block_native_windows.csv` header includes the new raw HRV columns directly.
- `summary.json` records `block_native.derived_from = "task_state_label_segments"`.

## Full dataset run

Started full validation run:

```text
python -m mndm.cli all --dataset ds003838 --config mndm/config/config_ingest_ds003838.yaml --out-dir H:/SourceRepo2/NeuralManifoldDynamics/.full_ds003838_hrv_blocknative --n-jobs 2
```

Current status during this diary entry:

- the memory heuristic reduced runtime to `1` worker,
- the run is progressing through real subjects without HRV/block-native exceptions,
- multiple subjects (`sub-013` through `sub-019` at time of inspection) have already completed ECG/PPG/pupil preprocessing and feature emission successfully.

The full cohort summarize/export phase was still running at the time of writing, so this diary records a **successful smoke validation plus a stable in-progress full-dataset validation** rather than a completed full-cohort result.
