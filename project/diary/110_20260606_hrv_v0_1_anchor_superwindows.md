# HRV v0.1 anchor superwindows

## Research question

Implement a first explicit HRV v0.1 surface for Noetic Anchoring Dynamics without changing the canonical MNPS contract:

1. compute longer-window ECG HRV estimates aligned to the MNPS grid,
2. expose them as stable `ecg_hrv_*` feature columns with QC,
3. prefer them inside `anchor_state`,
4. and verify the result on `ds003838`.

## What changed

### ECG feature extraction

- Extended `mndm/src/mndm/features/ecg.py` with an additive HRV-superwindow path under `features.ecg.hrv`.
- New aligned columns:
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
- Added configurable HRV window policy:
  - `superwindow_s`
  - `window_mode`
  - `min_nn_intervals`
  - `min_coverage_fraction`
  - `max_artifact_fraction`
  - `pnn50_threshold_ms`

### Anchor integration

- Updated `mndm/src/mndm/pipeline/anchor_state.py` so anchor indices prefer HRV v0.1 columns when present and fall back to legacy short-window ECG columns otherwise.
- `ecg_quality` inside `anchor_quality` now prefers `ecg_hrv_quality_score` when available.
- The source manifest in `anchor_state` now makes the HRV preference explicit.

### Config and summary surface

- Added default-disabled HRV policy to `mndm/config/config_ingest_common_eeg.yaml`.
- Enabled HRV v0.1 for `ds003838` in `mndm/config/config_ingest_ds003838.yaml` with a 60 s centered superwindow.
- Added `anchor_hrv_v0_1` to per-run `summary.json` manifests in `mndm/src/mndm/pipeline/summary.py`.

## Validation

- Added HRV-specific regression coverage to `mndm/tests/test_anchor_features.py`.
- Added AnchorState preference coverage to `mndm/tests/test_anchor_state.py`.
- Focused regression suites passed:
  - `python -m pytest mndm/tests/test_anchor_features.py mndm/tests/test_anchor_state.py -q`
  - `python -m pytest mndm/tests/test_parallel.py mndm/tests/test_parallel_merge.py mndm/tests/test_anchor_features.py mndm/tests/test_anchor_state.py mndm/tests/test_preprocess_modalities.py mndm/tests/test_bids_index_multimodal.py -q`

## Smoke run

Targeted smoke validation completed successfully with:

```text
python -m mndm.cli all --dataset ds003838 --subject 032 --config mndm/config/config_ingest_ds003838.yaml --out-dir H:/SourceRepo2/NeuralManifoldDynamics/.smoke_hrv_v01 --n-jobs 1
```

Observed outcomes:

- run finished successfully with exit code `0`,
- `features.csv` and `features.parquet` were written,
- `sub-032_digit_span` and `sub-032_rest` both wrote H5 + summary artifacts,
- `digit_span` event-locked sidecars still emitted correctly,
- `features.csv` now includes the new `ecg_hrv_*` columns,
- `sub-032_digit_span/summary.json` includes `anchor_hrv_v0_1`,
- `anchor_state.source_features` shows `sympathetic_index` / `vagal_index` using the new HRV columns.

## Notes and boundaries

- This is intentionally `HRV v0.1`, not a full classical HRV package.
- The implementation focuses on reviewer-legible short-list metrics from longer aligned windows and does not yet add frequency-domain HRV or a separate dedicated `/anchor/hrv/*` HDF5 subtree.
- The current summary/export contract is:
  - raw and robust-z HRV features in the feature tables / H5 feature matrices,
  - explicit `anchor_hrv_v0_1` manifest entry in `summary.json`,
  - HRV-preferred use inside `anchor_state`.
