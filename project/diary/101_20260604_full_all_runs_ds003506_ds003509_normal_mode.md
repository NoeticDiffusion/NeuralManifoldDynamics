# 101 20260604 full_all_runs_ds003506_ds003509_normal_mode

## Context

After disabling block-native for `ds003506` and `ds003509`, we ran full
cohort `features + summarize` workflows (`mndm.cli all`) for both datasets.

## Commands

```powershell
python -m mndm.cli all --dataset ds003506 --config mndm/config/config_ingest_ds003506.yaml --n-jobs 1
python -m mndm.cli all --dataset ds003509 --config mndm/config/config_ingest_ds003509.yaml --n-jobs 1
```

## Results

### ds003506

- run dir: `M:\datasets\processed\openneuro\ds003506\neuralmanifolddynamics_ds003506_20260604_181839`
- command exit code: `0`
- features stage:
  - detected existing `features.parquet` content for all files
  - skipped recomputation (`Skipped 84 already processed files`)
- summarize stage:
  - completed full cohort (`h5=84`, `summary_json=84`)
  - no `run_errors.json`
  - block-native remained disabled:
    - `capabilities.has_block_native_windows=false`
    - `capabilities.counts.h5_with_block_native_windows=0`

### ds003509

- run dir: `M:\datasets\processed\openneuro\ds003509\neuralmanifolddynamics_ds003509_20260604_185413`
- command exit code: `0`
- features stage:
  - detected existing `features.parquet` content for all files
  - skipped recomputation (`Skipped 84 already processed files`)
- summarize stage:
  - completed full cohort (`h5=84`, `summary_json=84`)
  - no `run_errors.json`
  - block-native remained disabled:
    - `capabilities.has_block_native_windows=false`
    - `capabilities.counts.h5_with_block_native_windows=0`

## Notes

- Coordinate contracts are unchanged from prior normal-mode behavior:
  - requested: `subject_anchored`, `cohort_anchored`
  - realized: `subject_anchored`
  - cohort remains skipped without an active feature anchor.
