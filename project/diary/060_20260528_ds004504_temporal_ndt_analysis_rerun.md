# 060 - ds004504 temporal fix + slim ndt-analysis rerun

Date: 2026-05-28

## Context

User requested a narrow ds004504 rerun that still guarantees:
- 9D Jacobian support in downstream analysis (`global_mnps_jacobian_9d`)
- `mdr_median` rows in new analysis parquet outputs
- retained contrast set:
  - `AD_vs_Healthy`
  - `FTD_vs_Healthy`
  - `AD_vs_FTD`

Also requested a temporal-channel sanity check (same legacy temporal channel family as ds006036).

## Root-cause check (temporal region)

`ds004504` raw EEG channels across 88 files were audited and confirmed to use:
- `T3`, `T4`, `T5`, `T6`

not:
- `T7`, `T8`, `TP7`, `TP8` only.

This explained why earlier regional outputs lacked `temporal`.

## Config updates

Updated:
- `mndm/config/config_ingest_ds004504.yaml`

Change:
- `robustness.ensembles.groups.temporal` now includes:
  - `["T7", "T8", "TP7", "TP8", "T3", "T4", "T5", "T6"]`

No other ds004504 toggles were needed; Jacobians, 9D, regional MNPS, and block Jacobians were already enabled.

## MNDM rerun (fresh output root to avoid cache skip)

Command:

`python -m mndm.cli all --dataset ds004504 --config mndm/config/config_ingest_ds004504_m_drive_derivatives_anchor.yaml --out-dir M:/datasets/processed/openneuro_ds004504_temporal_v2 --n-jobs 4 --fit-anchor --anchor-id ds004504_derivatives_cohort_anchor_v2_2_20260528`

Outcome:
- exit code `0`
- run dir:
  - `M:/datasets/processed/openneuro_ds004504_temporal_v2/ds004504/neuralmanifolddynamics_ds004504_20260528_114734`
- 88 subject H5 written
- temporal network materialized:
  - regional logs show `4 networks computed`
  - block Jacobian logs show `... for 4 networks`

Feature header check:
- groups present:
  - `frontal`, `central`, `parietal_occipital`, `temporal`

## ndt-analysis slim rerun (from analysis repo)

Repo:
- `H:/SourceRepo2/NoeticDiffusion/ndt-analysis`

### 1) Raw H5 -> cleaned parquets

Command:

`python ndt-analysis.py run --config config/ndt-analysis-ds004504.yaml --input-root M:/datasets/processed/openneuro_ds004504_temporal_v2/ds004504/neuralmanifolddynamics_ds004504_20260528_114734 --output-root ../data/cleaned --workers 6 --analyses global_mnps_jacobian_9d,global_mnps_jacobian_3d,regional_3d,tier2_emmi,global_mnps_jacobian_9d_block_jacobian`

Outcome:
- exit code `0`
- cleaned outputs include:
  - `ds004504_global_mnps_jacobian_3d_20260528_140218.parquet`
  - `ds004504_global_mnps_jacobian_9d_20260528_140221.parquet`
  - `ds004504_global_mnps_jacobian_9d_block_jacobian_20260528_140224.parquet`
  - `ds004504_regional_3d_20260528_140227.parquet`

### 2) Cleaned -> analysis parquets

Command:

`python ndt-analysis.py analyze --config config/analysis-ds004504.yaml --cleaned-root ../data/cleaned --output-root ../data/analysis --analyses global_mnps_jacobian_9d,global_mnps_jacobian_3d,regional_3d,tier2_emmi,global_mnps_jacobian_9d_block_jacobian --contrasts AD_vs_Healthy,FTD_vs_Healthy,AD_vs_FTD`

Outcome:
- exit code `0`
- wrote:
  - `ds004504_subject_metrics_20260528_140304.parquet`
  - `ds004504_contrast_results_20260528_140304.parquet`
  - `ds004504_qc_summary_20260528_140304.parquet`
- rows:
  - `subject_metrics`: 11352
  - `contrast_results`: 393

## Acceptance criteria verification

Verified in latest analysis parquet outputs:

- `analysis_type == "global_mnps_jacobian_9d"` and `metric == "mdr_median"`
  - present in `subject_metrics`: `88` rows
  - present in `contrast_results`: `3` rows

Contrasts present:
- `AD_vs_Healthy`
- `FTD_vs_Healthy`
- `AD_vs_FTD`

Requested minimal + optional blocks represented in `subject_metrics` analysis types:
- `global_mnps_jacobian_9d`
- `global_mnps_jacobian_3d`
- `global_mnps_jacobian_9d_block_jacobian`
- `regional_3d`
- `tier2_emmi`
