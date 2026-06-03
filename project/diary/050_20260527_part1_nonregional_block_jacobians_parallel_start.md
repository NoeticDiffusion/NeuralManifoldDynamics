# Session Diary 050 - Part1 non-regional block-Jacobians (no ComBat) parallel start

Date: 2026-05-27

## Goal

User requested:

- clone part1 YAML for block-Jacobians,
- keep run non-regional,
- skip ComBat,
- keep simple normalization,
- start this run in parallel.

## Config update

Updated:

- `mndm/config/config_ingest_physionet_i-care_2_1_part1_0_12h_regional_block_jacobians.yaml`

Key settings now:

- `regional_mnps.enabled: false` (non-regional run)
- `mnps_9d.enabled: true`
- `mnps_9d.block_jacobians.enabled: true`
  - `preset: mde_families_v1`
  - `csv_output: stratified_block_jacobians_subjects.csv`
- `normalization.enabled: false` (ComBat disabled)

Simple normalization remains active through inherited projection defaults from common EEG config:

- `mnps_projection.normalize: robust_z`
- `mnps_projection.feature_standardization` pipelines (robust_z/clip family)

## Run start

Started summarize in background (parallel to ongoing run):

- `python -m mndm.cli summarize --dataset physionet_icare_2_1 --config "config/config_ingest_physionet_i-care_2_1_part1_0_12h_regional_block_jacobians.yaml" --n-jobs 16`

Early log confirms startup + QC filter pass and sidecar write:

- `INFO: Summarizing physionet_icare_2_1`
- `INFO: Dropped 90014/2936906 epochs by qc_ok_eeg (policy=eeg_only)`
- `INFO: Wrote JSON: .../normalization_report.json`
