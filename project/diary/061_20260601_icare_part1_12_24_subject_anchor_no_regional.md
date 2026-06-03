# 061 - i-care part1 12-24h subject-anchor no-regional

Date: 2026-06-01

## Context

User requested:
- no ComBat / normalization (still under development),
- no regional MNPS outputs,
- 9D + 9D block Jacobians enabled,
- within-subject anchor only,
- run `features` then `summarize`.

## Config updates

Updated:
- `mndm/config/config_ingest_physionet_i-care_2_1_12_24.yaml`
- `mndm/config/config_ingest_physionet_i-care_2_1_part1_12_24h_no_regional.yaml`

Changes applied:
- switched dataset root to downloaded 12-24h subset:
  - `E:/Science_Datasets/physionet/received/i-care_2_1_part1_cpc15_longitudinal_12_24h/training`
- kept normalization disabled (`normalization.enabled: false`)
- kept regional disabled (`regional_mnps.enabled: false`, stratified false, block_jacobians false)
- enabled 9D block Jacobians (`mnps_9d.block_jacobians.enabled: true`)
- explicitly forced no cohort anchor:
  - `mnps_projection.anchor.enabled: false`

## Run launch

Started chained run (PowerShell-compatible sequence):

1) `features`
2) `summarize` only if features exits with code 0

Config:
- `mndm/config/config_ingest_physionet_i-care_2_1_part1_12_24h_no_regional.yaml`

Output root:
- `E:/Science_Datasets/physionet/processed_part1_cpc15_12_24h_no_regional_subject_anchor`

Current status at log capture:
- resumed from prior partial intermediates
- skipped 58 already processed files
- continuing features on remaining files before summarize phase starts
