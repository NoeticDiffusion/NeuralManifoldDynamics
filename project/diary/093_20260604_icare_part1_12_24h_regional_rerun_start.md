# 093 - 2026-06-04 - i-care part1 12-24h regional rerun start

Date: 2026-06-04

## Context

The earlier 12-24h part1 run used the no-regional config and failed on four EEG
records that were later confirmed to have corrupted local `.mat` payloads.

Those four records were re-downloaded with checksum verification and now pass
direct `wfdb.rdrecord(...)` reads:

- `1005_022_024_EEG`
- `1008_019_026_EEG`
- `1009_027_033_EEG`
- `0299_027_041_EEG`

## Config added

Added a dedicated first-pass regional config:

- `mndm/config/config_ingest_physionet_i-care_2_1_part1_12_24h_regional_subject_anchor.yaml`

Key choices:

- same 12-24h CPC1/CPC5 dataset root:
  - `E:/Science_Datasets/physionet/received/i-care_2_1_part1_cpc15_longitudinal_12_24h/training`
- normalization disabled
- regional MNPS enabled
- stratified regional outputs enabled
- block Jacobians disabled for:
  - `regional_mnps`
  - `mnps_9d`
- no cohort anchor:
  - `mnps_projection.anchor.enabled: false`

## Launch

Started a fresh chained run:

1. `features`
2. `summarize` only if `features` exits successfully

Config:

- `mndm/config/config_ingest_physionet_i-care_2_1_part1_12_24h_regional_subject_anchor.yaml`

Dedicated output root:

- `E:/Science_Datasets/physionet/processed_part1_cpc15_12_24h_regional_subject_anchor`

## Intent

This run is the first clean regional 12-24h baseline:

- after repairing the four corrupted EEG records
- with regional MNPS available
- without block-Jacobian expansion

This should make it easier to inspect whether the repaired subset is now stable
before enabling heavier regional derivatives.
