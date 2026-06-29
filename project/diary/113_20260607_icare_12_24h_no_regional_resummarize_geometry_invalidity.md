# 113 - 2026-06-07 - i-care 12-24h no-regional summarize rerun after geometry invalidity policy

Date: 2026-06-07

## Context

User requested a fresh summarize rerun for the I-CARE part1 12-24h no-regional
configuration after the new default geometry invalidity / infinity-handling
policy was added to the summarize pipeline.

Config checked:

- `mndm/config/config_ingest_physionet_i-care_2_1_part1_12_24h_no_regional.yaml`

## Config verification

Confirmed the no-regional overlay is configured as intended:

- `regional_mnps.enabled: false`
- `regional_mnps.block_jacobians.enabled: false`
- `mnps_9d.enabled: true`
- `mnps_9d.block_jacobians.enabled: false`

This keeps 9D coordinates enabled while preventing the much larger 9D block
Jacobian exports.

## Rerun decision

Only `summarize` needed to be rerun.

Reason:

- the new invalidity policy is implemented in summarize-time geometry handling
- it affects exported geometry, Jacobian retention, summary manifests, and HDF5
  provenance
- it does not require recomputing the existing feature table

## Existing input retained

Reused:

- `E:/Science_Datasets/physionet/processed_part1_cpc15_12_24h_no_regional_subject_anchor/physionet_icare_2_1/features.parquet`

## Action taken

Started a fresh summarize rerun with:

- dataset: `physionet_icare_2_1`
- config: `mndm/config/config_ingest_physionet_i-care_2_1_part1_12_24h_no_regional.yaml`
- output root: `E:/Science_Datasets/physionet/processed_part1_cpc15_12_24h_no_regional_subject_anchor`
- workers: `4`

## Early status

The rerun started normally and reached:

- `INFO: Summarizing physionet_icare_2_1`
