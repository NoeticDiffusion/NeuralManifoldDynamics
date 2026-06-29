# 112 - 2026-06-07 - i-care 12-24h summarize rerun after geometry invalidity policy

Date: 2026-06-07

## Context

User requested a rerun of the I-CARE part1 12-24h regional export after the
new standard geometry invalidity / infinity-handling policy was implemented.

Relevant policy note:

- `project/diary/104_20260605_standard_geometry_invalidity_policy.md`

## Decision

Only `summarize` needed to be rerun.

Reason:

- the new policy is implemented in the summarize pipeline
- it changes exported geometry handling after projection / before Jacobian export
- it affects `summary.json`, `qc_summary.json`, HDF5 provenance, and Jacobian
  window retention
- it does not require recomputing `features.parquet`

## Existing input retained

Reused the already computed feature table:

- `E:/Science_Datasets/physionet/processed_part1_cpc15_12_24h_regional_subject_anchor/physionet_icare_2_1/features.parquet`

## Action taken

Started a fresh summarize rerun with:

- dataset: `physionet_icare_2_1`
- config: `mndm/config/config_ingest_physionet_i-care_2_1_part1_12_24h_regional_subject_anchor.yaml`
- output root: `E:/Science_Datasets/physionet/processed_part1_cpc15_12_24h_regional_subject_anchor`
- workers: `4`

## Early status

The rerun started normally and reached:

- `INFO: Summarizing physionet_icare_2_1`

No immediate startup failure was observed.
