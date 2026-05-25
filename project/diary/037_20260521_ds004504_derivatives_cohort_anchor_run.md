# 037 — 2026-05-21 — ds004504 derivatives cohort-anchor run

## Context
- User requested a full `features + summarize` run for `ds004504` from:
  - `M:/datasets/received/openneuro/ds004504/derivatives`
- User requested cohort-anchored output (instead of subject-anchored primary contract).

## Config and launch
- Added run-specific config:
  - `mndm/config/config_ingest_ds004504_m_drive_derivatives_anchor.yaml`
- Key overrides:
  - `paths.dataset_received_dirs.ds004504 = M:/datasets/received/openneuro/ds004504/derivatives`
  - `metadata_extraction.datasets.ds004504.participants.path = ../participants.tsv`
- Launched:
  - `mndm all --dataset ds004504 --config mndm/config/config_ingest_ds004504_m_drive_derivatives_anchor.yaml --out-dir M:/datasets/processed/openneuro --n-jobs 4 --fit-anchor --anchor-id ds004504_derivatives_cohort_anchor_v2_1_20260521`

## Outcome
- Run completed successfully (`exit_code: 0`).
- Run directory:
  - `M:/datasets/processed/openneuro/ds004504/neuralmanifolddynamics_ds004504_20260521_082947`
- Outputs:
  - `88` HDF5 files
  - `run_manifest.json` present
  - One-shot fitted anchor artifact present:
    - `anchors/ds004504_derivatives_cohort_anchor_v2_1_20260521.json`

## Contract verification
- Sample H5 confirms:
  - `primary_coordinate_contract = cohort_anchored`
  - `anchor_id = ds004504_derivatives_cohort_anchor_v2_1_20260521`
  - `feature_anchors` group present
  - both subject-anchored and cohort-anchored 3D/9D coordinate layers present
- Participant provenance confirms table source resolves to:
  - `M:/datasets/received/openneuro/ds004504/derivatives/../participants.tsv`
