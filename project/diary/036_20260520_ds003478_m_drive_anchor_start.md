# 036 — 2026-05-20 — ds003478 M-drive anchor start

## Context
- User requested starting MNDM processing for `ds003478` from `M:/datasets/received/openneuro/ds003478`.
- Requested explicit inclusion of `participants.tsv` and cohort-anchored coordinate export.

## Changes made
- Added run-specific config:
  - `mndm/config/config_ingest_ds003478_m_drive_anchor.yaml`
- Config behavior:
  - Imports `config_ingest_ds003478.yaml`.
  - Overrides `paths.dataset_received_dirs.ds003478` to `M:/datasets/received/openneuro/ds003478`.
  - Sets explicit participants table path:
    - `metadata_extraction.datasets.ds003478.participants.path: "participants.tsv"`.

## Run started
- Started one-shot full pipeline with cohort anchor fit:
  - `mndm all --dataset ds003478 --config mndm/config/config_ingest_ds003478_m_drive_anchor.yaml --out-dir M:/datasets/processed/openneuro --n-jobs 4 --fit-anchor --anchor-id ds003478_depression_eeg_cohort_anchor_v2_1_20260520`
- Early status:
  - File index created (`243` EEG files, `122` subjects).
  - Intermediate and QC JSON outputs are actively being written.
  - Run is currently in features stage and progressing.

## Notes
- Existing long-running jobs (PhysioNet summarize/download) were left untouched.
- Worker count was set to `4` to reduce resource contention while still progressing the new run.
