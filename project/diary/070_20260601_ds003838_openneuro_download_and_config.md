# Session Diary 070 - ds003838 OpenNeuro download and config

Date: 2026-06-01

## Goal

Download OpenNeuro `ds003838` via DataLad to long-term storage on `G:` and add an
initial MNDM EEG ingest overlay for the dataset.

## Dataset facts verified

Verified from the cloned dataset metadata and public dataset records:

- dataset id: `ds003838`
- DOI: `10.18112/openneuro.ds003838.v1.0.6`
- BIDS version: `1.1.1`
- EEG files are stored as EEGLAB `.set`
- EEG recordings exist for:
  - `task-rest`
  - `task-memory`
- the `task-memory` recording contains both passive listening control trials and
  memorization trials in the same run

## Download

Target root requested by user:

- `G:/Science_Datasets_longtime_storage/openneuro/received`

Started DataLad fallback download in background with:

```powershell
python "H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src/datalad_fallback.py" --dataset ds003838 --data-dir "G:/Science_Datasets_longtime_storage/openneuro/received" --jobs 4 --on-failure continue --batch-targets 8 --report-every 1 --targets-preview 8
```

Observed startup status:

- clone started at `G:/Science_Datasets_longtime_storage/openneuro/received/ds003838`
- top-level BIDS tree became available immediately
- dataset metadata files were readable:
  - `dataset_description.json`
  - `participants.tsv`
  - `README`
- tracked EEG files confirmed in the clone:
  - `sub-032/eeg/sub-032_task-rest_eeg.set`
  - `sub-032/eeg/sub-032_task-memory_eeg.set`

The full annex content download remains in progress after startup verification.

## Repo changes

Added:

- `mndm/config/config_ingest_ds003838.yaml`

Main config choices:

- point `received_dir` to `G:/Science_Datasets_longtime_storage/openneuro/received`
- point `processed_dir` to `G:/Science_Datasets_longtime_storage/processed/openneuro`
- keep a per-dataset root override for `ds003838`
- set source metadata to OpenNeuro `v1.0.6`
- enable regional MNPS, stratified outputs, and 9D block Jacobians
- keep CSD disabled in the first pass
- normalize BIDS `task-memory` to `digit_span`
- keep `condition` unset for now because the meaningful control vs memory split is
  event-level within the same recording, not filename-level

## Validation

Configuration design was validated against the cloned BIDS tree:

- `participants.tsv` contains usable demographic covariates such as `age` and `sex`
- `task` labels parse cleanly from filenames
- event sidecars exist for both `task-rest` and `task-memory`
- memory-event labels include trial-level distinctions such as:
  - `control`
  - `memory`
  - sequence lengths `05`, `09`, `13`
  - correctness coding

No event-contract was added in this step; that remains a later refinement once the
initial ingest path is confirmed.
