# Session Diary 052 - ds006036 OpenNeuro download start (with fallback)

Date: 2026-05-27

## Goal

Set up and start OpenNeuro ingest for `ds006036` (AD/FTD/Healthy EEG cohort), using `ds004504` as template. If primary downloader fails, switch to fallback downloader.

## Config work

Added new MNDM dataset config:

- `mndm/config/config_ingest_ds006036.yaml`

Template basis:

- `mndm/config/config_ingest_ds004504.yaml`

Main settings:

- `datasets: [ds006036]`
- enabled 9D block-jacobians (`mde_families_v1`)
- enabled CSD preprocessing and regional MNPS (same profile as ds004504 template)
- metadata group normalization:
  - `A -> AD`
  - `F -> FTD`
  - `C -> Healthy`
- `condition.default: eyes_open_photic`
- `task.from_filename: true`

YAML validation: OK.

## Download attempts

### Primary attempt (openneuro_ingest)

Command:

- `python -m openneuro.cli download --dataset ds006036 --config openneuro_ingest/config/config_ingest.yaml`

Result:

- failed after retries due broken `openneuro-py` CLI entrypoint in current env:
  - `ModuleNotFoundError: No module named 'openneuro._cli'`

### Fallback attempt (DataLad)

Started fallback command in background:

- `.venv/Scripts/python.exe openneuro_ingest/src/datalad_fallback.py --dataset ds006036 --config openneuro_ingest/config/config_ingest.yaml --jobs 4 --on-failure continue`

Observed runtime status:

- clone of `https://github.com/OpenNeuroDatasets/ds006036.git` started and completed metadata/object phases
- install completed at `E:/Science_Datasets/openneuro/received/ds006036`
- content fetch stage started (`targets=97`, derivatives skipped by default)

Fallback download remains running.
