# Session Diary 121 - ds007526 OpenNeuro download start

Date: 2026-06-09

## Goal

Start fallback DataLad download for OpenNeuro dataset `ds007526` to the
user-requested storage root:

- `N:/received`

## Download command

Started from the repository root using the repository fallback downloader:

```powershell
python "H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src/datalad_fallback.py" --dataset ds007526 --data-dir "N:/received" --jobs 4 --on-failure continue --batch-targets 8 --report-every 1 --targets-preview 8
```

## Observed startup status

Verified after launch:

- target root `N:/received` exists
- active Python environment is `C:/Program Files/Python312/python.exe`
- `datalad` imported successfully with version `1.2.3`
- clone started from `https://github.com/OpenNeuroDatasets/ds007526.git`
- clone/install completed at `N:/received/ds007526`
- DataLad content fetch started for `ds007526`
  - `targets=154`
  - derivatives skipped by default
  - batched into `20` fetch batches
- the terminal log became quiet after `get()` started, but runtime inspection
  showed active `git annex get ... -J4 -- .` child processes
- on-disk footprint increased during monitoring from about `0.817 GB` to
  `0.832 GB`, indicating continued download progress despite minimal logging

## Current state

The fallback download completed successfully.

## Outcome

Observed final terminal status:

- all `20` fetch batches completed
- script ended with `exit_code: 0`
- terminal elapsed runtime was about `5203` seconds
- script reported `All requested datasets fetched successfully.`

Quick on-disk verification after completion:

- dataset path present at `N:/received/ds007526`
- total file count under the dataset tree: `1827`
- total file size under the dataset tree: about `4.264 GB`
