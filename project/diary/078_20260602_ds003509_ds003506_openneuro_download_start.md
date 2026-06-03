# Session Diary 078 - ds003509 and ds003506 OpenNeuro download start

Date: 2026-06-02

## Goal

Start fallback DataLad downloads for OpenNeuro datasets `ds003509` and `ds003506`
to the user-requested storage root:

- `M:/datasets/received/openneuro`

## Download command

Started in background from the repository root with explicit `PYTHONPATH` so the
repo modules resolve cleanly:

```powershell
$env:PYTHONPATH="H:/SourceRepo2/NeuralManifoldDynamics/core/src;H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics"
python "H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src/datalad_fallback.py" --dataset ds003509 ds003506 --data-dir "M:/datasets/received/openneuro" --jobs 4 --on-failure continue --batch-targets 8 --report-every 1 --targets-preview 8
```

## Observed startup status

Verified from the background terminal output:

- target root `M:/datasets/received/openneuro` exists
- `ds003509` clone started from `https://github.com/OpenNeuroDatasets/ds003509.git`
- object enumeration / receive / delta-resolve completed for the clone
- dataset install completed at `M:/datasets/received/openneuro/ds003509`
- DataLad content fetch started for `ds003509`
  - `targets=66`
  - derivatives skipped by default
  - batched into `9` fetch batches
- `ds003506` is queued in the same command and will start after the current
  dataset stage advances/completes

## Current state

The fallback download was later aborted externally before completion.

Observed end state after the abort:

- the terminal ended with `exit_code: unknown`, which is consistent with an
  external interruption rather than a clean script exit
- `ds003509` reached clone/install and had entered the DataLad content-fetch
  phase
- `ds003506` had not yet started cloning at the time of the abort
- partial checkout remains at `M:/datasets/received/openneuro/ds003509`

## Follow-up restart for ds003506

The user then requested a retry for `ds003506` only.

Restarted in background with:

```powershell
$env:PYTHONPATH="H:/SourceRepo2/NeuralManifoldDynamics/core/src;H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics"
python "H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src/datalad_fallback.py" --dataset ds003506 --data-dir "M:/datasets/received/openneuro" --jobs 4 --on-failure continue --batch-targets 8 --report-every 1 --targets-preview 8
```

Observed restart status:

- existing checkout detected at `M:/datasets/received/openneuro/ds003506`
- content fetch restarted immediately without recloning
- DataLad reported:
  - `targets=66`
  - `9` batches
- before restart verification, `153 / 168` EEG signal files were still tiny
  placeholder files, confirming the dataset was incomplete before the retry

The retry remains in progress after startup verification.

## Retry outcome for ds003506

The background terminal later ended with:

- `exit_code: unknown`
- elapsed runtime about `510` seconds

Despite that terminal status, on-disk verification after the run showed that the
dataset fetch had materially completed:

- `ds003506` signal files with placeholder-like tiny sizes: `0 / 168`
- total non-git file count: `827`
- total non-git bytes: about `17.4 GB`

Representative large EEG payload files were present under:

- `M:/datasets/received/openneuro/ds003506/sub-029/ses-02/eeg/...fdt`
- `M:/datasets/received/openneuro/ds003506/sub-041/ses-01/eeg/...fdt`

Current interpretation:

- terminal bookkeeping marked the job as aborted/unknown
- but `ds003506` now appears fully downloaded for the main EEG signal payloads
