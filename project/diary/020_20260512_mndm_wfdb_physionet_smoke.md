# 020 — 2026-05-12 — MNDM WFDB Support + PhysioNet 0332 Smoke

## Session goal

Verify that the next MNDM pipeline stage can read downloaded I-CARE WFDB files
from `E:/Science_Datasets/physionet/received/i-care_2_1_random100_longitudinal/training/0332`
and run `features`/`summarize` end-to-end.

## Implemented

1. Added WFDB ingestion support in MNDM preprocessing:
   - `mndm/src/mndm/preprocess.py`
   - New optional dependency import: `wfdb`
   - New path handler:
     - `.hea` routed to `preprocess_wfdb(...)`
   - `preprocess_wfdb(...)`:
     - loads WFDB record from `.hea` + paired signal file (`.mat`/`.dat`)
     - maps channel types (EEG/ECG/EOG/EMG/RESP heuristic)
     - applies crop, resample, notch, bandpass, and average rereference for EEG
     - exports MNDM-compatible `signals/channels/meta` payload
     - returns clear error when paired signal file is missing

2. Extended file indexing for PhysioNet WFDB:
   - `mndm/src/mndm/bids_index.py`
   - Added `.hea` to EEG index extensions.
   - Added numeric subject fallback support for non-BIDS paths (e.g., `0332`).
   - Added guard: skip `.hea` entries if paired `.mat`/`.dat` is not present
     (prevents false failures during active download).

3. Updated memory heuristic for WFDB headers:
   - `mndm/src/mndm/orchestrate.py`
   - `.hea` estimator now uses sibling `.mat`/`.dat` size when available.

4. Added PhysioNet-specific MNDM config:
   - `mndm/config/config_ingest_physionet_i-care_2_1.yaml`
   - Includes:
     - dataset root mapping to random100 training folder
     - WFDB source metadata
     - conservative preprocess profile for EEG
     - filename regex parsing for subject/run/acq in summarize grouping

5. Dependency updates:
   - `requirements.txt`
   - Added `wfdb==4.3.1`
   - Updated `pandas` to `3.0.3` for compatibility with current WFDB dependency constraints.

6. Improved summarize subject filtering for non-BIDS sources:
   - `mndm/src/mndm/pipeline/summary.py`
   - Subject filtering now uses parsed file entities (including
     `metadata_extraction.datasets.<id>.filename_parse`) instead of only a raw
     filename substring match.
   - Enables `--subject 0332` to work for I-CARE-style filenames.

## Validation performed

- MNDM prerequisite check:
  - Indexed PhysioNet `.hea` records successfully.
  - Dataset/config paths resolved.

- MNDM features smoke (`subject 0332`):
  - Successfully read and processed WFDB EEG records.
  - Generated:
    - `features.csv` / `features.parquet`
    - intermediate per-file JSON
    - QC artifact JSON

- MNDM summarize smoke:
  - Initial run failed to group because subject IDs were not parsed from filenames.
  - After adding `metadata_extraction.datasets.<id>.filename_parse` regex to config,
    summarize produced MNPS HDF5 outputs per run/acq for subject `0332`.
  - After subject-filter fix, `summarize --subject 0332` also works as expected.
  - One very short segment was skipped by coverage threshold (`min_seconds=120`), as expected.

## Artifacts

- Feature smoke output:
  - `E:/Science_Datasets/physionet/processed_smoke_wfdb/physionet_icare_2_1/features.parquet`
- Summarize run directory:
  - `E:/Science_Datasets/physionet/processed_smoke_wfdb/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260512_093435/`

## Notes

- During active downloader operation, partial `.hea` arrivals are expected.
  The new index guard now excludes incomplete pairs until signal files exist.
