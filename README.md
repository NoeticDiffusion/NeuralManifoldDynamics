# NoeticDiffusionDataIngest

This repository hosts the data ingest and processing stack used for Noetic
Diffusion research. It is organized into three Python packages that separate
concerns and make the pipeline easier to maintain:

- `openneuro_ingest` — download and staging of OpenNeuro datasets
- `mndm` — Meta Noetic Diffusion Model processing: features, MNPS, summaries
- `core` — shared utilities, schemas, IO, and configuration helpers

## What this repo does

- Ingests BIDS datasets (primarily from OpenNeuro) and builds file indices.
- Extracts features and computes MNPS (including regional and stratified MNPS).
- Writes per-run HDF5 outputs plus JSON/CSV summaries for analysis.

## Quick start

Install dependencies in your environment, then run the pipeline from the repo
root. Adjust paths to match your machine.

```bash
# Download data (OpenNeuro only)
python -m openneuro.cli download --dataset ds003059 --data-dir G:\Science_Datasets

# Summarize with MNDM
python -m mndm.cli summarize --dataset ds003059 --data-dir G:\Science_Datasets --config mndm\config\config_ingest.yaml
```

## Configuration

Pipeline configuration lives under `mndm/config/`. The main config file is:

- `mndm/config/config_ingest.yaml`

You can also define a simple data source descriptor in:

- `mndm/config/source.yaml`

## RichSleep extraction pipeline

This repository also contains the `richsleep` extraction pipeline, which
ingests EDF + annotation files from the RichSleep dataset and converts them
into compressed HDF5 assets ready for Noetic Diffusion Theory analysis.

## Docs

- Operator notes: `docs/LLM_instruction.md`
- MNDM details and commands: `mndm/README.md`
- CLI cheat sheet: `mndm/Command_cheat_sheet.md`

## Notes

- This repo supports optional dependencies (`h5py`, `mne`, `scipy`); some steps
  will be skipped if these are not installed.
- BIDS is the standard dataset format used by OpenNeuro and this pipeline.