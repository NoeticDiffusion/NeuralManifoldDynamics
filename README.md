# NeuralManifoldDynamics

Monorepo for data ingest, feature extraction, MNPS summarization, and downstream artifact generation for EEG and fMRI workflows.

This root README is intentionally high-level. Package-specific usage, schema details, and command references live in each subproject.

## What This Repo Contains

The repository is organized around a shared pipeline:

1. Acquire or locate source datasets.
2. Index and preprocess raw recordings.
3. Compute per-epoch or per-window features.
4. Project features into MNPS spaces.
5. Write subject-level and run-level outputs for analysis and QC.

## Main Packages

- `mndm`: Core MNPS pipeline. Handles `features`, `summarize`, `all`, `pack`, and structure validation. See `mndm/README.md`.
- `openneuro_ingest`: OpenNeuro-facing download and ingest utilities. Use this when pulling public datasets before MNDM processing.
- `apollo_ingest`: Ingest helpers for Apollo-style sources used in this repo.
- `vitaldb_ingest`: Ingest helpers for VitalDB-style sources used in this repo.
- `core`: Shared config loading, path resolution, I/O helpers, and common utilities used across packages.

## Typical Workflow

For most projects, the workflow is:

```text
download or locate data -> ingest/index -> mndm features -> mndm summarize
```

If the dataset is already present on disk, you usually work directly with `mndm`.

## Quick Start

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

If you run directly from this source tree, set `PYTHONPATH` so the package modules resolve correctly:

```powershell
$repo_root="C:/path/to/NeuralManifoldDynamics"
$env:PYTHONPATH="$repo_root/mndm/src;$repo_root/core/src;$repo_root/openneuro_ingest/src;$repo_root/apollo_ingest/src;$repo_root/vitaldb_ingest/src"
```

Example MNDM run:

```powershell
python -m mndm.cli all --dataset ds003490 --config mndm/config/config_ingest_ds003490.yaml --n-jobs 12
```

## Where To Read Next

- MNDM usage and output contracts: `mndm/README.md`
- MNDM command reference: `mndm/Command_cheat_sheet.md`
- MNDM output schema details: `mndm/Output_variables_guide.md`
- OpenNeuro ingest details: `openneuro_ingest/`

## Repository Layout

```text
NeuralManifoldDynamics/
├── core/
├── mndm/
├── openneuro_ingest/
├── apollo_ingest/
├── vitaldb_ingest/
├── requirements.txt
└── README.md
```

## Outputs At A Glance

Most processed outputs are written under a dataset-specific processed directory. In current MNDM runs, summarized outputs typically appear in run folders named like:

```text
<processed>/<dataset>/neuralmanifolddynamics_<dataset>_<timestamp>/
```

Those runs usually contain:

- `run_manifest.json`
- `features_snapshot.json`
- per-subject or per-run subdirectories with `summary.json`, QC JSON, and HDF5 outputs

## Development Notes

- `requirements.txt` is shared from the repo root.
- `pyarrow` is recommended so feature tables can use parquet cleanly.
- Worker count and memory budget are controlled from the CLI, especially for `mndm.cli features` and `mndm.cli all`.


## Read the docs

https://neuralmanifolddynamics.readthedocs.io/en/latest/index.html


## License

Se LICENSE in the root folder.
