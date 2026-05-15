# DANDI Ingest

Config-driven DANDI asset ingestion for this repository.

This package handles:
- listing assets from a Dandiset,
- writing machine-readable manifests,
- downloading selected assets to local storage,
- probing local NWB files to summarize structure and modality hints,
- emitting dataset-adapter triage notes for downstream MNDM runs.

## Requirements

- Python 3.11+
- `dandi` (required for DANDI API listing)
- `h5py` (recommended for basic HDF5/NWB probing)
- `pynwb` (optional but recommended for richer NWB probe metadata)

## Configuration

Use a YAML config passed via `--config`.

Included examples:
- `dandi_ingest/configs/dandi_000718.yaml`
- `dandi_ingest/configs/dandi_000458.yaml`

Important sections:
- `dataset`: `dandiset_id`, `version`, `adapter`, `config_id`
- `storage`: output roots for raw files, manifests, and triage artifacts
- `selection`: path/subject/session filters and optional `asset_limit`
- `outputs`: explicit output file paths (or use defaults)

## Commands

From repo root:

```powershell
# List assets, apply filters, and write manifest + triage
python -m dandi_ingest.cli list --config dandi_ingest/configs/dandi_000718.yaml

# Download selected assets (or filtered assets if adapter selection is empty)
python -m dandi_ingest.cli download --config dandi_ingest/configs/dandi_000718.yaml

# Probe local NWB assets and write probe summaries
python -m dandi_ingest.cli probe --config dandi_ingest/configs/dandi_000718.yaml
```

## Outputs

Typical outputs under `storage.output_root`:
- `manifests/<dandiset>/<config_id>_manifest.json`
- `manifests/<dandiset>/<config_id>_manifest.csv`
- `triage/<dandiset>/<config_id>_triage.md`
- `triage/<dandiset>/<config_id>_probe.json`
- `raw/<dandiset>/...` (downloaded NWB assets)

## Integration with MNDM

After download/probe, point your MNDM NWB overlay config at the local raw root.
Examples in `mndm/config/` include:
- `config_ingest_dandi_000718.yaml`
- `config_ingest_common_nwb.yaml`

This keeps DANDI acquisition and triage concerns separated from MNPS feature and summarize processing.
