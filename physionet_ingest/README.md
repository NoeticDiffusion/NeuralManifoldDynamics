# PhysioNet Ingest

Config-driven ingestion for PhysioNet datasets, starting with I-CARE v2.1.

This pipeline uses:
- the `physionet` Python client for project metadata and checksum endpoint access,
- direct HTTP streaming (`/files/...`) for file downloads,
- subset selection from the dataset `RECORDS` manifest (first N or random N).

## Configuration

General settings:
- `physionet_ingest/config/config_ingest.yml`

Dataset-specific settings (I-CARE v2.1):
- `physionet_ingest/config/config_i-care_2_1.yml`
- `physionet_ingest/config/config_i-care_2_1_longitudinal.yml` (100 random patients, 300 GB cap)

Key knobs:
- `subset.strategy`: supports `first_n_patients` and `random_n_patients`
- `subset.patient_count`: number of patients to include
- `subset.random_seed`: deterministic sampling for random strategy
- `subset.max_total_gb`: optional cap with global-size-based estimation
- `subset.enforce_budget`: reduce selected cohort size if estimated size exceeds cap
- `subset.min_patient_count`: lower bound when budget enforcement is active
- `subset.max_files_per_patient`: optional cap per selected patient directory
- `subset.include_file_globs`: glob filters for files discovered in checksums
- `subset.expand_record_extensions`: extensions appended to extension-less `RECORDS` entries
- `download.dry_run`: plan only, no bytes downloaded
- `download.verify_checksum`: verify SHA256 when checksums are available
- `download.max_parallel_downloads`: number of files to download concurrently

## Credentials (optional)

For authenticated PhysioNet API endpoints, set:

```powershell
$env:PHYSIONET_USERNAME="your_username"
$env:PHYSIONET_PASSWORD="your_password"
```

Open datasets can still be downloaded without credentials via direct file URLs.

## Run

From repo root:

```bash
python -m physionet_ingest.script.download_physionet --config-general physionet_ingest/config/config_ingest.yml --config-dataset physionet_ingest/config/config_i-care_2_1.yml
```

Force dry run:

```bash
python -m physionet_ingest.script.download_physionet --dry-run
```

Force real download (override YAML):

```bash
python -m physionet_ingest.script.download_physionet --no-dry-run
```

## Outputs

Under `paths.metadata_root/<output_subdir>/`:
- `selected_patients.txt`
- `planned_files.txt`
- `download_manifest.csv`
- `download_manifest.jsonl`
- `run_summary.json`

Under `paths.download_root/<output_subdir>/`:
- downloaded dataset files, preserving remote folder structure.
