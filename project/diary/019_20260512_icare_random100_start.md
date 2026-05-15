# 019 — 2026-05-12 — I-CARE Random100 Download Start

## Session goal

Launch a larger longitudinal I-CARE cohort download with 100 randomly sampled
patients to improve coverage for NDT/DoC analysis under a 300 GB storage cap.

## Configuration updates

- Updated:
  - `physionet_ingest/config/config_i-care_2_1_longitudinal.yml`
- Changes:
  - `subset.patient_count: 100`
  - `download.output_subdir: i-care_2_1_random100_longitudinal`
  - retained:
    - `subset.strategy: random_n_patients`
    - `subset.random_seed: 20260512`
    - `subset.max_total_gb: 300`
    - `subset.enforce_budget: true`
    - `subset.min_patient_count: 30`

## Validation before launch

- Dry-run command:
  - `python -m physionet_ingest.script.download_physionet --config-dataset physionet_ingest/config/config_i-care_2_1_longitudinal.yml --dry-run`
- Result:
  - selected patients: 100
  - planned files: 12,519
  - estimated subset size: ~259.91 GB (below 300 GB cap)

## Download launch

- Started real run with `.venv`:
  - `python -m physionet_ingest.script.download_physionet --config-dataset physionet_ingest/config/config_i-care_2_1_longitudinal.yml --no-dry-run`
- Initial health checks:
  - process started successfully
  - selected 100 patients and planned 12,519 files
  - output bytes started increasing in target folder

## Artifacts

- `E:/Science_Datasets/physionet/metadata/i-care_2_1_random100_longitudinal/selected_patients.txt` (100 IDs)
- `E:/Science_Datasets/physionet/metadata/i-care_2_1_random100_longitudinal/planned_files.txt` (12,519 entries)
