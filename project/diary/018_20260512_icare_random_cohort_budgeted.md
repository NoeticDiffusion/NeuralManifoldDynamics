# 018 — 2026-05-12 — I-CARE Random Longitudinal Cohort with Budget Guard

## Session goal

Enable cohort sampling for longitudinal coma analysis in I-CARE with a practical
storage cap (300 GB), and produce a reproducible random patient subset suitable
for pilot NDT analyses.

## Implemented

1. Extended PhysioNet ingest subset logic in:
   - `physionet_ingest/script/download_physionet.py`

   New features:
   - `subset.strategy: random_n_patients` in addition to `first_n_patients`.
   - `subset.random_seed` for deterministic random sampling.
   - `subset.max_total_gb`, `subset.enforce_budget`, `subset.min_patient_count`.
   - Global-size-based storage estimation using PhysioNet project metadata
     (`main_storage_size`) and available patient pool size.
   - Budget-aware re-selection path when `enforce_budget: true`.
   - Extended run summary fields:
     - requested vs selected patient count,
     - pool size,
     - random seed,
     - budget config,
     - budget estimation payload.

2. Added a dedicated longitudinal config:
   - `physionet_ingest/config/config_i-care_2_1_longitudinal.yml`
   - Defaults:
     - `random_n_patients`,
     - `patient_count: 40`,
     - `random_seed: 20260512`,
     - `max_total_gb: 300`,
     - `enforce_budget: true`,
     - `min_patient_count: 30`,
     - EEG-only longitudinal files (`*_EEG.hea`, `*_EEG.mat`) + patient `.txt`.

3. Updated docs:
   - `physionet_ingest/README.md` with random strategy and budget options.

4. Expanded test coverage:
   - `physionet_ingest/tests/test_download_physionet.py`
   - Added deterministic random-sampling test and budget-estimation test.

## Validation

- Tests:
  - `python -m pytest physionet_ingest/tests/test_download_physionet.py`
  - Result: 9 passed.

- Dry-run with `.venv` and longitudinal config:
  - `python -m physionet_ingest.script.download_physionet --config-dataset physionet_ingest/config/config_i-care_2_1_longitudinal.yml --dry-run`
  - Result:
    - selected patients: 40 (from pool of 607),
    - planned files: 4,885 (4,882 in `training/*` + 3 top-level),
    - estimated size: ~103.96 GB (under 300 GB cap),
    - errors: 0.

- Dry-run sensitivity at 50 patients (runtime config override):
  - Result:
    - selected patients: 50,
    - planned files: 6,357,
    - estimated size: ~129.95 GB (still under 300 GB cap).

## Cohort artifact paths

- `E:/Science_Datasets/physionet/metadata/i-care_2_1_random40_longitudinal/selected_patients.txt`
- `E:/Science_Datasets/physionet/metadata/i-care_2_1_random40_longitudinal/planned_files.txt`
- `E:/Science_Datasets/physionet/metadata/i-care_2_1_random40_longitudinal/run_summary.json`

## Notes and limitations

- Checksums endpoint via API returned HTTP 401 without credentials, and the
  pipeline correctly fell back to direct public file download.
- Budget estimate is based on global dataset size and patient count; actual
  subset size can vary by patient recording duration and segment density.
