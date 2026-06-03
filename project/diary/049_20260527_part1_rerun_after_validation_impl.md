# Session Diary 049 - Part1 rerun start after validation/family-wise ComBat rollout

Date: 2026-05-27

## Goal

- Ensure `neuroCombat` is available in active Python and `.venv`.
- Confirm dependency is present in `requirements.txt`.
- Start I-CARE part1 summarize rerun using the correct regional+ComBat YAML.

## Actions

1. Verified dependency listing:
   - `requirements.txt` already contains `neuroCombat==0.2.12`.
2. Verified interpreter state:
   - Active interpreter (`C:\Program Files\Python312\python.exe`) initially missing `neuroCombat`.
   - `.venv` interpreter already had `neuroCombat`.
3. Installed in active interpreter:
   - `python -m pip install neuroCombat==0.2.12`.
4. Re-verified module import:
   - active interpreter: installed,
   - `.venv` interpreter: installed.
5. Confirmed part1 features table exists:
   - `E:/Science_Datasets/physionet/processed/physionet_icare_2_1/features.parquet`.
6. Started summarize rerun (part1) with correct config:
   - command:
     - `python -m mndm.cli summarize --dataset physionet_icare_2_1 --config "config/config_ingest_physionet_i-care_2_1_part1_0_12h_regional.yaml" --n-jobs 16`
   - working directory: `mndm/`
   - early runtime log indicates normal startup and QC filtering.

## Notes

- This rerun uses the config that includes:
  - regional MNPS enabled,
  - MNPS 9D enabled,
  - ComBat enabled,
  - `normalization.validation` enabled,
  - `combat.family_wise` enabled.
