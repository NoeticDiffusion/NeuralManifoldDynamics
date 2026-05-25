# 031 - 2026-05-18 - Balanced next-140 CPC1/CPC5 cohort

## Session goal

Create a new non-overlapping validation cohort of 140 patients (70 CPC1 + 70 CPC5), balance baseline metadata between groups as closely as possible, and start 0-12h EEG download into a new destination.

## What was added

- New planner script:
  - `physionet_ingest/script/plan_icare_balanced_cpc_cohort.py`
  - Purpose: select an explicit CPC1/CPC5 cohort with metadata-balance optimization.
  - Candidate pool excludes locally downloaded patients from:
    - `E:/Science_Datasets/physionet/received/i-care_2_1_random100_longitudinal/training`
  - Uses random restarts + local swap hill-climb with a balance objective over:
    - age
    - sex
    - hospital/site
    - OHCA
    - shockable rhythm
    - TTM class
    - ROSC availability/value

- Generated downloader config:
  - `physionet_ingest/config/config_i-care_2_1_next_140_longitudinal_0_12h.yml`
  - Strategy: `explicit_patient_ids`
  - Target IDs: 140 total (70 CPC1 + 70 CPC5)
  - 0-12h cap: `subset.max_eeg_hours_per_patient: 12`
  - Local verify mode:
    - `download.verify_checksum: false`
    - `download.verify_existing_size: true`
  - Output destination subdir:
    - `i-care_2_1_next_140_longitudinal_0_12h`

- Generated plan report:
  - `E:/Science_Datasets/physionet/metadata/i-care_2_1_next_140_longitudinal_0_12h/balanced_cpc15_plan.json`

## Selection outcome

- Pool overview:
  - Total pool: 607
  - Excluded local overlap: 162
  - Non-overlap candidates: 445
  - CPC1 candidates: 110
  - CPC5 candidates: 282

- Selected:
  - CPC1: 70
  - CPC5: 70

- Balance summary from report:
  - Sex counts matched exactly: 49 male / 21 female in each group
  - Site counts matched exactly: A:38, B:9, D:7, E:7, F:9 in each group
  - OHCA counts matched exactly: true 53 / false 13 / missing 4 in each group
  - TTM counts matched exactly: 33:50 / 36:8 / none:12 in each group
  - Shockable near-matched: CPC1 true 46 vs CPC5 true 45
  - Age means close: 60.64 (CPC1) vs 61.04 (CPC5)
  - ROSC medians close: 16 vs 15 min (known n=27 in each group)

## Download status

- Started downloader with new config:
  - `python -m physionet_ingest.script.download_physionet --config-dataset physionet_ingest/config/config_i-care_2_1_next_140_longitudinal_0_12h.yml --no-dry-run`
- Destination:
  - `E:/Science_Datasets/physionet/received/i-care_2_1_next_140_longitudinal_0_12h`
- Startup logs confirm explicit 140-patient selection and size-based local verification mode.
