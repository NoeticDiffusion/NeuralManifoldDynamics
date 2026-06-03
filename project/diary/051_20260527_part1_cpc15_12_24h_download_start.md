# Session Diary 051 - Part1 CPC1/CPC5 12-24h download start

Date: 2026-05-27

## Goal

Start a follow-up PhysioNet I-CARE download for part1-style cohort constraints:

- only CPC 1 and CPC 5,
- approximately 70 per class,
- EEG time window overlapping patient-relative 12-24h.

## Cohort grounding

Read local metadata from:

- `E:/Science_Datasets/physionet/received/i-care_2_1_random100_longitudinal/training`

Observed counts in local part1 cohort:

- total local patients with parsed metadata: 162
- CPC counts: `{1: 71, 2: 10, 3: 7, 4: 3, 5: 71}`
- selected IDs for this follow-up: all CPC1/CPC5 only (`71 + 71 = 142`)

## Config added

Created:

- `physionet_ingest/config/config_i-care_2_1_part1_cpc15_longitudinal_12_24h.yml`

Key settings:

- `subset.strategy: explicit_patient_ids`
- `subset.patient_ids`: 142 IDs (CPC1/CPC5 only from local part1 cohort)
- `subset.min_eeg_hours_per_patient: 12`
- `subset.max_eeg_hours_per_patient: 24`
- `download.output_subdir: i-care_2_1_part1_cpc15_longitudinal_12_24h`
- `download.verify_checksum: false`
- `download.verify_existing_size: true`

## Validation before launch

- YAML parse OK.
- Config patient count: 142.
- CPC verification against local metadata:
  - `cpc_counts {1: 71, 5: 71}`
  - `missing_in_local: 0`

## Download launch

Started:

- `python -m physionet_ingest.script.download_physionet --config-dataset "physionet_ingest/config/config_i-care_2_1_part1_cpc15_longitudinal_12_24h.yml" --no-dry-run`

Early log:

- `INFO: Subset strategy: explicit_patient_ids, patient_count=142`
- `INFO: Explicit patient IDs requested: 142`
- `INFO: Existing-file verification mode: size (checksum=False, size_check=True)`
- `WARNING: physionet package unavailable, continuing without API client: No module named 'physionet'`

Process remains running in background.
