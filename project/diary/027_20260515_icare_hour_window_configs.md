# 027 - 2026-05-15 - I-CARE EEG hour-window download support

## Session goal

Support faster I-CARE ingestion by selecting only patient-relative EEG time windows during download planning (not only first-N hours), and add a ready-to-use 12-24h config.

## Changes made

- Extended `physionet_ingest/script/download_physionet.py`:
  - Added lower-bound support with `subset.min_eeg_hours_per_patient`.
  - Kept `subset.max_eeg_hours_per_patient` as upper bound.
  - Updated WFDB hour filtering logic to keep runs that overlap a configurable patient-relative window `[min_h, max_h]`.
  - Preserved backward compatibility:
    - if only `max_eeg_hours_per_patient` is set, behavior remains "earliest 0-max hours".
  - Added validation:
    - `min_eeg_hours_per_patient >= 0`
    - `max_eeg_hours_per_patient > 0`
    - `min < max`
    - `min` requires `max`
  - Added filter provenance fields in run summary (`subset.min_eeg_hours_per_patient`, `subset.max_eeg_hours_per_patient`, `subset.eeg_hours_filter`).

- Added new config:
  - `physionet_ingest/config/config_i-care_2_1_longitudinal_12_24h.yml`
  - Uses `min_eeg_hours_per_patient: 12`, `max_eeg_hours_per_patient: 24`
  - Separate output subdir: `i-care_2_1_random100_longitudinal_12_24h`

- Updated docs:
  - `physionet_ingest/README.md` now documents min/max EEG hour-window knobs and includes the new 12-24h config in the config list.

## Validation

- Unit tests: `physionet_ingest/tests/test_download_physionet.py`
  - All passing (`15 passed`).
  - Added tests for:
    - WFDB duration parsing
    - 12-24h overlap window selection
    - config validation when `min_eeg_hours_per_patient` is misconfigured

## Expected operational impact

- Enables dataset-size and runtime reduction by excluding non-target patient-relative EEG windows at the ingest stage.
- Provides separate, reproducible download profiles for:
  - first 0-12h (`config_i-care_2_1_longitudinal.yml`)
  - 12-24h (`config_i-care_2_1_longitudinal_12_24h.yml`)
