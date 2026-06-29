# i-care coma pack v1 (EEG-only proxies)

## Research question

Can we add reviewer-expected coma prognostic markers into the i-care pipeline as sidecar-compatible outputs, while clearly separating what is EEG-derivable from what requires external clinical data?

## What changed

- Added a new `coma` comparator pack to the conventional EEG layer in:
  - `mndm/src/mndm/features/eeg.py`
  - `mndm/src/mndm/pipeline/conventional_summary.py`
- Implemented EEG-only coma proxy features:
  - `eeg_conventional_coma_suppression_ratio`
  - `eeg_conventional_coma_burst_suppression_proxy`
  - `eeg_conventional_coma_continuity_proxy`
  - `eeg_conventional_coma_alpha_delta_ratio`
  - `eeg_conventional_coma_reactivity_proxy`
- Added explicit clinical-marker availability metadata in summarize output when `coma` pack is enabled:
  - `SSEP`, `NSE`, `GCS`, `S100B` exported as `status: unavailable`
  - marked as EEG-only proxy mode until external clinical source is ingested
- Enabled the new pack in the i-care no-regional profile:
  - `mndm/config/config_ingest_physionet_i-care_2_1_part1_12_24h_no_regional.yaml`
- Extended template/config documentation surfaces:
  - `mndm/config/config_template.yaml`
  - `mndm/README.md`
  - `mndm/Output_variables_guide.md`
  - `README.md`

## Validation

- Added new feature tests in `mndm/tests/test_features_eeg.py` for:
  - coma output columns
  - continuity vs suppression consistency
  - reactivity proxy emission
  - reactivity proxy disable behavior
- Added summarize export test in `mndm/tests/test_dataset_subject_runner.py` for:
  - `families.coma` export
  - clinical marker unavailable metadata in payload/manifest
- Focused test runs passed:
  - `python -m pytest mndm/tests/test_features_eeg.py -k "conventional_coma" -q`
  - `python -m pytest mndm/tests/test_dataset_subject_runner.py -k "conventional_eeg_coma_summary" -q`
  - `python -m pytest mndm/tests/test_features_eeg.py -k "conventional" -q`
  - `python -m pytest mndm/tests/test_dataset_subject_runner.py -k "conventional_eeg" -q`

## Evidence category

- **Internal validated result**:
  - new coma pack computation path is implemented and test-covered
  - summarize/manifest export path includes coma family and explicit unavailable clinical markers
- **Plausible interpretation**:
  - EEG reactivity proxy is a useful trend feature for no-annotation datasets, but not a substitute for stimulation-annotated clinical reactivity scoring

## Remaining caveats

- `SSEP`, `NSE`, `GCS`, and `S100B` are still unavailable without external clinical sidecars.
- `eeg_conventional_coma_reactivity_proxy` is a deterministic EEG-only proxy and should be treated as a comparator feature, not as definitive bedside reactivity.
