# ds003645 MEG shadow mapping

## Question
Can ds003645 MEG be integrated into the existing EEG-centered NMD contract without creating a parallel geometry stack?

## What was implemented
- Added `mndm/src/mndm/features/meg.py` with a minimal MEG shadow surface:
  - diagnostic MAG/GRAD columns such as `meg_mag_alpha`, `meg_grad_alpha`
  - combined shadow columns such as `meg_delta`, `meg_alpha`, `meg_beta_alpha`, `meg_hjorth_mobility`, `meg_permutation_entropy`, `meg_highfreq_power_30_45`
- Registered MEG feature extraction in `mndm/src/mndm/parallel.py`.
- Extended `mndm/src/mndm/preprocess.py` so FIF preprocessing emits:
  - `meg`
  - `meg_mag`
  - `meg_grad`
- Added ds003645 config policy in `mndm/config/config_ingest_ds003645.yaml`:
  - `modality: meg`
  - ds003645-specific `mnps_9d` subcoords using `meg_*`
  - `mnps_projection.feature_standardization` entries for `meg_*`
  - explicit `meg_mapping` provenance block
  - five-subject pilot validation gates for `sub-002` to `sub-006`
- Added export/manifest provenance surfacing in:
  - `mndm/src/mndm/pipeline/summary.py`
  - `mndm/src/mndm/pipeline/run_manifest.py`
- Updated `mndm/Output_variables_guide.md` for MEG shadow features and `/provenance/mapping/*`.

## Validation completed
- Targeted tests passed:
  - MEG feature extraction
  - preprocess MAG/GRAD exposure
  - embodied proxy fallback priority
  - run-manifest config excerpt
  - subject-level provenance export
- Command run:
  - `pytest mndm/tests/test_features_meg.py mndm/tests/test_preprocess_modalities.py mndm/tests/test_parallel.py mndm/tests/test_run_manifest.py mndm/tests/test_dataset_subject_runner.py -q`
- Result:
  - `9 passed, 5 skipped`

## Evidence category
- Internal validated result:
  - ds003645 now has a config-defined MEG shadow mapping path into the existing 9D/3D export contract.
- Plausible interpretation:
  - MAG/GRAD robust-z median combination is a reasonable v0.1 sensor-space bridge into the EEG contract.
- Not established:
  - physiological equivalence between EEG and MEG
  - source-space comparability
  - task-level geometry agreement beyond the configured validation gates

## Remaining limits
- This is a sensor-space shadow mapping only.
- The five-subject validation gates are defined in config/provenance but not yet executed on full downloaded ds003645 content.
- The canonical export contract is preserved; this work does not introduce a separate MEG-only H5 schema.
