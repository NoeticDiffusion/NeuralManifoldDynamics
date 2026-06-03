# 057_20260528_ds006036_full_rerun_subject_cohort_new_event_contract

## Research question
- Execute a full ds006036 rerun with the updated event-provenance + stage-blocking configuration, with two summarize passes:
  1) subject-anchor baseline,
  2) cohort-anchor pass with one-shot anchor fitting.

## Run context
- Dataset: `ds006036` (Alzheimer/FTD EEG cohort)
- Config: `mndm/config/config_ingest_ds006036.yaml`
- Full-feature root used: `E:/Science_Datasets/openneuro/processed_ds006036_subject_then_cohort_anchor`
  - Verified this root contains full features (`features.parquet` with 88 files / 6820 rows).

## Commands
1. Subject-anchor summarize:
   - `python -m mndm.cli summarize --dataset ds006036 --config mndm/config/config_ingest_ds006036.yaml --out-dir E:/Science_Datasets/openneuro/processed_ds006036_subject_then_cohort_anchor --n-jobs 6`
2. Cohort-anchor summarize:
   - `python -m mndm.cli summarize --dataset ds006036 --config mndm/config/config_ingest_ds006036.yaml --out-dir E:/Science_Datasets/openneuro/processed_ds006036_subject_then_cohort_anchor --n-jobs 6 --fit-anchor --anchor-id ds006036_cohort_anchor_v2_2`

## Outputs
- Subject-anchor run dir:
  - `E:/Science_Datasets/openneuro/processed_ds006036_subject_then_cohort_anchor/ds006036/neuralmanifolddynamics_ds006036_20260528_084046`
- Cohort-anchor run dir:
  - `E:/Science_Datasets/openneuro/processed_ds006036_subject_then_cohort_anchor/ds006036/neuralmanifolddynamics_ds006036_20260528_084641`
- Cohort anchor artifact:
  - `.../neuralmanifolddynamics_ds006036_20260528_084641/anchors/ds006036_cohort_anchor_v2_2.json`

## Validation summary
- Both runs completed with:
  - `h5 = 88`, `summary_json = 88`
  - `run_status = completed`
  - `run_errors.count = 0`, `groupings_total = 88`
- Subject-anchor capabilities:
  - `feature_anchors = false`
  - `coords_3d_cohort_anchored = false`
  - `coords_9d_cohort_anchored = false`
- Cohort-anchor capabilities:
  - `feature_anchors = true`
  - `coords_3d_cohort_anchored = true`
  - `coords_9d_cohort_anchored = true`
  - `anchor_id = ds006036_cohort_anchor_v2_2`
- Stage/event QC (both runs, run-level aggregate):
  - `subjects_with_raw_25hz = 6`
  - `subjects_with_raw_30hz = 2`
  - `missing_expected_frequencies_hz_raw = []`
  - `stage_mapping_qc.json` written and linked in `run_manifest.json`

## Notes
- A quick summarize run in the default `processed` root was not used for final rerun because that root currently contains only a single-subject feature table for ds006036. The full rerun was therefore executed in the dedicated full-feature root above.
