# 054_20260528_ds006036_subject_then_cohort_anchor_runs

## Research question
- Run `ds006036` with the requested order:
  1) standalone `features`,
  2) `summarize` with subject-anchored primary coordinates,
  3) `summarize` with one-shot cohort anchor.

## Run setup
- Config: `mndm/config/config_ingest_ds006036.yaml`
- Output root (dedicated): `E:/Science_Datasets/openneuro/processed_ds006036_subject_then_cohort_anchor`
- Runtime: `.venv` Python with `PYTHONPATH` pointing to `mndm/src` + `core/src`

## Executed sequence
1. Features:
   - `mndm.cli features --dataset ds006036 ... --out-dir processed_ds006036_subject_then_cohort_anchor --n-jobs 6`
2. Summarize round 1 (subject-anchor baseline):
   - `mndm.cli summarize --dataset ds006036 ... --out-dir processed_ds006036_subject_then_cohort_anchor --n-jobs 6`
3. Summarize round 2 (cohort-anchor):
   - `mndm.cli summarize --dataset ds006036 ... --out-dir processed_ds006036_subject_then_cohort_anchor --n-jobs 6 --fit-anchor --anchor-id ds006036_cohort_anchor_v2_1`

## Results
- Features completed:
  - `features.csv` + `features.parquet` written
  - `6820` epochs total
- Summarize round 1 completed:
  - Run dir: `neuralmanifolddynamics_ds006036_20260528_055905`
  - `88` H5 files
  - `run_errors_count = 0`
  - Capability check: `feature_anchors=false`, `coords_3d_cohort_anchored=false`
- Summarize round 2 completed:
  - Run dir: `neuralmanifolddynamics_ds006036_20260528_060416`
  - `88` H5 files
  - `run_errors_count = 0`
  - Anchor created: `anchors/ds006036_cohort_anchor_v2_1.json`
  - Capability check: `feature_anchors=true`, `coords_3d_cohort_anchored=true`, `coords_9d_cohort_anchored=true`

## Notes
- Both runs preserve subject-anchored layers; round 2 additionally exposes cohort-anchored layers and anchor provenance.
- Existing 9D conditioning warnings can still appear in a subset of sessions and are captured in standard outputs/provenance.

