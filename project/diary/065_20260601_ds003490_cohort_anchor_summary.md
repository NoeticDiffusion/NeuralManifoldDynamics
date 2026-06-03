# ds003490 cohort anchor summary

Date: 2026-06-01

## Question

Run a cohort-anchored summarize pass for `ds003490` using the same dataset config
and worker settings as the existing subject-anchored run, so both versions are
available for article work.

## Command

```powershell
python -m mndm.cli summarize --dataset ds003490 --config mndm/config/config_ingest_ds003490.yaml --n-jobs 4 --fit-anchor --anchor-id ds003490_parkinson_cohort_v2_1
```

## Output

Run directory:

- `M:/datasets/processed/openneuro/ds003490/neuralmanifolddynamics_ds003490_20260601_122351`

Anchor artifact:

- `M:/datasets/processed/openneuro/ds003490/neuralmanifolddynamics_ds003490_20260601_122351/anchors/ds003490_parkinson_cohort_v2_1.json`

## Result

The run completed successfully with `exit_code: 0`.

Key counts from `run_manifest.json`:

- `h5: 75`
- `summary_json: 75`
- `qc_summary_json: 75`
- `qc_reliability_json: 75`
- `regional_csv: 2`
- `block_csv: 2`

## Cohort-anchor validation

Confirmed from `run_manifest.json`:

- `feature_anchors: true`
- `coords_3d_subject_anchored: true`
- `coords_3d_cohort_anchored: true`
- `coords_9d_subject_anchored: true`
- `coords_9d_cohort_anchored: true`
- `fit_anchor: true`

Also confirmed per-file coverage:

- `h5_with_feature_anchors: 75`
- `h5_with_coords_3d_cohort_anchored: 75`
- `h5_with_coords_9d_cohort_anchored: 75`

This means the cohort-anchor layer was not just fitted, but also propagated into
all summarized H5 outputs.

## Runtime notes

- The run emitted the same style of `9D MNPS falsified` warnings already seen in
  the subject-anchored pass.
- These warnings did not prevent successful completion.
- `stage_mapping_qc.json`, regional MNPS CSV, and block Jacobian CSV outputs were
  also written in the cohort-anchor run directory.

## Evidence category

- Internal validated result:
  - `ds003490` now has both a subject-anchored summarize run and a
    cohort-anchored summarize run
  - the cohort-anchored run includes embedded frozen feature anchors and cohort
    coordinate layers in all 75 H5 outputs
