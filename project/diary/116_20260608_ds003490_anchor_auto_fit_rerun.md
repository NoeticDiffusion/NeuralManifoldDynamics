## Session: ds003490 anchor_auto_fit rerun

### Date
2026-06-08

### Goal
Ensure `ds003490` emits both subject-anchored and cohort-anchored MNDM 2.1 coordinate layers when no external anchor file is provided, then rerun smoke + full dataset.

### Changes made
1. Updated `mndm/config/config_ingest_ds003490.yaml`:
   - Kept:
     - `mnps_projection.export_contracts.subject_anchored: true`
     - `mnps_projection.export_contracts.cohort_anchored: true`
   - Added:
     - `mnps_projection.anchor_auto_fit.enabled: true`
     - `anchor_id: "ds003490_cohort_auto_v2_1"`
     - `anchor_source: "ds003490_all_subjects_features_table"`
     - `scale_method: "iqr"`
     - `min_subjects: 3`

2. Updated docs:
   - `mndm/README.md`
     - Added explicit `anchor_auto_fit` YAML example.
     - Clarified that `export_contracts.cohort_anchored: true` alone does not emit cohort layers unless an anchor source is active.
   - `mndm/CONFIG_GUIDE.md`
     - Added MNDM 2.1 anchored output snippet with `anchor_auto_fit`.
     - Added same clarification about required active anchor source.

### Validation and runs
1. Micro-smoke (`--subject 001`) was intentionally too small for cohort anchor fit:
   - Command: `python -m mndm.cli all --dataset ds003490 --subject 001 --config mndm/config/config_ingest_ds003490.yaml --n-jobs 2`
   - Result: `Feature anchor file produced no usable anchors` (expected with too few subjects for `min_subjects: 3`).

2. Short anchor smoke (passes):
   - Command: `python -m mndm.cli anchor-smoke --dataset ds003490 --data-dir M:/datasets/received/openneuro --h5-root M:/datasets/processed/openneuro --config mndm/config/config_ingest_ds003490.yaml --max-files 30`
   - Result highlights:
     - `h5_files_found: 30`
     - `subjects_with_rows: 16`
     - `anchor_usable_features: 64`
     - Built anchor id: `ds003490_smoke_v2_1`

3. Full rerun (passes):
   - Command: `python -m mndm.cli all --dataset ds003490 --config mndm/config/config_ingest_ds003490.yaml --n-jobs 12`
   - New run dir:
     - `M:/datasets/processed/openneuro/ds003490/neuralmanifolddynamics_ds003490_20260608_121833`
   - Key evidence:
     - one-shot fitted anchor written:
       - `anchors/ds003490_cohort_auto_v2_1.json`
     - `run_manifest.json` reports:
       - `feature_anchors: true`
       - `coords_3d_cohort_anchored: true`
       - `coords_9d_cohort_anchored: true`
       - `h5_with_feature_anchors: 75`
       - `h5_with_coords_3d_cohort_anchored: 75`
       - `h5_with_coords_9d_cohort_anchored: 75`

### Notes
- The `--subject` micro-smoke is not suitable for validating cohort anchor fit when `min_subjects` is enforced.
- For anchor validation, use either:
  - multi-subject summarize/all, or
  - `anchor-smoke` with enough files/subjects.
