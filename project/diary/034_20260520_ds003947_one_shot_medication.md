# 034 2026-05-20 ds003947 one-shot medication

## Research question

Can the new one-shot cohort-anchor workflow run cleanly on `ds003947`, and can we carry useful phenotype metadata, starting with `phenotype/medication.tsv`, into the per-subject MNDM outputs?

## What changed

- Added support for merging extra tabular participant metadata into the base participant table via `metadata_extraction.datasets.<id>.participants.extra_tables`.
- Configured `ds003947` to merge `phenotype/medication.tsv` and retain `CPZ_at_scan`.
- Added a focused metadata extraction test covering participant-table plus phenotype-table merging.

## Validation

- `python -m pytest tests/test_metadata_extraction.py`
  - Passed (`30 passed`).
- `python -m mndm.cli prerequisite-check --dataset ds003947 --data-dir "M:\datasets\received\openneuro" --config "config/config_ingest_ds003947.yaml"`
  - Passed.
- `python -m mndm.cli all --dataset ds003947 --data-dir "M:\datasets\received\openneuro" --config "config/config_ingest_ds003947.yaml" --fit-anchor --n-jobs 4`
  - Completed successfully.

## Run outputs

- Run directory:
  - `E:\Science_Datasets\openneuro\processed\ds003947\neuralmanifolddynamics_ds003947_20260520_114135`
- One-shot anchor:
  - `E:\Science_Datasets\openneuro\processed\ds003947\neuralmanifolddynamics_ds003947_20260520_114135\anchors\ds003947_all_subjects_iqr_v2_1.json`
- Manifest confirms:
  - `61` subject H5 files
  - `fit_anchor: true`
  - both `Control` and `FEP` groups present
  - `coords_3d_*` and `coords_9d_*` subject/cohort anchored layers present in all H5 files
  - embedded `/feature_anchors` present in all H5 files

## Phenotype result

- `CPZ_at_scan` was merged into participant metadata for `ds003947`.
- Quick real-data check before summarize showed `31 / 61` rows with non-null `CPZ_at_scan`.
- Sample verification from finished H5s:
  - `sub-2237A_rest_rest.h5` contains `participant_meta.CPZ_at_scan = 311.5264798`
  - `sub-2235A_rest_rest.h5` contains `participant_meta.CPZ_at_scan = NaN` as expected for a control

## Notes

- The `all` command reused already computed features from the processed store (`Found 61 already processed files in features.parquet`), then performed a fresh one-shot anchored summarize pass.
- Regional MNPS remained unavailable for this dataset under the current channel grouping despite `regional_mnps` being enabled, but the main 3D/9D MNPS and Jacobian outputs were all produced.

## Status

- Internal validated result:
  - one-shot cohort anchoring works on `ds003947`
  - medication phenotype merge works and survives into per-subject H5 metadata
- Plausible next step:
  - add one or two more phenotype tables (`bprs.tsv`, `saps.tsv`, or `sans.tsv`) if we want symptom burden available alongside medication in downstream analysis
