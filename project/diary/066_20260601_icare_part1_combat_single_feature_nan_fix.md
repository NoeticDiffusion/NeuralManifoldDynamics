# 066 — 2026-06-01 — i-care part1 ComBat 9D e_m NaN root-cause + fix

## Research question
Why does the part1 ComBat run (`neuralmanifolddynamics_physionet_icare_2_1_20260527_101007_NeruroComBat_cleaned`) produce unusable 9D coordinates with `coords_9d_row_valid_fraction = 0.0`?

## Findings
- Full-run audit over all 3104 H5 files confirmed:
  - every file has `coords_9d_row_valid_fraction = 0.0`
  - only one subcoordinate is non-finite: `e_m`
  - all other subcoordinates (`m_a..e_s`) are finite.
- `features_snapshot.json` in the same run shows:
  - `embodied_arousal_proxy.missing_rate = 1.0` post-normalization
  - this is the only feature in that state, and it maps directly to `e_m` in MNPS 9D.
- Root-cause reproduction on real part1 features (QC-filtered rows = 2,846,892):
  - running `neuroCombat` on the single feature `embodied_arousal_proxy` (family-wise `__other__`, one-column chunk) returns all NaN.
  - this reproduces the exact failure mode seen in the ComBat run.

## Root cause (internal validated)
ComBat is applied chunk-wise per family. For a single-feature chunk, `neuroCombat` can return all-NaN output.  
In this run, `embodied_arousal_proxy` was isolated as a one-feature family and got overwritten by NaNs, which then forced `e_m` (and thus full-row 9D validity) to zero.

## Code fix
File: `mndm/src/mndm/pipeline/summary.py`

Change:
- in `_apply_feature_normalization`, detect `len(prepared_rows) == 1`
- skip ComBat for that chunk and keep the original feature unchanged
- record skip as `skipped_columns.single_feature_family`
- emit warning with chunk/family/feature context.

This prevents single-feature families from being NaN-harmonized while preserving multi-feature family harmonization.

## Regression test
File: `mndm/tests/test_dataset_subject_runner.py`

Added:
- `test_dataset_runner_combat_preserves_single_feature_family`
  - verifies single-feature family values remain finite and unchanged
  - verifies skip accounting is recorded.

Also updated existing combat test fixture to include two EEG features so at least one family remains harmonized under the new guard.

## Validation
Executed:
- `pytest mndm/tests/test_dataset_subject_runner.py -k combat`

Result:
- 2 selected combat tests passed.

## Practical implication
Existing part1 ComBat outputs remain affected (already written with NaN `embodied_arousal_proxy`).  
To recover usable 9D (`e_m` finite), rerun summarize/analysis with this fix applied.
