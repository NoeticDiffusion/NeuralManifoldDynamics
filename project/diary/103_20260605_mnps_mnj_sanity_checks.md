# 103 - 2026-06-05 - MNPS/MNJ sanity checks

## Research question
Can `mndm` emit explicit reviewer-facing sanity checks for `mnps_3d`, `coords_9d`, and `MNJ` without changing the canonical measurement contract or attempting downstream reachability/persistence computation?

## What was implemented
- Added a new summarize-time QA block `robustness.review_qc.mnps_mnj_sanity`.
- Implemented `compute_mnps_mnj_sanity(...)` in `mndm/src/mndm/pipeline/robustness_helpers.py`.
- Wired the new block into:
  - `summary.json` manifest extras
  - `qc_summary.json`
- Added config surface in:
  - `mndm/config/config_template.yaml`
  - `mndm/config/config_ingest_physionet_i-care_2_1.yaml`

## Sanity surface
The new QA block now reports:
- projection-contract checks for the `mnps_3d` / `coords_9d` layer
- finite-row coverage for `mnps_3d` and `coords_9d`
- per-axis / per-subcoordinate degeneracy flags
- explicit separation of:
  - `mnps_3d` validity
  - `coords_9d` validity
  - `MNJ` validity
- `MNJ` condition-number / `rel_mse_baseline` warning surface
- combined degeneracy flags when broken 9D geometry and unstable MNJ co-occur

## Robustified comparison
Default QA behavior now includes a parallel robustified comparison path:
- winsorized coordinate descriptives
- Jacobian condition-number comparison with a QA-only ridge floor
- raw-vs-robust warning deltas

This path is additive and reviewer-facing only. It does not overwrite canonical exported coordinates or Jacobians.

## Validation
Targeted tests run:
- `python -m pytest mndm/tests/test_robustness.py mndm/tests/test_dataset_subject_runner.py -q`
- `python -m pytest mndm/tests/test_baseline_qc.py -q`

Results:
- `33 passed` in the first targeted batch
- `2 passed` in the second

## Notes
- Scope intentionally stopped at `MNPS 3D/9D` and `MNJ`.
- No downstream `tube_*` or `persistence_*` computation was added here.
- The output is meant to act as an internal gate before higher-level interpretation.
