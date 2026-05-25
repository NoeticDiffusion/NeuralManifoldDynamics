# 032 2026-05-20 - MNDM 2.1 Anchored Coordinates

## Research Question

Can MNDM preserve the current subject/session-relative geometry while adding a reviewer-auditable cohort-anchored coordinate contract for clinical group comparisons?

## Implemented

- Added `mndm.anchors`, a MNDM 2.1 feature-anchor module that fits subject-balanced anchors from exported `/features_raw` H5 surfaces.
- Added CLI commands:
  - `mndm anchors-fit`
  - `mndm anchor-smoke`
  - `mndm anchor-sensitivity`
- Added an `external_anchor` path through `projection.project_features`, `project_features_with_coverage`, and `project_features_v2`.
- Added MNDM 2.1 coordinate-layer support to `MNPSPayload` and the H5 writer:
  - `/coords_3d_subject_anchored`
  - `/coords_9d_subject_anchored`
  - `/coords_3d_cohort_anchored`
  - `/coords_9d_cohort_anchored`
  - `/feature_anchors`
- Wired summarize so configured anchors can make cohort-anchored coordinates the primary 2.1 coordinate contract while preserving subject-anchored layers.
- Updated run-manifest capability probing and field guide entries for MNDM 2.1 coordinate layers.
- Updated README/schema/command documentation and structure-check settings for MNDM 2.1 anchored coordinate outputs.
- Bumped `mndm.__version__` to `2.1.0`.

## Smoke-Test Targets

The first empirical targets are the three OpenNeuro EEG cohorts that have been hard to separate with the current subject-relative geometry:

- `ds003478`: depression / BDI
- `ds003944`: psychosis / FEP
- `ds004504`: AD / FTD / Healthy

The new `anchor-smoke` command is intended to run post-hoc on summarized H5 outputs while checking whether the raw dataset folders exist under `M:/datasets/received/openneuro`.

## Validation

Focused tests passed:

```text
python -m pytest mndm/tests/test_anchors.py mndm/tests/test_mnps_projection.py mndm/tests/test_schema.py mndm/tests/test_writers.py mndm/tests/test_run_manifest.py
42 passed
```

CLI help and syntax checks also passed for the new anchor commands and modified pipeline modules.

Raw-folder presence smoke check passed for all three target datasets under
`M:/datasets/received/openneuro`; no summarized H5 root was supplied in that
check, so no separation statistics were computed yet.

The structure checker now always validates the MNDM 2.1 subject-anchored 3D
coordinate layer and can enforce embedded anchor/cohort-coordinate outputs with
`common.require_anchor_outputs=true`.

## Claim Discipline

- Established: per-subject/session affine scaling can remove absolute group-level differences by construction.
- Internal validated: the code now supports external feature anchors and exports explicit subject/cohort coordinate layers in H5.
- Plausible next test: cohort anchoring may improve separation in `ds003478`, `ds003944`, and `ds004504`, where current subject-relative geometry has been weak.
- Not yet validated: improved clinical separation on those datasets. That requires running the new smoke commands on actual summarized H5 outputs.
- Deferred: atlas anchors, covariate-residualized anchors, and full dual-Jacobian export.
