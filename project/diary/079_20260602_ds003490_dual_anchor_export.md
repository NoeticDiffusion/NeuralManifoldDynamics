# 079 20260602 ds003490 dual anchor export

## Goal

Close the remaining `ds003490` export gap by ensuring a single summarize run emits both `subject_anchored` and `cohort_anchored` representations in the same H5 bundle, including explicit Jacobian layers and regional network outputs.

## What changed

- Added explicit additive Jacobian-layer support to the H5 payload/writer:
  - `jacobian_subject_anchored`
  - `jacobian_cohort_anchored`
  - `jacobian_9D_subject_anchored`
  - `jacobian_9D_cohort_anchored`
- Kept legacy `/jacobian` and `/jacobian_9D` paths as the primary-contract surfaces for backward compatibility.
- Extended regional export so `/regional_mnps/<network>/...` keeps legacy primary datasets while also embedding:
  - `/regional_mnps/<network>/subject_anchored/*`
  - `/regional_mnps/<network>/cohort_anchored/*`
- Wired regional trajectory building to accept an external anchor so subject and cohort regional layers are both computed from the same summarize pass.
- Added manifest/summary-sidecar pointers for the new Jacobian-layer paths and regional dual-anchor availability.

## Code touchpoints

- `mndm/src/mndm/schema.py`
  - Added `jacobian_layers` to `MNPSPayload`.
- `core/src/core/io/h5_writer.py`
  - Added writer support for explicit Jacobian layers.
  - Added nested regional contract writing while preserving legacy top-level regional paths.
- `mndm/src/mndm/pipeline/summary_regional.py`
  - Allowed regional projections to use an optional external cohort anchor.
- `mndm/src/mndm/pipeline/summary.py`
  - Built both subject/cohort Jacobian layers.
  - Ran regional summarization for both anchor contracts.
  - Exported dual-contract regional payloads and sidecar metadata.

## Tests

Targeted regressions passed:

- `python -m pytest "h:/SourceRepo2/NeuralManifoldDynamics/mndm/tests/test_writers.py" -k "write_h5"`
- `python -m pytest "h:/SourceRepo2/NeuralManifoldDynamics/mndm/tests/test_fmri_summarize.py" -k "one_shot_fit_anchor"`
- `python -m pytest "h:/SourceRepo2/NeuralManifoldDynamics/mndm/tests/test_dataset_subject_runner.py" -k "dual_anchor_jacobian_and_regional_layers"`

## ds003490 rerun

Command:

```powershell
$env:PYTHONPATH="H:/SourceRepo2/NeuralManifoldDynamics/mndm/src;H:/SourceRepo2/NeuralManifoldDynamics/core/src;H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/apollo_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/vitaldb_ingest/src"
python -m mndm.cli summarize --dataset ds003490 --config "mndm/config/config_ingest_ds003490.yaml" --out-dir "M:/datasets/processed/openneuro_parkinson_qeeg_refresh" --n-jobs 12 --fit-anchor
```

Fresh run:

- `M:/datasets/processed/openneuro_parkinson_qeeg_refresh/ds003490/neuralmanifolddynamics_ds003490_20260602_144145`

Runtime:

- exit code `0`
- elapsed `707239 ms`

## Verification

Programmatic inspection over the full run found:

- `75` H5 files
- `0` files missing any of:
  - `coords_3d_subject_anchored`
  - `coords_3d_cohort_anchored`
  - `jacobian_subject_anchored`
  - `jacobian_cohort_anchored`

Sample file verified:

- `sub-001_OFF_rest/sub-001_OFF_rest.h5`

Observed in the sample:

- primary contract: `cohort_anchored`
- both `coords_9d_subject_anchored` and `coords_9d_cohort_anchored`
- both `jacobian_9D_subject_anchored` and `jacobian_9D_cohort_anchored`
- regional network groups (`central`, `frontal`, `parietal_occipital`, `temporal`)
- per-network `subject_anchored` and `cohort_anchored` subgroups inside `/regional_mnps`

Sample shapes:

- `jacobian_subject_anchored/J_hat`: `[133, 3, 3]`
- `jacobian_cohort_anchored/J_hat`: `[133, 3, 3]`
- `jacobian_9D_subject_anchored/J_hat`: `[133, 9, 9]`
- `jacobian_9D_cohort_anchored/J_hat`: `[133, 9, 9]`
- `/regional_mnps/central/subject_anchored/mnps`: `[135, 3]`
- `/regional_mnps/central/cohort_anchored/mnps`: `[135, 3]`

Sidecars also reflect the new surfaces:

- `summary.json` now includes `jacobian_h5.layer_paths`
- `summary.json` now includes `regional_outputs_h5.available_coordinate_contracts = ["cohort_anchored", "subject_anchored"]`
- `run_manifest.json` confirms:
  - `feature_anchors: true`
  - `coords_3d_subject_anchored: true`
  - `coords_3d_cohort_anchored: true`
  - `coords_9d_subject_anchored: true`
  - `coords_9d_cohort_anchored: true`

## Remaining note

This pass closes the dual-anchor export gap for 3D, 9D, Jacobian, and regional network layers in the same run. Reachability is still not a first-class standalone H5 surface in this repository, so no new reachability-specific export was added here.
