# 098 20260604 cohort_anchor_emission_fix

## Context

Follow-up debugging was requested because `mnps_projection.export_contracts.cohort_anchored: true`
was set in dataset configs, but `run_manifest.json` kept reporting
`requested_but_not_emitted_in_probed_h5` for the cohort contract.

Observed behavior from prior runs:

- `coords_*_subject_anchored`: present
- `coords_*_cohort_anchored`: absent
- `jacobian*_subject_anchored`: present
- `jacobian*_cohort_anchored`: absent

## Root cause

The cohort contract was requested, but no active feature anchor was configured.

Current export logic requires an active anchor source (`mnps_projection.anchor.enabled=true`
or `mnps_projection.anchor_auto_fit.enabled=true`) to emit cohort-anchored layers.
Without this, export-contract resolution correctly skipped cohort emission.

## Fix implemented

Enabled one-shot anchor fitting in:

- `mndm/config/config_ingest_ds006036.yaml`

Added:

- `mnps_projection.anchor_auto_fit.enabled: true`
- stable dataset `anchor_id`
- explicit `anchor_source`
- `scale_method: "iqr"`
- `min_subjects: 3`

Also updated config templates to document both valid cohort-enabling options:

- external frozen anchor (`mnps_projection.anchor`)
- one-shot per-run fit (`mnps_projection.anchor_auto_fit`)

Scope note:

- Kept the runtime fix scoped to `ds006036` config.
- Briefly tested the same auto-fit pattern on `ds003490` single-subject smoke, where
  cohort anchor fitting can be unusable under tight filtering; reverted that broadening
  to avoid introducing regressions in existing smoke workflows.

## Validation

Ran full-cohort summarize for ds006036:

```powershell
python -m mndm.cli summarize --dataset ds006036 --config mndm/config/config_ingest_ds006036.yaml --n-jobs 1
```

Result:

- exit code: `0`
- run dir: `E:\Science_Datasets\openneuro\processed\ds006036\neuralmanifolddynamics_ds006036_20260604_132747`
- log confirms one-shot fit:
  - `Fitted one-shot feature anchor for ds006036: ...\anchors\ds006036_cohort_auto_v2_1.json`

Verified in `run_manifest.json`:

- `coordinate_contracts.requested_contracts = ["subject_anchored", "cohort_anchored"]`
- `coordinate_contracts.realized_contracts = ["subject_anchored", "cohort_anchored"]`
- `coordinate_contracts.skipped_contracts_with_reason = []`
- `h5_with_coords_3d_cohort_anchored = 88`
- `h5_with_coords_9d_cohort_anchored = 88`

Verified in a representative H5 (`sub-001`):

- `coords_3d_cohort_anchored`: present
- `coords_9d_cohort_anchored`: present
- `jacobian_cohort_anchored`: present
- `jacobian_9D_cohort_anchored`: present

## Notes

This resolves the practical contract gap for `ds006036`, where dual-anchor output
was requested but no active anchor configuration was present.
