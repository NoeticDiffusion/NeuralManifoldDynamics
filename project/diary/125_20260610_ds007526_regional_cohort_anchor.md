# Session Diary 125 - ds007526 regional cohort anchor

Date: 2026-06-10

## Goal

Keep the working regional `ds007526` run and add cohort-anchored exports using
the same `anchor_auto_fit` pattern already used in `ds006036`.

## Config change

Updated:

- `mndm/config/config_ingest_ds007526.yaml`

Added:

```yaml
mnps_projection:
  export_contracts:
    subject_anchored: true
    cohort_anchored: true
  anchor_auto_fit:
    enabled: true
    anchor_id: "ds007526_cohort_auto_v2_1"
    anchor_source: "ds007526_all_subjects_features_table"
    scale_method: "iqr"
    min_subjects: 3
```

This mirrors the `ds006036` pattern and lets summarize emit cohort-anchored
layers without supplying an external frozen anchor file by hand.

## Run

Ran summarize against the existing regional processed root:

```powershell
$env:PYTHONPATH="H:/SourceRepo2/NeuralManifoldDynamics/mndm/src;H:/SourceRepo2/NeuralManifoldDynamics/core/src;H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/apollo_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/vitaldb_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics"
python -m mndm.cli summarize --dataset ds007526 --config "H:/SourceRepo2/NeuralManifoldDynamics/mndm/config/config_ingest_ds007526.yaml" --data-dir "N:/received" --out-dir "N:/processed/openneuro_ds007526_regional" --n-jobs 4
```

Outcome:

- summarize completed successfully with `exit_code: 0`
- new run directory:
  - `N:/processed/openneuro_ds007526_regional/ds007526/neuralmanifolddynamics_ds007526_20260610_121725`

## Anchor verification

Observed early in the summarize log:

- `Fitted one-shot feature anchor for ds007526`

Anchor file present on disk:

- `N:/processed/openneuro_ds007526_regional/ds007526/neuralmanifolddynamics_ds007526_20260610_121725/anchors/ds007526_cohort_auto_v2_1.json`

## H5 verification

Inspected:

- `N:/processed/openneuro_ds007526_regional/ds007526/neuralmanifolddynamics_ds007526_20260610_121725/sub-029_rest/sub-029_rest.h5`

Confirmed top-level cohort exports:

- `coords_3d_subject_anchored`
- `coords_3d_cohort_anchored`
- `jacobian_cohort_anchored`
- `feature_anchors`

Confirmed regional cohort exports under:

- `/regional_mnps/frontal/cohort_anchored`

with datasets:

- `mnps`
- `mnps_dot`
- `jacobian`
- `stratified`

The sibling `subject_anchored` subgroup also remains present, so the regional run
now carries both contracts.

## Note

The summarize log also emitted at least one:

- `CRITICAL WARNING: 9D MNPS falsified for this session. Matrix is effectively 3-dimensional.`

This did not fail the run, but it is worth remembering as a real-data geometry
warning for at least one recording in this cohort-anchor pass.
