# ds006036 dual anchored export in same run

## Research question

Now that `mnps_projection.export_contracts` supports exporting both subject- and
cohort-anchored layers in the same summarize run, can `ds006036` be rerun so
that one fitted-anchor run contains both contracts together?

## Config updates

Updated:

- `mndm/config/config_template.yaml`
  - restored the template example to show:
    - `subject_anchored: true`
    - `cohort_anchored: true`
  - clarified that both can now be exported together in one run when a fitted
    or external anchor is active.

- `mndm/config/config_ingest_ds006036.yaml`
  - added:

```yaml
mnps_projection:
  export_contracts:
    subject_anchored: true
    cohort_anchored: true
```

## Summarize rerun

Used existing overlap-0.75 feature base:

- `E:/Science_Datasets/openneuro/processed_ds006036_overlap075_subject_and_cohort_v1`

Command:

- `python -m mndm.cli summarize --dataset ds006036 --config H:/SourceRepo2/NeuralManifoldDynamics/mndm/config/config_ingest_ds006036.yaml --out-dir E:/Science_Datasets/openneuro/processed_ds006036_overlap075_subject_and_cohort_v1 --n-jobs 6 --fit-anchor --anchor-id ds006036_overlap075_dual_contract_v1`

## Output

New run:

- `E:/Science_Datasets/openneuro/processed_ds006036_overlap075_subject_and_cohort_v1/ds006036/neuralmanifolddynamics_ds006036_20260603_141859`

Anchor:

- `E:/Science_Datasets/openneuro/processed_ds006036_overlap075_subject_and_cohort_v1/ds006036/neuralmanifolddynamics_ds006036_20260603_141859/anchors/ds006036_overlap075_dual_contract_v1.json`

## Verification

`run_manifest.json` for the new run reports:

- `fit_anchor: true`
- `labels_stage: true`
- `h5_with_stage: 88`
- `subject_anchored: true`
- `cohort_anchored: true`
- `anchor_id: ds006036_overlap075_dual_contract_v1`

## Outcome

`ds006036` now has a fitted-anchor summarize run where both anchored export
contracts are emitted together in the same run directory, rather than requiring
separate subject-anchor and cohort-anchor summarize runs.
