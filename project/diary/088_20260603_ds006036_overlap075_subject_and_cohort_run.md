# ds006036 overlap 0.75 subject + cohort anchored run

## Research question

After choosing `overlap_frac_ge: 0.75` as the preferred compromise policy for
`ds006036`, can we produce a clean pair of outputs with both:

- subject-anchored summarize
- cohort-anchored summarize

under a fresh output root?

## Config basis

Active config:

- `mndm/config/config_ingest_ds006036.yaml`

Relevant stage-blocking policy:

```yaml
stage_blocking:
  window_membership:
    mode: "overlap_frac_ge"
    min_overlap_fraction: 0.75
```

## Validation before run

Ran:

- `pytest mndm/tests/test_epoch_selection_point_events.py mndm/tests/test_event_alignment.py`

Result:

- `19 passed`

## Fresh run

Output base:

- `E:/Science_Datasets/openneuro/processed_ds006036_overlap075_subject_and_cohort_v1`

Commands:

1. Features
   - `python -m mndm.cli features --dataset ds006036 --config H:/SourceRepo2/NeuralManifoldDynamics/mndm/config/config_ingest_ds006036.yaml --out-dir E:/Science_Datasets/openneuro/processed_ds006036_overlap075_subject_and_cohort_v1 --n-jobs 6`
2. Summarize (subject-anchor)
   - `python -m mndm.cli summarize --dataset ds006036 --config H:/SourceRepo2/NeuralManifoldDynamics/mndm/config/config_ingest_ds006036.yaml --out-dir E:/Science_Datasets/openneuro/processed_ds006036_overlap075_subject_and_cohort_v1 --n-jobs 6`
3. Summarize (cohort-anchor)
   - `python -m mndm.cli summarize --dataset ds006036 --config H:/SourceRepo2/NeuralManifoldDynamics/mndm/config/config_ingest_ds006036.yaml --out-dir E:/Science_Datasets/openneuro/processed_ds006036_overlap075_subject_and_cohort_v1 --n-jobs 6 --fit-anchor --anchor-id ds006036_overlap075_cohort_anchor_v1`

## Outputs

Subject-anchored run:

- `E:/Science_Datasets/openneuro/processed_ds006036_overlap075_subject_and_cohort_v1/ds006036/neuralmanifolddynamics_ds006036_20260603_135446`

Cohort-anchored run:

- `E:/Science_Datasets/openneuro/processed_ds006036_overlap075_subject_and_cohort_v1/ds006036/neuralmanifolddynamics_ds006036_20260603_140203`

Anchor file:

- `E:/Science_Datasets/openneuro/processed_ds006036_overlap075_subject_and_cohort_v1/ds006036/neuralmanifolddynamics_ds006036_20260603_140203/anchors/ds006036_overlap075_cohort_anchor_v1.json`

## Verification

- subject run manifest reports `fit_anchor: false`
- cohort run manifest reports `fit_anchor: true`
- cohort run manifest records:
  - `anchor_id: ds006036_overlap075_cohort_anchor_v1`
- both runs report:
  - `labels_stage: true`
  - `h5_with_stage: 88`
- full chained run completed with exit code `0`

## Outcome

The preferred `overlap 0.75` policy is now available in both subject-anchored
and cohort-anchored form under a single fresh output base, with an explicit new
cohort anchor file for downstream reuse.
