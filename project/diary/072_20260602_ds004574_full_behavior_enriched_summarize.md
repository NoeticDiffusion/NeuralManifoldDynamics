# ds004574 full behavior-enriched summarize

Date: 2026-06-02

## Question

Now that `ds004574` has an external behavior-enriched staged root, can the full
cohort be processed end-to-end so the article-facing summarize outputs carry the
new oddball subtype provenance?

## Commands

```powershell
python -m mndm.cli features --dataset ds004574 --config mndm/config/config_ingest_ds004574.yaml --n-jobs 4
python -m mndm.cli summarize --dataset ds004574 --config mndm/config/config_ingest_ds004574.yaml --n-jobs 4
```

## Feature result

The full feature pass completed successfully.

Observed from `M:/datasets/processed/openneuro/ds004574/features.parquet`:

- `27715` epoch rows
- `146` unique source files

This confirms the full cohort now has feature coverage under the
behavior-enriched config.

## Summarize result

The full summarize pass also completed successfully.

Run directory:

- `M:/datasets/processed/openneuro/ds004574/neuralmanifolddynamics_ds004574_20260602_062756`

Key counts from `run_manifest.json`:

- `h5: 146`
- `summary_json: 146`
- `qc_summary_json: 146`
- `qc_reliability_json: 146`

Capabilities confirmed for all H5 outputs:

- `mnps3d: true`
- `mnps9d: true`
- `mnj: true`
- `coords_3d_subject_anchored: true`
- `coords_9d_subject_anchored: true`
- `h5_with_stage: 146`
- `h5_with_raw_features: 146`
- `h5_with_robust_z_features: 146`

Subject/task/condition coverage confirmed:

- `146` subjects
- task = `oddball`
- condition = `not_applicable`
- groups = `Parkinson`, `Control`

## Event provenance result

The new subtype-aware labels are present in the full-cohort summarize outputs,
including:

- `raw_boundary_onset_sec`
- `raw_oddball_auditory_precue_onset_sec`
- `raw_oddball_auditory_arrowcue_onset_sec`
- `raw_oddball_auditory_response_onset_sec`
- `raw_oddball_visual_precue_onset_sec`
- `raw_oddball_visual_arrowcue_onset_sec`
- `raw_oddball_visual_response_onset_sec`
- `raw_standard_precue_onset_sec`
- `raw_standard_arrowcue_onset_sec`
- `raw_standard_response_onset_sec`

This means the full cohort now carries article-relevant trial subtype provenance
without introducing new ingest code.

## Residual caveat

One residual cleanup target remains:

- `raw_nan_onset_sec` still appears in the run-manifest label keys

This is not a run failure. It reflects unmatched or partially matched enriched
event rows in a minority of subjects.

Quick audit from `stage_mapping_qc.json`:

- `27` subjects still contain at least one `nan` raw event label
- the worst case is `sub-025`

The full run still completed cleanly for all `146` subjects, but if we want a
fully polished article-ready event namespace, the next small cleanup would be to
replace these residual unmatched labels with a more explicit sentinel such as
`unmatched`.

## Evidence category

- Internal validated result:
  - `ds004574` has now been processed end-to-end with the behavior-enriched staged
    event stream
  - the full cohort summarize outputs preserve subtype-aware oddball provenance
  - the remaining residual issue is a namespace cleanup problem, not a failed run
