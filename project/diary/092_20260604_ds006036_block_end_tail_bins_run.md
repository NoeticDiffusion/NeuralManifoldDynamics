# 092 20260604 ds006036 block end tail bins run

## Research question

Can `ds006036` be rerun with a block-end event-locked hypothesis layout that
tests a primary late-tail window plus early and late post-block windows, while
also supporting a separate exploratory short-tail export?

## Config decision

Updated `mndm/config/config_ingest_ds006036.yaml` so the primary block-end
event-locked profile uses:

- `in_block_tail_ms: [-8.0, 0.0]`
- `post_block_early_ms: [0.0, 8.0]`
- `post_block_late_ms: [8.0, 16.0]`

Important implementation detail:

- `exclude_stage_transition_margin_sec: 0.0`

This was necessary because derived block-end events occur exactly at block
boundaries; leaving the default stage-transition exclusion active removed all
event rows and left only matched controls.

Because event-locked bins are first-match-wins and therefore cannot overlap in a
single profile, the exploratory short tail `[-4, 0]` was run as a separate
sidecar export rather than in the same primary bin set.

## Dataset run used

- Base processed run:
  `E:/Science_Datasets/openneuro/processed_ds006036_overlap075_subject_and_cohort_v1/ds006036/neuralmanifolddynamics_ds006036_20260603_141859`
- Raw events source:
  `E:/Science_Datasets/openneuro/received/ds006036/sub-XXX/eeg/sub-XXX_task-photomark_events.tsv`

## Outputs written

Per-subject sidecars were written beside each H5, for example:

- `sub-001_eyes_open_photic_photomark_stage_block_end_event_locked_tail8.parquet`
- `sub-001_eyes_open_photic_photomark_stage_block_end_event_locked_tail8.csv`
- `sub-001_eyes_open_photic_photomark_stage_block_end_event_locked_tail4.parquet`
- `sub-001_eyes_open_photic_photomark_stage_block_end_event_locked_tail4.csv`

Run-level summaries:

- `primary_tail8_summary.json`
- `exploratory_tail4_summary.json`

## Aggregate results

### Primary tail-8 export

- subjects processed: `88`
- derived block-end events total: `407`
- event rows total: `2339`
- matched-control rows total: `1034`
- bin counts:
  - `in_block_tail_ms`: `813`
  - `post_block_early_ms`: `777`
  - `post_block_late_ms`: `749`

### Exploratory tail-4 export

- subjects processed: `88`
- derived block-end events total: `407`
- event rows total: `1932`
- matched-control rows total: `1034`
- bin counts:
  - `in_block_tail_short_ms`: `406`
  - `post_block_early_ms`: `777`
  - `post_block_late_ms`: `749`

## Outcome

The first real `ds006036` block-end event-locked dataset run now exists in two
usable forms:

- a primary `tail_8` export for the main hypothesis
- a separate exploratory `tail_4` export for a stricter late-tail probe

The critical runtime lesson was that derived block-end events should not inherit
the default stage-transition exclusion margin in this profile.
