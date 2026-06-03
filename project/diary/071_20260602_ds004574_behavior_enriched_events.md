# ds004574 behavior-enriched events

Date: 2026-06-02

## Question

Can `ds004574` be pushed one step further than the initial conservative metadata
overlay by joining the richer per-trial subtype information from `beh.tsv` into
the EEG event stream, without adding new MNDM ingest code?

## Decision

Yes.

Instead of changing MNDM internals, an external staged dataset root was created:

- `M:/datasets/received/openneuro/ds004574_behavior_enriched`

This staged root keeps the original EEG payload untouched and rewrites only the
`*_events.tsv` files with additional columns derived from `beh.tsv`.

The repo config `mndm/config/config_ingest_ds004574.yaml` now points at this
staged root.

## Alignment finding

The key technical question was whether `beh.tsv` rows could be aligned reliably to
EEG trial markers.

Observed across the cohort:

- `146` subjects total
- `140` subjects classified as `good`
- `5` subjects classified as `usable`
- `1` subject classified as `partial`

Alignment method used in staging:

- derive a subject-specific constant offset between EEG `S  1` precue events and
  `beh.tsv` `TrialTime`
- search a small sequence shift to account for leading missing trials
- apply an order-preserving greedy match with a tolerance window

This was sufficient to recover trial-level subtype assignments across the cohort
without modifying MNDM code.

Alignment audit written to:

- `M:/datasets/received/openneuro/ds004574_behavior_enriched/behavior_event_alignment.tsv`

## Staged event enrichment

Each staged `*_events.tsv` now includes columns such as:

- `event_role`
- `trial_sequence_index`
- `trial_subtype`
- `event_subtype_role`
- `beh_trial_num`
- `beh_block`
- `beh_trial_time_sec`
- `beh_rt_sec`
- `beh_accuracy`
- `beh_cue_direction`
- `beh_response_direction`
- `beh_odd_visual`
- `beh_odd_audio`
- `beh_alignment_status`
- `beh_alignment_offset_sec`
- `beh_alignment_residual_sec`

Important labels now available in the event stream:

- `standard_*`
- `oddball_auditory_*`
- `oddball_visual_*`

The staged root also carries an augmented:

- `task-Oddball_events.json`

documenting the added columns.

## Repo change

Updated:

- `mndm/config/config_ingest_ds004574.yaml`

Key config changes:

- point `dataset_received_dirs.ds004574` to the staged root
- use `stage_columns: ["event_subtype_role"]` only as an event provenance surface
- keep `prefer_events_stage_in_summary: false` so point events do not become a
  misleading canonical per-window stage
- enable `event_mapping` so summarize emits binary labels from the enriched raw
  event families

## Validation

Preflight:

```powershell
python -m mndm.cli prerequisite-check --dataset ds004574 --config mndm/config/config_ingest_ds004574.yaml
```

Result:

- `overall_ok: True`
- indexed `146` EEG files from the staged root

Focused summarize validation:

```powershell
python -m mndm.cli summarize --dataset ds004574 --config mndm/config/config_ingest_ds004574.yaml --subject 001
python -m mndm.cli summarize --dataset ds004574 --config mndm/config/config_ingest_ds004574.yaml --subject 099
```

Validated run directory:

- `M:/datasets/processed/openneuro/ds004574/neuralmanifolddynamics_ds004574_20260602_061052`

Confirmed from `run_manifest.json` and the subject H5 outputs:

- dataset received root = `M:/datasets/received/openneuro/ds004574_behavior_enriched`
- `labels_stage = true` only because a raw event provenance surface was attached
- label keys now include:
  - `raw_standard_precue_onset_sec`
  - `raw_standard_arrowcue_onset_sec`
  - `raw_standard_response_onset_sec`
  - `raw_oddball_auditory_precue_onset_sec`
  - `raw_oddball_auditory_arrowcue_onset_sec`
  - `raw_oddball_auditory_response_onset_sec`
  - `raw_oddball_visual_precue_onset_sec`
  - `raw_oddball_visual_arrowcue_onset_sec`
  - `raw_oddball_visual_response_onset_sec`

Also confirmed from `stage_mapping_qc.json`:

- `source_event_column = "event_subtype_role"`
- subtype-specific raw event label counts are present
- the events remain intentionally unmapped to canonical stage codes

Additional spot check:

```powershell
python -m mndm.cli summarize --dataset ds004574 --config mndm/config/config_ingest_ds004574.yaml --subject 025
```

Observed:

- `sub-025` reported `No epochs`

That warning is about available epochs for that subject in the existing feature
layer, not a failure of the behavior-enriched event staging itself.

## Evidence category

- Internal validated result:
  - `ds004574` now has a no-new-code path for behavior-enriched oddball subtype
    event provenance
  - summarize emits subtype-aware binary labels from the staged event stream
  - the raw dataset was left untouched; enrichment lives in an external staged root
