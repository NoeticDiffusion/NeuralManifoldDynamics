# 055_20260528_ds006036_event_provenance_and_photic_blocks

## Research question
- Fix `ds006036` ingest/summary so downstream photic analysis can rely on ingest-certified stage surfaces and auditable event provenance.

## Implemented changes
- Added continuous photic block labeling with dataset-configurable `stage_blocking` policy:
  - expands sparse `PHOTO XHz` onset markers into stable per-window labels,
  - uses `Photo/HV mark` as support signal for block inference (without overriding inferred photic windows),
  - preserves non-photic labels only outside inferred photic blocks.
- Extended `ds006036` stage codebook/mapping:
  - `Photo/HV mark -> 54`,
  - `PHOTO 25Hz -> 55`,
  - `PHOTO 30Hz -> 56`.
- Added per-subject event provenance extraction from raw `*_events.tsv` into summarized payload:
  - `payload.events` now includes grouped raw-onset arrays (`raw_*_onset_sec`),
  - `payload.event_table_columns` now stores a columnar event table in H5 `/events/` (`raw_event_label`, `mapped_stage_code`, `onset_sec`, `duration_sec`, `mapping_mode`, etc.).
- Added summarize override control for stale feature-stage surfaces:
  - `epoching.datasets.<ds>.sampling.prefer_events_stage_in_summary: true`
  - lets `summarize` replace old `features.csv` stage labels with freshly inferred BIDS-event stage surfaces.
- Added stage-mapping QC reporting:
  - per-subject QC embedded in `summary.json` (`stage_mapping_qc`),
  - run-level `stage_mapping_qc.json` with aggregate + per-subject rows,
  - explicit `25/30 Hz` presence/absence flags and missing-expected-frequency reporting.
- Framework hardening (not dataset-locked):
  - removed dataset-specific stage-block defaults from code (no built-in `PHOTO*` / `Photo/HV mark` assumptions),
  - introduced generic stage-block config aliases:
    - `stage_event_regex` (alias of old `photic_regex`),
    - `bridge_marker_labels` (alias of old `hv_mark_labels`),
    - `use_bridge_markers` (alias of old `use_hv_marks`),
    - `preserve_block_assignments` (alias of old `preserve_photic_blocks`),
    - `expected_stage_frequencies_hz` (alias of old `expected_frequencies_hz`),
  - documented these defaults in shared templates (`config_ingest_common_eeg.yaml`, `config_template.yaml`),
  - added generic QC field `raw_stage_frequency_event_counts_hz` (kept `raw_photic_frequency_event_counts_hz` as compatibility alias).
- Added run-manifest integration:
  - `run_manifest.json` now includes `stage_mapping_qc` run-level report pointer + aggregate summary.

## Files touched
- `mndm/src/mndm/pipeline/summary_events.py`
- `mndm/src/mndm/pipeline/summary.py`
- `mndm/src/mndm/features/epoch_selection.py`
- `mndm/src/mndm/features/eeg.py`
- `mndm/config/config_ingest_ds006036.yaml`
- `mndm/tests/test_epoch_selection_point_events.py`
- `mndm/tests/test_sleep_stage_labels.py`

## Validation
- Tests:
  - `python -m pytest mndm/tests/test_epoch_selection_point_events.py mndm/tests/test_sleep_stage_labels.py -q`
  - Result: `9 passed`.
- Runtime smoke:
  - `python -m mndm.cli summarize --dataset ds006036 --config mndm/config/config_ingest_ds006036.yaml --subject 001 --n-jobs 1`
  - Result: successful run with:
    - non-empty `summary.json.events`,
    - populated `summary.json.stage_mapping_qc`,
    - populated `summary.json.event_provenance`,
    - run-level `stage_mapping_qc.json`,
    - H5 `/events/` containing both legacy onset arrays and event table columns.
  - Re-run after framework hardening also passed with the same outputs:
    - run dir `neuralmanifolddynamics_ds006036_20260528_082846`.

## Current limitations / observations
- For smoke subject `sub-001`, raw events contain only `5/10/15/20 Hz`; `25/30 Hz` are correctly reported as missing in QC.
- Control-character labels from raw TSV are preserved as-is in raw label counts (auditable but noisy).
- Regional surface scope remains unchanged in this patch (separate issue).

