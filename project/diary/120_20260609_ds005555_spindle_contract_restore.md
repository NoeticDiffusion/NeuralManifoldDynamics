## Session: ds005555 spindle contract restore

### Date
2026-06-09

### Goal
Restore the sleep-spindle event-locked contract for `ds005555` so article scripts can find and consume:

- legacy-named sidecars: `*_event_locked_v1_psg_c3.parquet` / `*_event_locked_v1_psg_f3.parquet`,
- spindle-compatible `condition` labels,
- and required summary columns (including `rate_per_min`).

### Code changes
1. `mndm/src/mndm/pipeline/summary.py`
   - Added event-locked source-kind resolution helper.
   - Added CSV source discovery from per-run `*_events.tsv` context.
   - Added default spindle CSV discovery pattern (`*_spindles_yasa_v1_*.csv`).
   - Routed `event_locked` export through discovered CSV sources (instead of task-state-derived segments when CSV sources are available).
   - Added safe filename tokenization and legacy export prefix construction to avoid invalid characters in Windows filenames.
   - Kept `task_state_label` segmentation as a fallback only when no concrete event source export runs.

2. `mndm/src/mndm/pipeline/event_locked_export.py`
   - Added `_estimate_rate_per_min(...)` and exported `rate_per_min` in both event and matched-control rows.
   - This restores compatibility with downstream scripts expecting this field in sidecars.

3. `mndm/config/config_ingest_ds005555_sleep_spindles.yaml`
   - Declared `event_source.kind: csv`.
   - Added `csv_source_glob: "{events_core}_spindles_yasa_v1_*.csv"`.
   - Set export labels:
     - `event_condition_label: "spindle_event"`
     - `control_condition_label: "matched_control"`

4. Tests
   - Added `mndm/tests/test_event_locked_source_resolution.py` to lock source discovery behavior.
   - Extended `mndm/tests/test_event_locked_export.py` with `rate_per_min` regression coverage.

### Validation
Passed:

- `python -m pytest "mndm/tests/test_event_locked_export.py" "mndm/tests/test_event_locked_runner.py" "mndm/tests/test_event_locked_source_resolution.py" -q`

Smoke run:

- `python -m mndm.cli summarize --dataset ds005555 --subject sub-8 --config "mndm/config/config_ingest_ds005555_sleep_spindles.yaml" --out-dir "M:/datasets/processed/openneuro" --n-jobs 1`

Observed in smoke output:

- Sidecars written for both channels:
  - `sub-8_Sleep_acq-psg_event_locked_v1_psg_c3.parquet`
  - `sub-8_Sleep_acq-psg_event_locked_v1_psg_f3.parquet`
- `condition` values: `spindle_event`, `matched_control`
- `bin_label` includes `control` for matched controls
- `alignment_reference`: `peak`
- `rate_per_min` present and finite.

### Notes
- This restores the spindle event-locked contract path expected by article scripts and keeps generic fallback behavior available when no CSV source resolves.
