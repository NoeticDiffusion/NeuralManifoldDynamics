# Block-Native Window: Implementation Milestones

This document translates the architectural plan in `implementation_plan.md`
into concrete, sequenced, testable milestones. Each milestone has a clear
goal, a precise list of files to create or modify, a minimum set of test
cases, and a defined success criterion.

## Dataset test matrix

| Dataset   | Block type                  | Block source kind  | Stage events                               |
|-----------|-----------------------------|--------------------|---------------------------------------------|
| ds006036  | Photic stimulation          | `stage_blocking`   | PHOTO 3–30 Hz (regex-matched)               |
| ds003490  | Eyes-open / eyes-closed rest| `duration_events`  | "Eyes Closed: Every 1000 ms" (10), EO (11) |
| ds003509  | Simon conflict task phases  | `task_phase`       | `training_*` prefix, `test_*` prefix        |
| ds003506  | RL choose/match task phases | `task_phase`       | `choose_*` prefix, `match_*` prefix         |

---

## M1 — Core infrastructure: block_windows + block_native_config

**Goal**: introduce block-native objects and config parsing without changing
any runtime behavior. No dataset needs to be run.

### Files to create

#### `mndm/src/mndm/pipeline/block_windows.py`

Core window generation logic.

Public surface:
- `BlockWindowSpec` — frozen dataclass for window generation profile.
  Fields: `kind` (sliding / tail / post_offset / partitioned),
  `window_length_sec`, `step_sec`, `tail_sec`, `post_offset_bins`,
  `partitions`, `min_windows_per_block`, `min_block_sec`.
- `BlockWindowRow` — frozen dataclass for one generated window.
  Fields: `block_id`, `window_id_within_block`, `stage_code`,
  `block_start_sec`, `block_end_sec`, `block_duration_sec`,
  `window_start_sec`, `window_end_sec`, `window_center_sec`,
  `relative_time_in_block_sec`, `distance_to_block_end_sec`,
  `relative_pos_0_1`, `partition_label`, `is_post_offset`.
- `generate_block_windows(blocks, spec)` → `List[BlockWindowRow]`
  Dispatches to the correct internal generator based on `spec.kind`.

Internal generators:
- `_generate_sliding_windows` — dense sliding over full block or partitions.
- `_generate_tail_windows` — sliding over a tail window anchored at block end.
- `_generate_post_offset_windows` — windows in named bins after block end.

#### `mndm/src/mndm/pipeline/block_native_config.py`

Config parsing for all block-native YAML keys.

Public surface:
- `BlockWindowProfileConfig` — frozen dataclass (parsed from
  `block_native.datasets.<id>.window_profile`).
- `BlockSourceConfig` — frozen dataclass (parsed from
  `block_native.datasets.<id>.source`). Contains `kind` field plus
  kind-specific sub-fields.
- `BlockNativeDatasetConfig` — top-level per-dataset config container.
- `analysis_mode_from_config(config)` → `str`  (`"global"` or `"block_native"`).
- `block_native_dataset_config_from_config(config, dataset_id)` →
  `BlockNativeDatasetConfig`.
- `infer_blocks_from_events(events_df, source_cfg, *, stage_map, sampling_cfg)`
  → `List[StageBlockInterval]` — dispatches to the correct block inference
  function based on `source_cfg.kind`.

#### `mndm/tests/test_block_windows.py`

Minimum test cases (pytest):

1. **Sliding windows — basic geometry**
   - Block `[0, 20]`, `window_length_sec=4`, `step_sec=2`.
   - Expected: 9 windows, first at `[0, 4]`, last starts at `[16, 20]`.
   - All `relative_pos_0_1` in `[0, 1]`.
   - `distance_to_block_end_sec` for last window center ≈ `2.0`.

2. **Sliding windows — short block skipped**
   - Block `[0, 3]`, `window_length_sec=4`.
   - Expected: 0 windows (block too short for even one window).

3. **Tail windows — correct anchor**
   - Block `[0, 20]`, `tail_sec=8`, `window_length_sec=4`, `step_sec=2`.
   - Expected: 3 windows, all starting ≥ `12.0`.
   - `partition_label` is `"tail"` for all.

4. **Post-offset windows**
   - Block `[10, 30]`, post_offset_bins `[("post_early", 0, 8), ("post_late", 8, 16)]`,
     `window_length_sec=4`, `step_sec=4`.
   - Windows should be generated from `30` onwards.
   - `is_post_offset` is `True` for all.

5. **Partitioned windows — sub-interval only**
   - Block `[0, 20]`, partitions `[("early", 0, 8), ("tail", -8, 0)]`,
     `window_length_sec=2`, `step_sec=2`.
   - Windows in `[0, 8]` labeled `"early"`, windows in `[12, 20]` labeled `"tail"`.

6. **Relative position correctness**
   - Block `[10, 30]`, sliding, `window_length_sec=4`.
   - First window center at 12 → `relative_time_in_block_sec ≈ 2`.
   - `relative_pos_0_1 ≈ 0.1`.

7. **Adjacent blocks — no window leakage**
   - Two blocks `[0, 10]` and `[10, 20]`, sliding `step=4`.
   - No window `window_start_sec` below 0 or above 20.
   - No window spans both blocks.

8. **Min windows per block filter**
   - Block `[0, 5]`, `window_length_sec=4`, `min_windows_per_block=2`.
   - Expected: 0 (only 1 window fits, below minimum).

### Success criterion

All `test_block_windows.py` tests pass. No existing tests break.

---

## M2 — Block-source abstraction layer

**Goal**: generalize block inference in `block_native_config.py` to support
three source kinds without adding a new file. The dispatcher is the
`infer_blocks_from_events` function already defined in M1.

### Source kinds

#### `stage_blocking` (existing infrastructure, ds006036)

- Delegates to `infer_stage_block_intervals()` from
  `mndm/src/mndm/pipeline/stage_blocking.py`.
- Requires `sampling_cfg.stage_blocking.enabled: true`.
- The `BlockSourceConfig` for this kind carries no extra fields beyond
  column names; the real config lives in `epoching.datasets.<id>.sampling.stage_blocking`.

#### `duration_events` (ds003490)

- Reads events with explicit `duration` field from events_tsv.
- Filters rows to those matching `block_event_labels`.
- Infers one block per matching event: `[onset, onset + duration]`.
- Config fields: `block_event_labels`, `block_event_stage_codes`,
  `min_block_sec`, `max_block_sec`, `label_column`.

Example YAML (ds003490):

```yaml
block_native:
  datasets:
    ds003490:
      enabled: true
      source:
        kind: "duration_events"
        label_column: "trial_type"
        block_event_labels:
          - "Eyes Closed: Every 1000 ms"
          - "Eyes Open: Every 1000 ms"
        block_event_stage_codes:
          "Eyes Closed: Every 1000 ms": 10
          "Eyes Open: Every 1000 ms": 11
        min_block_sec: 30.0
        max_block_sec: 600.0
      window_profile:
        kind: "sliding"
        window_length_sec: 8.0
        step_sec: 4.0
        emit_relative_position: true
      export:
        write_parquet: true
        write_csv: true
```

#### `task_phase` (ds003509, ds003506)

- Groups consecutive events sharing the same phase prefix into a block.
- A block ends when: the prefix changes, or the gap to the next event
  exceeds `gap_tolerance_sec`.
- Config fields: `phase_prefixes` (dict: phase_name → prefix string),
  `gap_tolerance_sec`, `min_block_sec`, `label_column`.

Example YAML (ds003509):

```yaml
block_native:
  datasets:
    ds003509:
      enabled: true
      source:
        kind: "task_phase"
        label_column: "event_subtype_role"
        phase_prefixes:
          training: "training_"
          test: "test_"
        gap_tolerance_sec: 10.0
        min_block_sec: 20.0
        max_block_sec: 600.0
      window_profile:
        kind: "sliding"
        window_length_sec: 4.0
        step_sec: 2.0
        emit_relative_position: true
      export:
        write_parquet: true
        write_csv: true
```

Example YAML (ds003506):

```yaml
block_native:
  datasets:
    ds003506:
      enabled: true
      source:
        kind: "task_phase"
        label_column: "event_subtype_role"
        phase_prefixes:
          choose: "choose_"
          match: "match_"
        gap_tolerance_sec: 10.0
        min_block_sec: 20.0
        max_block_sec: 600.0
      window_profile:
        kind: "sliding"
        window_length_sec: 4.0
        step_sec: 2.0
        emit_relative_position: true
      export:
        write_parquet: true
        write_csv: true
```

### Additional test cases for M2

9. **duration_events — basic block from explicit duration**
   - Events tsv with `onset=10`, `duration=60`, `trial_type="Eyes Closed: Every 1000 ms"`.
   - Expected: one `StageBlockInterval` from `[10, 70]`, `stage_code=10`.

10. **duration_events — max_block_sec cap applied**
    - `duration=700`, `max_block_sec=600`.
    - Expected: block ends at `onset + 600`.

11. **task_phase — two-phase separation**
    - Events: `training_stimulus` at 0, 1, 2; `test_stimulus` at 100, 101, 102.
    - `gap_tolerance_sec=5`, `min_block_sec=2`.
    - Expected: two `StageBlockInterval` objects — one for training, one for test.

12. **task_phase — gap breaks block**
    - Events: `training_stimulus` at 0, 1; then at 50, 51.
    - `gap_tolerance_sec=5`.
    - Expected: two training-phase blocks (gap at 49 s exceeds tolerance).

### Success criterion

Tests 9–12 pass. `infer_blocks_from_events` returns correct intervals for
all three source kinds in isolated unit tests.

---

## M3 — Sidecar-only export (ds006036 primary validation)

**Goal**: produce a block-native sidecar Parquet/CSV alongside the existing
`event_locked` output, without modifying the canonical HDF5 output.

### Files to create

#### `mndm/src/mndm/pipeline/block_native_export.py`

Sidecar export logic.

Public surface:
- `build_block_native_table(*, blocks, profile, payload, subject_id, ...)` →
  `List[Dict[str, Any]]` — one row per block-native window, with optional
  MNPS coordinate lookup when `payload` is provided.
- `write_block_native_parquet(rows, out_path)` → `Optional[Path]`
- `write_block_native_csv(rows, out_path)` → `Optional[Path]`
- `block_native_export_manifest_entry(rows, *, dataset_id, out_paths, profile)`
  → `Dict[str, Any]` — JSON-serializable dict for `run_manifest.json`.

Mandatory row columns:
`subject_id`, `session_id`, `run_id`, `dataset_id`, `block_id`,
`window_id_within_block`, `stage_code`, `block_start_sec`, `block_end_sec`,
`block_duration_sec`, `window_start_sec`, `window_end_sec`,
`window_center_sec`, `relative_time_in_block_sec`, `distance_to_block_end_sec`,
`relative_pos_0_1`, `partition_label`, `is_post_offset`.

Optional MNPS columns (when payload provided):
`m`, `d`, `e`, `m_dot`, `d_dot`, `e_dot`, `mnps_finite`.

### Files to modify

#### `mndm/config/config_ingest_ds006036.yaml`

Add `block_native:` section with ds006036-specific profile:

```yaml
block_native:
  datasets:
    ds006036:
      enabled: true
      source:
        kind: "stage_blocking"
        label_column: "value"
        onset_column: "onset"
        duration_column: "duration"
      window_profile:
        kind: "sliding"
        window_length_sec: 4.0
        step_sec: 2.0
        emit_relative_position: true
        min_block_sec: 4.0
        min_windows_per_block: 2
      export:
        write_parquet: true
        write_csv: true
```

### Validation (manual inspection after first run)

Compare against `event_locked` block-end profile:
- `event_locked.bins.in_block_tail_ms: [-8, 0]` covers 4 windows in tail
  with `window_length_sec=4`, `step_sec=2`.
- Block-native tail profile: `tail_sec=8`, same length/step.
- The two sets should map to identical time ranges for complete blocks.

Metrics to record:
- Total blocks found.
- Total windows generated.
- Distribution of `block_duration_sec`.
- Distribution of `relative_pos_0_1` (should fill 0–1 continuously).
- Fraction of windows with valid `mnps_finite == 1`.

### Success criterion

ds006036 produces a block-native Parquet sidecar with `block_id`,
`relative_time_in_block_sec`, and `distance_to_block_end_sec` populated for
all windows, without modifying any existing H5 file.

---

## M4 — Dataset adapters for task-based datasets

**Goal**: add `block_native:` YAML sections to ds003490, ds003509, and
ds003506. Verify block inference returns sensible intervals for each.

### Files to modify

- `mndm/config/config_ingest_ds003490.yaml` — add `block_native:` with
  `source.kind: duration_events`.
- `mndm/config/config_ingest_ds003509.yaml` — add `block_native:` with
  `source.kind: task_phase`, `phase_prefixes: {training: "training_", test: "test_"}`.
- `mndm/config/config_ingest_ds003506.yaml` — add `block_native:` with
  `source.kind: task_phase`, `phase_prefixes: {choose: "choose_", match: "match_"}`.

### Success criterion

`block_native_dataset_config_from_config(config, dataset_id)` returns
`enabled=True` and the correct `source.kind` for all three datasets.
A smoke-test on real events_tsv data yields at least one block per subject.

---

## M5 — H5 additive contract

**Goal**: add optional `/blocks/` and `/block_windows/` groups to HDF5
outputs when block-native windows are present. Existing readers that do not
read these groups are unaffected.

### Files to modify

#### `mndm/src/mndm/schema.py`

Add two new optional fields to `MNPSPayload`:

```python
block_table_columns: MutableMapping[str, Any] = field(default_factory=dict)
block_window_table_columns: MutableMapping[str, Any] = field(default_factory=dict)
```

Update `as_dict()` to include them.

Update `normalize_payload()` to normalize them via
`_normalize_columnar_mapping()` when non-empty.

#### `core/src/core/io/h5_writer.py`

After writing `/event_windows/`, add:

```python
if payload.block_table_columns:
    _write_nested_mapping_group(f, "blocks", payload.block_table_columns,
                                schema_version="block_native_v1")
if payload.block_window_table_columns:
    _write_nested_mapping_group(f, "block_windows", payload.block_window_table_columns,
                                schema_version="block_native_v1")
```

Minimal `/blocks/` datasets:
`block_id`, `stage_code`, `start_sec`, `end_sec`, `duration_sec`.

Minimal `/block_windows/` datasets:
`block_id`, `window_id_within_block`, `window_start_sec`, `window_end_sec`,
`window_center_sec`, `relative_time_in_block_sec`, `distance_to_block_end_sec`,
`relative_pos_0_1`.

#### `mndm/src/mndm/pipeline/run_manifest.py`

In the capability-flags section, add:

```python
"has_block_native_windows": bool(
    isinstance(getattr(payload, "block_window_table_columns", None), Mapping)
    and len(payload.block_window_table_columns) > 0
),
```

### Success criterion

An H5 file written from a payload that contains `block_table_columns` and
`block_window_table_columns` gains `/blocks/` and `/block_windows/` groups.
`run_manifest.json` gains `has_block_native_windows: true` for that run.
Existing H5 files that have no block data remain unchanged.

---

## M6 — Summarize / MNPS integration

**Goal**: make `mndm.cli all` understand `analysis_mode: block_native` as a
first-class execution path. Block-native windows become the analysis grid
instead of the global epoch grid.

### Files to modify

#### `mndm/src/mndm/features/epoch_selection.py`

Add a new entry point:

```python
def build_block_native_epoch_meta(
    blocks: Sequence[StageBlockInterval],
    spec: BlockWindowSpec,
) -> List[tuple[int, int, int]]:
    """Return (epoch_id, start_idx, end_idx) from block-native windows."""
```

This converts block windows to the same `(epoch_id, start_sample, end_sample)`
contract used downstream, preserving compatibility with the feature extractor.

#### `mndm/src/mndm/pipeline/summary.py`

In the run-level assembly function, detect `analysis_mode`:

```python
if analysis_mode_from_config(config) == "block_native":
    blocks = infer_blocks_for_run(...)
    epoch_meta = build_block_native_epoch_meta(blocks, spec)
    block_window_rows = generate_block_windows(blocks, spec)
    payload.block_window_table_columns = _rows_to_columns(block_window_rows)
    payload.block_table_columns = _blocks_to_columns(blocks)
else:
    epoch_meta = build_epoch_meta(...)  # existing path
```

### Design constraints

- The global epoch path (`analysis_mode: global`) must remain unchanged.
- `block_native` mode should still support Jacobian computation over the
  block-native window grid (Jacobians on block-internal trajectories).
- Relative block position should be preserved as a label or annotation
  alongside the MNPS time axis.

### Success criterion

A `ds006036` run with `analysis_mode: block_native` produces:
- MNPS trajectory covering only windows inside inferred photic blocks.
- H5 output with `/blocks/` and `/block_windows/` present.
- `run_manifest.json` with `has_block_native_windows: true`.
- No regression in `analysis_mode: global` runs on any dataset.

---

## M7 — Comparative validation

**Goal**: demonstrate that block-native windows improve upon the current
overlap-based labeling approach, and that the approach generalizes cleanly
across all four test datasets.

### Comparison: ds006036

Run twice:

1. **Current config**: `window_membership.mode: overlap_frac_ge`,
   `min_overlap_fraction: 0.75`.
2. **Block-native config**: `analysis_mode: block_native` with
   `window_profile.kind: sliding`, `window_length_sec=4`, `step_sec=2`.

Metrics to compare:
- `stage_frac_labeled`: fraction of windows assigned a photic stage code.
- Boundary contamination rate (windows straddling block start/end).
- Distribution of `relative_pos_0_1` (block-native should cover 0–1 uniformly).
- Jacobian variance at block onset vs block tail
  (`relative_pos_0_1 < 0.2` vs `> 0.8`).

### Generalization: ds003490, ds003509, ds003506

For each dataset, verify:
- At least N blocks found per subject (N depends on task design).
- Windows generated span the expected phase durations.
- No empty-block subjects (all blocks meet `min_block_sec`).
- Sidecar Parquet is written and contains valid `relative_pos_0_1` values.

### Documentation deliverable

Write a diary entry in `project/diary/` after each validation run, including:
- Block counts per dataset.
- Window counts per block kind.
- MNPS trajectory coverage percentage (block-native vs global).
- Any boundary artifacts or malformed blocks found.
- Conclusion on methodological benefit.

---

## Backward compatibility requirements

- `analysis_mode: global` is the default; any config that does not set
  `analysis_mode` runs identically to today.
- Adding a `block_native:` section to a config file does not affect datasets
  whose `block_native.datasets.<id>.enabled` is `false` or absent.
- H5 groups `/blocks/` and `/block_windows/` are additive; existing HDF5
  readers that do not look for them are unaffected.
- The `event_locked` pipeline is not modified by block-native mode.
- All existing tests must continue to pass.

---

## Documentation to update after implementation

- `mndm/Output_variables_guide.md` — document `/blocks/` and `/block_windows/`
  H5 groups, their column semantics, and `has_block_native_windows` flag.
- `mndm/config/config_template.yaml` — add a commented `block_native:` section.
- `mndm/config/eeg_config_ingest_template.yaml` — add a photic-stimulation
  example `block_native:` section.
- `README.md` — brief mention of block-native mode alongside the `stage_blocking`
  and `event_locked` descriptions.
