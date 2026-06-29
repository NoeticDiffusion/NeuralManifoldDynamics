# Block-Native Window Implementation Plan

## Goal

Define a high-level implementation plan for a **block-native / interval-native**
variant in NeuralManifoldDynamics where inferred stimulation blocks become
primary analysis objects and analysis windows are generated **from the block
itself**, rather than by labeling a pre-existing global epoch grid after the
fact.

This plan is intentionally architectural. It does **not** implement the design.

## Design Intent

The target design is **not**:

- one photic block = one single long epoch

The target design **is**:

- infer exact block intervals first
- treat blocks as first-class objects
- generate block-internal windows from those intervals
- attach block-relative coordinates to each window
- support whole-block, early, mid, tail, and post-block analyses from the same
  general contract

This keeps the dynamic character of NMD while removing the need to reconstruct
block membership from overlap rules after windows already exist.

## Non-Goals For The First Implementation

- Do not remove the current global-epoch pipeline immediately.
- Do not break existing `features -> summarize` datasets that are not
  block-native.
- Do not make a block-native design depend on `ds006036`-specific naming.
- Do not start with "one 20 s block = one datapoint".

## Recommended Strategy

Implement block-native support as an **additive execution path** that can
coexist with the current global epoching path.

Recommended high-level toggle:

- `analysis_mode: "global"` (existing default)
- `analysis_mode: "block_native"` (new path)

This avoids destabilizing the current pipeline while allowing one dataset to
exercise the new contract cleanly.

## Core Conceptual Objects

The implementation should introduce or formalize four concepts:

### 1. Block Interval

Primary absolute interval on the run clock.

Fields should include at least:

- `block_id`
- `stage_code`
- `block_start_sec`
- `block_end_sec`
- `block_duration_sec`
- `block_duration_ms`
- provenance such as `derived_from`, `end_reason`, `is_inferred`

### 2. Block-Native Window

Window generated **inside** a block or at a defined position relative to a
block boundary.

Fields should include at least:

- `block_id`
- `window_id_within_block`
- `window_start_sec`
- `window_end_sec`
- `window_center_sec`
- `relative_time_in_block_sec`
- `distance_to_block_end_sec`
- optionally normalized position such as `relative_pos_0_1`

### 3. Block Window Profile

Configuration object describing how windows are generated from blocks:

- full sliding windows within block
- tail-only windows
- early/mid/late partitions
- post-offset windows

### 4. Block-Level Export / H5 Contract

Additive storage contract for blocks and block-generated windows, without
overloading the meaning of the current global `/time` axis unless explicitly
chosen.

## Existing Files To Reuse

These files are the strongest existing seams and should be reused rather than
reimplemented.

### `mndm/src/mndm/pipeline/stage_blocking.py`

Use as the canonical block inference layer.

Why it matters:

- already defines `StageBlockInterval`
- already centralizes carrier/bridge semantics
- already contains provenance such as `end_reason`,
  `membership_mode`, `bridge_tail_sec`, and `is_inferred`

Planned role:

- remain the source of truth for inferred block intervals
- may need lightweight extension if block-native execution needs richer block
  metadata

### `mndm/src/mndm/pipeline/intervals.py`

Use for shared interval semantics and QC.

Why it matters:

- already defines `TimeInterval`
- already contains overlap and membership helpers

Planned role:

- still useful for validation and backward compatibility
- less central for window generation itself in block-native mode, because
  windows should already be valid by construction

### `mndm/src/mndm/pipeline/event_alignment.py`

Use as the model for relative-bin alignment and export semantics.

Why it matters:

- already supports bin definitions and relative-time logic
- already solves auditable row-level interval export

Planned role:

- reuse concepts and maybe some helper logic for block-relative bins
- likely should not be overloaded directly for all block-native work, because
  blocks are not exactly the same as point events

### `mndm/src/mndm/pipeline/event_locked_runner.py`

Use as the nearest existing interval-driven orchestration pattern.

Why it matters:

- already resolves derived block-end events from `StageBlockInterval`
- already shows how to bolt interval-driven exports on top of existing H5s

Planned role:

- can serve as transitional infrastructure and validation path
- useful for "block-end locked" analyses even after block-native windows exist

### `mndm/src/mndm/schema.py`

Use as the main payload contract definition.

Why it matters:

- `MNPSPayload` currently defines the H5-facing contract
- current `/time`, `/window_start`, `/window_end`, labels, events, and
  event-window groups are all shaped here

Planned role:

- extend rather than replace
- likely needs new optional block-native payload fields

### `core/src/core/io/h5_writer.py`

Use as the main H5 serialization point.

Why it matters:

- writes the payload contract to disk
- already handles additive groups like `/events` and `/event_windows`

Planned role:

- add new block-native groups in an additive way

### `mndm/src/mndm/pipeline/summary.py`

Use as the central summarize/MNPS assembly point.

Why it matters:

- assumes one per-run window axis today
- constructs `time`, `window_start`, `window_end`
- coordinates Jacobians, projections, labels, and export layers

Planned role:

- this is one of the key files that will need a controlled branch for
  `analysis_mode: block_native`

### `mndm/src/mndm/features/epoch_selection.py`

Use as the current global-window construction reference.

Why it matters:

- current epoch grid is defined here
- current stage-blocking only labels those windows after creation

Planned role:

- either branch or refactor so window creation can be block-first

## New Files To Create

The following new modules would make architectural sense and keep concerns
separated cleanly.

### `mndm/src/mndm/pipeline/block_windows.py`

Purpose:

- generate windows from `StageBlockInterval`
- support sliding, partitioned, tail-only, and post-offset profiles
- emit rich block-relative metadata

Recommended responsibilities:

- `BlockWindowSpec`
- `BlockWindowRow`
- `generate_block_windows(...)`
- validation helpers for short or malformed blocks

### `mndm/src/mndm/pipeline/block_native_config.py`

Purpose:

- parse dataset YAML for the new block-native mode

Recommended responsibilities:

- `analysis_mode`
- block-native window profile selection
- full-block vs tail vs post-offset configuration
- min windows per block
- normalized-position toggles

### `mndm/src/mndm/pipeline/block_native_export.py`

Purpose:

- export a row-level block-window table or block-summary table in a consistent,
  auditable format

Recommended responsibilities:

- block window sidecar export
- optional whole-block aggregate table
- manifest entry helper

### Optional: `mndm/tests/test_block_windows.py`

Purpose:

- focused unit tests for window generation inside intervals

This should exist early once implementation starts.

## Existing Files To Modify

### `mndm/src/mndm/features/epoch_selection.py`

Modify to support a branch where epoch windows are generated from inferred
blocks instead of from the global stride.

High-level change:

- add a block-native window generation path
- keep current global path untouched as default

### `mndm/src/mndm/features/eeg.py`

Modify only if feature extraction currently assumes global epoch meta in a way
that block-native windows cannot reuse directly.

High-level change:

- ensure `t_start` / `t_end` can come from block-native windows
- preserve downstream feature table compatibility

### `mndm/src/mndm/pipeline/summary.py`

Modify to understand block-native windows and possibly export new block-aware
payload groups.

High-level change:

- summarize should not assume that every window belongs to one global flat grid
  with no higher-level grouping
- add block-aware payload wiring

### `mndm/src/mndm/schema.py`

Extend with optional block-native payload fields.

Suggested additive fields:

- `block_table_columns`
- `block_window_table_columns`
- optional block-level labels or codebooks

### `core/src/core/io/h5_writer.py`

Extend H5 writing with additive groups such as:

- `/blocks/*`
- `/block_windows/*`

Recommended minimal block group:

- `/blocks/block_id`
- `/blocks/stage_code`
- `/blocks/start_sec`
- `/blocks/end_sec`
- `/blocks/duration_sec`
- `/blocks/duration_ms`

Recommended minimal block-window group:

- `/block_windows/block_id`
- `/block_windows/window_id_within_block`
- `/block_windows/window_start_sec`
- `/block_windows/window_end_sec`
- `/block_windows/window_center_sec`
- `/block_windows/relative_time_in_block_sec`
- `/block_windows/distance_to_block_end_sec`

### `mndm/src/mndm/pipeline/run_manifest.py`

Extend the manifest field guide and capability flags.

High-level change:

- add documentation for `/blocks` and `/block_windows`
- add capability flag such as `has_block_native_windows`

### `mndm/Output_variables_guide.md`

Update to document the new additive contract.

### `mndm/config/config_template.yaml`

Add example configuration for block-native mode.

### `mndm/config/eeg_config_ingest_template.yaml`

Add EEG-oriented example for photic/stimulation block-native analysis.

## Proposed Configuration Shape

Suggested direction:

```yaml
analysis_mode: "block_native"

block_native:
  datasets:
    ds006036:
      enabled: true
      source: "stage_blocking"
      window_profile:
        kind: "sliding_with_partitions"
        window_length_sec: 4.0
        step_sec: 2.0
        emit_full_block_windows: true
        emit_relative_position: true
        partitions:
          early: [0.0, 8.0]
          mid: [8.0, 16.0]
          tail: [-8.0, 0.0]
        post_offset:
          enabled: true
          bins:
            post_block_early_ms: [0.0, 8.0]
            post_block_late_ms: [8.0, 16.0]
```

The exact naming can change, but the important design choice is:

- block inference is reused from `stage_blocking`
- window generation is block-first
- relative block position becomes an explicit contract

## Recommended Implementation Phases

### Phase 1. Additive Block Contract

Goal:

- introduce block-native objects and config without changing runtime behavior

Work:

- create `block_native_config.py`
- create `block_windows.py`
- extend `StageBlockInterval` usage as block source
- add tests for block window generation

Success criterion:

- windows can be generated from blocks in memory with full provenance

### Phase 2. Sidecar-Only Block-Native Export

Goal:

- validate block-native semantics without touching the primary H5 contract yet

Work:

- create `block_native_export.py`
- export block-window tables as Parquet/CSV sidecars
- compare against current `event_locked` block-end analyses

Why:

- safest place to validate methodology before changing core payload semantics

Success criterion:

- one dataset can produce block-native sidecars with early/mid/tail/post bins

### Phase 3. H5 Additive Contract

Goal:

- promote validated block-native objects into H5 as additive groups

Work:

- extend `schema.py`
- extend `h5_writer.py`
- extend manifest docs and field guide

Success criterion:

- H5 contains `/blocks` and `/block_windows` without disturbing existing
  readers that do not care about them

### Phase 4. Summarize / MNPS Integration

Goal:

- make summarize understand block-native windows as a first-class execution path

Work:

- branch `summary.py`
- branch or refactor `epoch_selection.py`
- ensure derivatives, Jacobians, and coordinate export remain meaningful

Success criterion:

- a dataset can run through `features -> summarize` with block-native windows as
  the actual analysis grid

### Phase 5. Comparative Validation

Goal:

- show where block-native improves over current membership-overlay logic

Work:

- compare current `window_membership` approach vs block-native windows on
  `ds006036`
- compare:
  - early vs mid vs tail
  - tail8 vs short-tail
  - in-block vs post-block

Success criterion:

- methodology benefit is documented, not just architectural cleanliness

## Tests To Add

At minimum:

- block inference still yields stable intervals from the same events
- windows generated inside blocks are fully contained by construction
- short blocks and malformed blocks are either skipped or flagged explicitly
- adjacent blocks do not leak windows across boundaries
- post-offset windows are generated correctly
- H5 additive groups serialize with schema/version markers
- summarize/manifest behavior is unchanged for non-block-native datasets

## Important Constraints

### 1. Do not mix overlapping bins in one first-match export path

The current event/bin logic uses first-match-wins semantics. If overlapping
bins are needed (for example `tail8` and `tail4`), they should be:

- separate profiles, or
- represented by a block-window contract that supports multiple labels per
  window explicitly

### 2. Keep "block" terminology distinct

There are already "block" concepts around Jacobian or stratified exports. The
new design must distinguish:

- stage/stimulation blocks
- Jacobian aggregation blocks
- block-native windows

### 3. Preserve backward compatibility

The safest rollout is:

- sidecar-first
- additive H5 groups second
- only later allow block-native to become the actual primary analysis mode

## Block Source Kinds

Block inference must generalize beyond `stage_blocking` to support all test
datasets. Three source kinds are defined:

### `stage_blocking` — ds006036

Delegates to the existing `infer_stage_block_intervals()` pipeline.
The `BlockSourceConfig` for this kind requires `sampling_cfg.stage_blocking.enabled: true`
in the dataset's `epoching` section. No new inference logic is needed.

### `duration_events` — ds003490

Infers one block per event row that has an explicit `duration` field.
Suitable for datasets where EO/EC blocks are represented as single events with
known durations (e.g. `"Eyes Closed: Every 1000 ms"` in ds003490).

Config fields: `block_event_labels`, `block_event_stage_codes`, `label_column`,
`min_block_sec`, `max_block_sec`.

### `task_phase` — ds003509 / ds003506

Groups consecutive events that share a configured label prefix into a phase block.
A block boundary is emitted when: the prefix changes, or the gap to the next
event exceeds `gap_tolerance_sec`.

Suitable for Parkinson cognitive-task datasets with compact event families:
- ds003509: `training_*` vs `test_*` (Simon conflict).
- ds003506: `choose_*` vs `match_*` (reinforcement learning).

Config fields: `phase_prefixes` (dict: phase_name → prefix), `gap_tolerance_sec`,
`min_block_sec`, `label_column`.

See `project/block-native_window/milestones.md` for dataset-specific YAML
examples and test case specifications for all three source kinds.

## Recommended First Consumer

Use `ds006036` as the first consumer because:

- stage-block inference already exists and is exercised
- block-end event-locked sidecars already exist
- the scientific question specifically cares about early/mid/tail dynamics

## Deliverable For A Future Implementation Pass

A future implementation pass should aim to deliver:

1. a new block-native config section
2. a block-window generator module
3. sidecar exports for block-native windows
4. additive H5 `/blocks` and `/block_windows` groups
5. one validated `ds006036` block-native run

That is the smallest sequence that is both methodologically meaningful and
architecturally safe.
