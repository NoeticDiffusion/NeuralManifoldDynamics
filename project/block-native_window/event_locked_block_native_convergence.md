# Event-Locked -> Block-Native Convergence Plan

## Goal

Converge toward a single interval-native contract where:

- `/blocks/*` is the canonical inferred interval source.
- `/block_windows/*` is the canonical in-block analysis grid.
- `event_locked` block-end bins are derived views over those canonical intervals.

The migration must stay non-breaking for existing `event_locked` downstream users.

## Phase 0 (current)

- Keep both systems active in parallel.
- `event_locked` continues to emit its current sidecars.
- `block_native` emits additive H5 groups and sidecars.
- Use config parity checks in analysis notebooks to compare equivalent bins.

## Phase 1 (compatibility bridge)

- Emit explicit block-native metadata in `summary.json`:
  - `block_source_kind`
  - `window_profile_kind`
  - `source_window_index` join coverage
- Keep event-locked profiles unchanged but document canonical equivalents:
  - `in_block_tail_ms[-8,0]` <-> `profile: tail8`
  - `post_block_early_ms[0,8]` <-> `profile: post_offset_0_8`
  - `post_block_late_ms[8,16]` <-> `profile: post_offset_8_16`

## Phase 2 (derived event view)

- Add optional mode in `event_locked` to consume canonical blocks directly.
- Derive synthetic point-events from block-native intervals at:
  - block start
  - block end
  - configurable offsets
- Keep output schema stable while source inference shifts to `/blocks`.

## Phase 3 (default convergence)

- Make block-native-derived source the default for block-end event profiles.
- Retain legacy source path behind explicit opt-out toggle.
- Mark legacy path as deprecated in docs once parity is validated on:
  - ds006036
  - ds003490
  - ds003509
  - ds003506

## Exit Criteria

- Equivalent windows selected for canonical paired profiles across all validation datasets.
- No regressions in existing event-locked summary stats.
- Downstream analysis repos can move to `/block_windows/source_window_index` joins without fallback heuristics.
