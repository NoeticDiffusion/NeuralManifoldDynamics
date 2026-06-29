# 095 — Block-Native Window Architecture: Implementation Complete, Validation Protocol

**Date**: 2026-06-04  
**Status**: M1–M6 implemented and tested. M7 (live dataset validation) pending first real-data run.

---

## What was implemented

The block-native window architecture (M1–M6) is now fully in place across the NMD pipeline. All code is additive — existing `analysis_mode: global` runs are unaffected.

### New files

| File | Purpose |
|------|---------|
| `mndm/src/mndm/pipeline/block_windows.py` | `BlockWindowSpec`, `BlockWindowRow`, `generate_block_windows()` |
| `mndm/src/mndm/pipeline/block_native_config.py` | Config parsing + `infer_blocks_from_events()` dispatcher |
| `mndm/src/mndm/pipeline/block_native_export.py` | Sidecar export, H5 column helpers, `inject_block_native_into_payload()` |
| `mndm/tests/test_block_windows.py` | 28 unit tests (M1 + M2), all passing |
| `project/block-native_window/milestones.md` | Full milestone spec M1–M7 |

### Modified files

| File | Change |
|------|--------|
| `mndm/src/mndm/schema.py` | Added `block_table_columns`, `block_window_table_columns` to `MNPSPayload` |
| `core/src/core/io/h5_writer.py` | Writes `/blocks/` and `/block_windows/` groups when columns are present |
| `mndm/src/mndm/pipeline/run_manifest.py` | `has_block_native_windows` capability flag + field guide entries |
| `mndm/src/mndm/features/epoch_selection.py` | Added `build_block_native_epoch_meta()` |
| `mndm/src/mndm/pipeline/summary.py` | M6 injection block before `write_summary_manifest_and_h5` |
| `mndm/config/config_ingest_ds006036.yaml` | Added `block_native:` section |
| `mndm/config/config_ingest_ds003490.yaml` | Added `block_native:` section |
| `mndm/config/config_ingest_ds003509.yaml` | Added `block_native:` section |
| `mndm/config/config_ingest_ds003506.yaml` | Added `block_native:` section |
| `project/block-native_window/implementation_plan.md` | Added block source kind documentation |

---

## M7: Validation protocol

### Primary validation — ds006036 (photic stimulation)

**Run command** (example):
```bash
python -m mndm.cli all --config mndm/config/config_ingest_ds006036.yaml
```

**Expected artifacts per subject/run:**
- H5 file contains `/blocks/` group: `block_id`, `stage_code`, `start_sec`, `end_sec`, `duration_sec`
- H5 file contains `/block_windows/` group: all `BlockWindowRow` columns
- `run_manifest.json` has `capabilities.has_block_native_windows: true`

**Key metrics to inspect:**
- `n_blocks` per run (expected: number of PHOTO stimulation epochs)
- Distribution of `block_duration_sec` (should match known PHOTO durations)
- Distribution of `relative_pos_0_1` (should be approximately uniform 0–1)
- `partition_label` = `""` for all windows (plain sliding mode)
- Overlap comparison with `event_locked.bins.in_block_tail_ms`:
  - With `window_length_sec=4`, `step_sec=2`, the last ~4 windows cover the same range as `in_block_tail_ms: [-8000, 0]`
  - These should identify the same MNPS windows as the event-locked `tail` bin

### Cross-dataset validation — ds003490, ds003509, ds003506

**ds003490 (EO/EC):**
- Expected blocks: 1–2 EO + 1–2 EC per run
- `block_duration_sec` should approximate the known EO/EC trial lengths
- `stage_code = 10` for EC, `stage_code = 11` for EO

**ds003509 (Simon conflict):**
- Expected blocks: 1 training block + 1 test block per run
- `block_duration_sec` should cover the full phase duration (minutes, not seconds)
- `stage_code = 1` for training, `stage_code = 2` for test
- No blocks found → check `gap_tolerance_sec` and event label format in actual events_tsv

**ds003506 (RL choose/match):**
- Expected blocks: multiple choose and match blocks interleaved
- `block_duration_sec` should reflect the choose/match trial structure
- Check actual `event_subtype_role` column values match `choose_` / `match_` prefixes

---

## Known limitations at M7

1. **M6 injection uses `stage_events_path`**: if this path is None (e.g. no BIDS events_tsv for the run), block-native injection is silently skipped. Verify `stage_events_path` is populated for all four target datasets.

2. **`stage_blocking` source kind needs `sampling_cfg`**: the `stage_blocking` source delegates to `infer_stage_block_intervals()` which requires `sampling_cfg.stage_blocking.enabled: true`. Verify this is present in ds006036 `epoching.datasets.ds006036.sampling.stage_blocking`.

3. **`normalize_payload()` columnar normalization**: the new `block_table_columns` and `block_window_table_columns` are normalized via `_normalize_columnar_mapping()`. String columns (`derived_from`, `end_reason`, `partition_label`) are converted to `str` dtype which h5py writes as variable-length UTF-8. Verify no dtype mismatch errors in the writer.

4. **`manifest_extra` in summary.py**: the `manifest_extra` dict must be in scope at the injection point. Verified at line 4699 of summary.py — `manifest_extra` is built incrementally before the payload write and is in scope.

---

## Methodological claim (pending validation)

**Claim (plausible interpretation):** Block-native window generation eliminates the ambiguity of overlap-fraction labeling at block boundaries. Windows generated inside inferred photic blocks (ds006036) should show a smoother distribution of MNPS coordinates across `relative_pos_0_1` bins compared to the discretized overlap-fraction labels of the current pipeline.

**What would validate this:**
- Run the same ds006036 subject with both `global` mode and `block_native` mode.
- Compare the distribution of `stage_code` across windows in each mode.
- Measure the fraction of windows with `relative_pos_0_1 < 0.05` or `> 0.95` (boundary regions) in each mode — block-native should have none, since windows are generated only inside blocks.
- Compare Jacobian variance at `relative_pos_0_1 ∈ [0, 0.2]` (early) vs `[0.8, 1.0]` (tail).

**Category**: Plausible interpretation. Requires empirical validation on real data.

---

## Next steps after first validation run

1. If `/block_windows/` group appears in H5 and counts are sensible: proceed to MNPS trajectory analysis by `relative_pos_0_1` bin.
2. If `stage_blocking` source fails for ds006036: check that `sampling_cfg` is forwarded correctly from the M6 injection point.
3. If `task_phase` source produces empty blocks for ds003509/ds003506: inspect the actual `event_subtype_role` column values in the events_tsv to confirm prefix matching.
4. Document comparative figures (labeled fraction, boundary contamination, Jacobian variance) as diary entry 096.
