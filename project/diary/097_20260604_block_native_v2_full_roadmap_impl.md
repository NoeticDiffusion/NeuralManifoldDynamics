# 097 20260604 block_native_v2_full_roadmap_impl

## Context

Implemented the full non-breaking Block-Native v2 roadmap in this session:

1. contract and docs hardening,
2. manifest/summary transparency,
3. joinability and richer provenance,
4. built-in sidecar exports + block-native QC,
5. named profile support + convergence planning note,
6. validation on real runs + regressions.

The roadmap plan file itself was not modified.

## What was implemented

### Phase 1: Contract + docs/tests hardening

- Synced docs to canonical H5 block contract:
  - `/blocks/start_sec`, `/blocks/end_sec`, `/blocks/duration_sec`
  - group attr `_schema_version` (not `schema_version`)
  - explicit `derived_from`, `end_reason`, plus added provenance fields
  - `/block_windows/source_window_index`
- Updated docs:
  - `README.md` (root)
  - `mndm/README.md`
  - `mndm/Output_variables_guide.md`
  - config templates (`config_template.yaml`, `eeg_config_ingest_template.yaml`)
- Added/extended regression coverage:
  - `mndm/tests/test_writers.py` now asserts canonical `/blocks` + `/block_windows` names and `_schema_version`.
  - New `mndm/tests/test_block_native_export.py`.
  - Extended `mndm/tests/test_block_windows.py` for named profile parsing.

### Phase 2: Manifest + summary transparency

- `summary.py` now runs block-native injection before `build_manifest(...)` so block-native sections are always reflected in written `summary.json`.
- Added explicit coordinate contract visibility into per-subject summary:
  - `coordinate_contracts.requested_contracts`
  - `coordinate_contracts.realized_contracts`
  - `coordinate_contracts.skipped_contracts_with_reason`
- Added explicit block-native coverage blocks in summary:
  - `block_native`
  - `block_native_coverage`
  - `block_native_qc` (per subject)
- `run_manifest.py` now reports:
  - `capabilities.coordinate_contracts.requested_contracts`
  - `capabilities.coordinate_contracts.realized_contracts`
  - `capabilities.coordinate_contracts.skipped_contracts_with_reason`

### Phase 3: Joinability + provenance

- Added source-window back-reference:
  - `/block_windows/source_window_index` (nearest source MNPS window index, `-1` if no robust match).
- Expanded `/blocks` provenance columns:
  - `frequency_hz`
  - `source_event_idx`
  - `support_event_count`
  - `membership_mode`
  - `bridge_tail_sec`
  - `bridge_tail_cap_sec`
  - `is_inferred`
- Sidecar rows include:
  - `source_window_index`
  - `frequency_hz`
  - `derived_from`
  - `end_reason`

### Phase 4: Built-in exporters + QC

- Wired block-native sidecar writes directly into injection path:
  - `block_native_windows.parquet`
  - `block_native_windows.csv`
- Added run-level sidecar:
  - `block_native_qc.json`
- Added per-subject QC aggregation with:
  - stage/frequency block counts
  - windows per stage
  - duration and gap distributions
  - end-reason counts
  - source-window match fraction
- Added label cleaning reporting in QC:
  - raw label counts
  - cleaned label counts
  - normalized label counts
  - mapping status counts

### Phase 5: Named profiles + convergence planning

- Added named profile aliases in `block_native_config.py`:
  - `whole_block`, `early_block`, `mid_block`, `late_block`,
  - `last5`, `tail8`, `post_offset_0_8`, `post_offset_8_16`
- Added parser support for `window_profile.profile` / `named_profile` / `preset`.
- Updated dataset configs to use named profile alias (`whole_block`) where relevant:
  - `config_ingest_ds006036.yaml`
  - `config_ingest_ds003490.yaml`
  - `config_ingest_ds003509.yaml`
  - `config_ingest_ds003506.yaml`
- Added convergence planning note:
  - `project/block-native_window/event_locked_block_native_convergence.md`

## Validation and regression checks

### Unit/regression tests

Ran:

```powershell
python -m pytest mndm/tests/test_block_windows.py mndm/tests/test_block_native_export.py mndm/tests/test_writers.py mndm/tests/test_run_manifest.py mndm/tests/test_sleep_stage_labels.py
```

Result: **53 passed**.

### Real data validation

#### ds006036 full-cohort run

Command:

```powershell
python -m mndm.cli summarize --dataset ds006036 --config mndm/config/config_ingest_ds006036.yaml --n-jobs 1
```

Result:

- completed successfully (`exit_code: 0`)
- run dir: `E:\Science_Datasets\openneuro\processed\ds006036\neuralmanifolddynamics_ds006036_20260604_124842`
- wrote:
  - per-subject `block_native_windows.parquet/.csv`
  - per-subject `summary.json` with `block_native`, `block_native_coverage`, `block_native_qc`
  - run-level `block_native_qc.json`
  - run-level `run_manifest.json` with explicit `capabilities.coordinate_contracts`
- run manifest counts:
  - `h5 = 88`
  - `h5_with_block_native_windows = 88`

#### Cross-dataset smoke runs (single-subject)

Commands:

```powershell
python -m mndm.cli summarize --dataset ds003490 --config mndm/config/config_ingest_ds003490.yaml --subject 001 --n-jobs 1
python -m mndm.cli summarize --dataset ds003509 --config mndm/config/config_ingest_ds003509.yaml --subject 002 --n-jobs 1
python -m mndm.cli summarize --dataset ds003506 --config mndm/config/config_ingest_ds003506.yaml --subject 002 --n-jobs 1
```

Results:

- all commands completed successfully (`exit_code: 0`)
- ds003509 and ds003506 emitted block-native windows and run-level `block_native_qc.json`
- ds003490 (subject 001 sample) emitted `block_native.status = "no_blocks"` for the sampled recordings, and run-level `block_native_qc` remained `status: "none"` (expected for this filtered smoke when no blocks are inferred)

### Contract spot checks on generated H5

Verified on generated files:

- `/blocks` contains canonical timing names (`start_sec`, `end_sec`, `duration_sec`)
- `/blocks` includes `frequency_hz` and richer provenance columns
- `/block_windows` includes `source_window_index`
- `/blocks.attrs._schema_version == "block_native_v1"`
- `/block_windows.attrs._schema_version == "block_native_v1"`

## Notes

- All changes kept additive/non-breaking for existing global analysis mode and legacy event-locked outputs.
- Cohort-anchored coordinate contract remained requested-but-skipped in these runs due no active external anchor; this is now explicitly visible in run-manifest contract reporting.
