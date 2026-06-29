# NeuralManifoldDynamics

Monorepo for data ingest, feature extraction, MNPS summarization, and downstream artifact generation for EEG and fMRI workflows.

Current documented release line: **NeuralManifoldDynamics 2.3**.

Version 2.3 keeps the versioned measurement-contract framing introduced in the
2.1 coordinate-contract release, and extends it with the **Embodied Anchoring
Principle**: additive embodied/interoceptive exports aligned to the canonical
MNPS time grid (`anchor_state`, `anchor_quality`, optional `anchor_coupling`,
and HRV-oriented raw feature surfaces) without redefining the canonical
`mnps_3d = [m,d,e]` chart.

This root README is intentionally high-level. Package-specific usage, schema details, and command references live in each subproject.

## What This Repo Contains

The repository is organized around a shared pipeline:

1. Acquire or locate source datasets.
2. Index and preprocess raw recordings.
3. Compute per-epoch or per-window features.
4. Project features into MNPS spaces.
5. Write subject-level and run-level outputs for analysis and QC.
6. Optionally export additive embodied/interoceptive surfaces beside MNPS (`anchor_state`, `anchor_quality`, optional `anchor_coupling`, and raw HRV-oriented feature families).
7. Optionally build derived event-locked sidecars from annotated or derived events.
8. Optionally generate block-native analysis windows aligned to inferred or derived temporal blocks (relative position within block, distance to block end, canonical block timing fields `start_sec/end_sec/duration_sec`, and per-window join key `source_window_index` exported to HDF5 `/blocks/` and `/block_windows/` groups as well as Parquet/CSV sidecars).
9. Optionally add config-driven conventional EEG comparator outputs beside MNPS.

## Main Packages

- `mndm`: Core MNPS pipeline. Handles `features`, `summarize`, `all`, `pack`, and structure validation. See `mndm/README.md`.
- `openneuro_ingest`: OpenNeuro-facing download and ingest utilities. Use this when pulling public datasets before MNDM processing.
- `dandi_ingest`: DANDI-facing listing, manifest, download, and NWB probing utilities for DANDI archive assets.
- `apollo_ingest`: Ingest helpers for Apollo-style sources used in this repo.
- `vitaldb_ingest`: Ingest helpers for VitalDB-style sources used in this repo.
- `core`: Shared config loading, path resolution, I/O helpers, and common utilities used across packages.

## Typical Workflow

For most projects, the workflow is:

```text
download or locate data -> ingest/index -> mndm features -> mndm summarize
```

If the dataset is already present on disk, you usually work directly with `mndm`.

## Quickstart Notebook

The fastest way to see the pipeline in action is to open `quickstart.ipynb` at the repo root.  
It runs entirely on **synthetic EEG** — no dataset download needed — and produces an interactive  
3D MNPS trajectory plot in under 5 minutes from a clean clone:

```text
Raw EEG (synthetic, 5 min)  →  band-power features  →  project_features()
  →  estimate_derivatives()  →  build_knn_indices()  →  estimate_local_jacobians()
  →  interactive 3D [m, d, e] trajectory
```

```powershell
jupyter lab quickstart.ipynb
```

---

## Quick Start

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

If you run directly from this source tree, set `PYTHONPATH` so the package modules resolve correctly:

```powershell
$repo_root="C:/path/to/NeuralManifoldDynamics"
$env:PYTHONPATH="$repo_root/mndm/src;$repo_root/core/src;$repo_root/openneuro_ingest/src;$repo_root/apollo_ingest/src;$repo_root/vitaldb_ingest/src;$repo_root"
```

Example MNDM run:

```powershell
python -m mndm.cli all --dataset ds003490 --config mndm/config/config_ingest_ds003490.yaml --n-jobs 12
```

Bootstrap a new dataset overlay from the generic template:

```powershell
copy mndm/config/config_template.yaml mndm/config/config_ingest_my_dataset.yaml
```

Example DANDI manifest/probe flow:

```powershell
python -m dandi_ingest.cli list --config dandi_ingest/configs/dandi_000718.yaml
python -m dandi_ingest.cli probe --config dandi_ingest/configs/dandi_000718.yaml
```

Example event-locked profile run for the sleep-spindle track:

```powershell
python -m mndm.cli all --dataset ds005555 --config mndm/config/config_ingest_ds005555_sleep_spindles.yaml --n-jobs 12
```

The sleep-spindle configuration keeps the canonical HDF5 measurement output separate from derived event-locked sidecars. Spindle annotations, event-window alignment, matched controls, and baseline-corrected summaries should be treated as downstream analysis artifacts with their own provenance.

The same sidecar layer can now also synthesize generic block-end point-events from
`epoching.datasets.<id>.sampling.stage_blocking`. This makes it possible to align
MNPS windows to inferred stimulation-block ends, for example to study early
post-photic effects in `ds006036`, without adding a new HDF5/summarize contract.

For task datasets such as `ds003838`, the repository now also supports
embodied/task-aware runs where:

- within-run task-state labels are written on the MNPS grid,
- event-locked sidecars can be derived from those task-state segments,
- block-native windows can be derived from the same task-state segments, and
- raw HRV v0.1 columns (`ecg_hrv_*`) can appear directly in block-native sidecars.

## Where To Read Next

- MNDM usage and output contracts: `mndm/README.md`
- MNDM generic config template: `mndm/config/config_template.yaml`
- MNDM command reference: `mndm/Command_cheat_sheet.md`
- MNDM output schema details: `mndm/Output_variables_guide.md`
- OpenNeuro ingest details: `openneuro_ingest/`
- DANDI ingest usage/configs/adapters: `dandi_ingest/README.md`

## Repository Layout

```text
NeuralManifoldDynamics/
├── core/
├── mndm/
├── openneuro_ingest/
├── dandi_ingest/
├── apollo_ingest/
├── vitaldb_ingest/
├── requirements.txt
└── README.md
```

## Outputs At A Glance

Most processed outputs are written under a dataset-specific processed directory. In current MNDM runs, summarized outputs typically appear in run folders named like:

```text
<processed>/<dataset>/neuralmanifolddynamics_<dataset>_<timestamp>/
```

Those runs usually contain:

- `run_manifest.json`
- `features_snapshot.json`
- `normalization_report.json` (when normalization is configured)
- `run_errors.json` (when any grouping/subject run failed)
- `stage_mapping_qc.json` (when event-based stage mapping QC is available)
- `block_native_qc.json` (when block-native windows are enabled and emitted)
- per-subject or per-run subdirectories with `summary.json`, QC JSON, and HDF5 outputs

Subject-level summarize outputs now also carry an always-on mathematical validity contract for geometry:

- `summary.json.geometry_contract`
- `qc_summary.json.geometry_contract`
- HDF5 `/provenance/geometry_contract/*`

This contract is separate from reviewer-facing QA. When MNPS rows or Jacobian windows are mathematically unusable, MNDM drops those rows/windows from the canonical export and records counts/reasons in `geometry_contract` instead of silently clamping values. Downstream analyses that interpret MNPS, `coords_9d`, Jacobians, or reachability-style summaries should check this block before treating a subject/run as fully valid.

MNDM 2.3 also adds an embodied anchoring surface that remains separate from both
the canonical coordinates and the cohort/external `feature_anchors` contract:

- `/anchor_state/*` and `/anchor_state_dot/*` for time-aligned embodied state
- `/anchor_quality/*` for modality support and quality
- optional `/anchor_coupling/*` for additive body-brain coupling diagnostics
- `summary.json.anchor_hrv_v0_1` when HRV v0.1 raw features are enabled

These are intended as additive analysis surfaces, not as a fourth canonical
MNPS axis and not as a replacement for the anchored coordinate contracts.

For EEG overlays with `conventional_eeg.enabled: true`, the feature table can
also include `eeg_conventional_*` comparator columns from the generic `tier1`,
`complexity`, `connectivity`, and `coma` packs, and summarized outputs expose a
separate `conventional_eeg` block plus `/extensions/conventional_eeg/*` in HDF5.

Typical EEG usage is to enable one or more of:

- `packs: ["tier1"]` for relative power, slowing ratios, and peak-frequency comparators
- `packs: ["complexity"]` for spectral entropy, permutation entropy, and Hjorth comparators
- `packs: ["connectivity"]` for summary synchrony metrics from configured EEG channel pairs
- `packs: ["coma"]` for EEG-only ICU/coma proxies (`suppression_ratio`,
  `burst_suppression_proxy`, `continuity_proxy`, `alpha_delta_ratio`,
  `reactivity_proxy`)

When the `coma` pack is enabled without external clinical sidecars, summarize
also records explicit `unavailable` status for multimodal coma biomarkers
(`SSEP`, `NSE`, `GCS`, `S100B`) so reviewer-facing outputs make the data boundary explicit.

Connectivity outputs currently follow names such as:

- `eeg_conventional_connectivity_alpha_FP_plv_mean`
- `eeg_conventional_connectivity_alpha_FB_coh_mean`

### Normalization and Batch Harmonization (ComBat pilot)

Recent MNDM runs can optionally apply feature-level ComBat harmonization in `summarize`
(`normalization.enabled: true`, `method: combat`, `scope: post_features`).

For multi-site datasets (for example I-CARE hospitals), this targets hardware/site batch
effects while preserving the standard MNPS output contract. Runtime provenance is written
to:

- `run_manifest.json` under `extra.normalization`
- `features_snapshot.json` under `normalization`

Current runtime implementation supports:

- chunked ComBat fitting (`normalization.combat.chunk_size`) for large feature tables
- batch assignment via participant metadata (`batch_key`, e.g. `hospital`)
- optional covariates (`covariates`, e.g. `age`, `sex`)
- lightweight Layer-0 style outlier damping via winsorization
  (`normalization.combat.winsorize_quantiles`)

Layer-0 context (invariance before heavy harmonization): average reference, spectral ratios,
slope/entropy-style scale-invariant families, and regional ensemble averaging are treated as
complementary design levers rather than replacements for explicit batch modeling.

For WFDB overlays with `time_reference.enabled: true`, H5 outputs also include:

- `/extensions/time_reference/run/*`
- `/extensions/time_reference/windows/*`

and `run_manifest.json` reports capability flags for time-reference presence.

MNDM 2.3 outputs retain the explicit coordinate contracts introduced in the
2.1 release line:

- `/coords_3d_subject_anchored` and `/coords_9d_subject_anchored` for within-subject geometry
- `/coords_3d_cohort_anchored` and `/coords_9d_cohort_anchored` for cohort/external-anchored clinical group comparisons
- `/feature_anchors` for the frozen anchor provenance and per-feature center/scale statistics

`run_manifest.json` reports these as capability flags so downstream analyses can choose the declared `primary_coordinate_layer`.

Some datasets also carry labels that vary within a single run instead of staying constant for the whole recording. MNDM now supports these as time-aligned labels on the MNPS axis, for example:

- `ds006623`: keeps run identity such as `task=imagery` but writes within-run anesthesia state as `pre_lor`, `unresponsive`, `post_ror`
- sleep datasets: can write changing sleep stages within one recording instead of forcing them into one scalar run condition

These labels are written alongside the MNPS trajectory rather than replacing run-level metadata.

### Event Provenance and Stage Blocking

MNDM summarize now supports a generic event-to-stage provenance surface and optional
continuous block labeling driven by YAML policy (not dataset-specific hardcoding).

Use dataset overrides under `epoching.datasets.<dataset_id>.sampling`:

```yaml
epoching:
  datasets:
    my_dataset:
      sampling:
        stage_columns: ["value"]
        prefer_events_stage_in_summary: true
        stage_blocking:
          enabled: true
          stage_event_regex: "(?i)^PHOTO\\s*(\\d+)\\s*Hz$"
          bridge_marker_labels: ["Photo/HV mark"]
          use_bridge_markers: true
          bridge_tail_sec: 0.5
          bridge_tail_cap_sec: 1.0
          min_block_sec: 2.0
          max_block_sec: 20.0
          preserve_block_assignments: true
          window_membership:
            # Interval geometry on the MNPS axis, not mnps.overlap.
            mode: "midpoint_in_interval"
            min_overlap_fraction: 0.0
          expected_stage_frequencies_hz: [5, 10, 15, 20, 25, 30]
```

Run outputs then expose:

- `summary.json.stage_mapping_qc` (per-subject mapping diagnostics)
- `summary.json.event_provenance` (auditable source/mapping contract)
- run-level `stage_mapping_qc.json` + `run_manifest.json -> extra.stage_mapping_qc`
- H5 `/events/*` columnar event provenance table when available

Window-membership options:

- `midpoint_in_interval` (default): historical behavior, good when you want a
  generous block surface.
- `fully_contained`: only MNPS windows fully inside the inferred block are
  labeled.
- `overlap_frac_ge`: require a configurable overlap fraction between each MNPS
  window and the inferred block.

Practical validation on `ds006036`:

- switching from midpoint-based photic labeling to
  `window_membership.mode: "fully_contained"` kept `labels_stage: true` for all
  88 H5 outputs,
- mean `stage_frac_labeled` changed from `0.584536` to `0.533997` (`-8.65%`),
- photic-labeled windows changed from `1856` to `979` (`-47.25%`).

This is a useful “clean photic epochs” setting when you want to suppress
boundary contamination rather than maximize photic window count.

Derived event-locked outputs, such as YASA-derived sleep-spindle sidecars for `ds005555` or derived `stage_block_end` sidecars for `ds006036`, are intentionally not part of the canonical HDF5 measurement surface unless a future release promotes a stable derived-event schema. They are joinable back to HDF5 trajectories by subject and window identifiers, and exported rows use generic `condition` labels (`event`, `matched_control`) rather than spindle-specific names.

### Block-Native Window Analysis

Block-native analysis generates analysis windows directly from inferred temporal blocks rather than labeling a pre-existing global epoch grid. Windows carry explicit relative-position metadata: `relative_time_in_block_sec`, `distance_to_block_end_sec`, and `relative_pos_0_1`.

Enable it per-dataset under `block_native.datasets.<id>`:

```yaml
block_native:
  datasets:
    ds006036:
      enabled: true
      source:
        kind: "stage_blocking"   # or "duration_events" / "task_phase"
        label_column: "value"
        onset_column: "onset"
        duration_column: "duration"
      window_profile:
        kind: "sliding"          # or "tail" / "post_offset" / "partitioned"
        window_length_sec: 4.0
        step_sec: 2.0
        emit_relative_position: true
        min_block_sec: 4.0
        min_windows_per_block: 2
      export:
        write_parquet: true
        write_csv: true
```

Block source kinds:

- `stage_blocking`: reuses the existing `stage_blocking` event-regex infrastructure (suitable for `ds006036` photic stimulation)
- `duration_events`: infers blocks from TSV event labels with explicit durations (suitable for resting-state eyes-open/closed designs)
- `task_phase`: groups consecutive events that share a common label prefix (suitable for trial-phase designs)

In the current `ds003838` embodied-anchoring path, block-native windows can also
be driven from derived task-state segments rather than raw one-second digit
events. This is the preferred surface for sustained HRV analyses, while
`event_locked` remains the better fit for onset/offset or short-event questions.

Run outputs include:

- HDF5 `/blocks/*` columnar table (one row per inferred block)
- HDF5 `/block_windows/*` columnar table (one row per generated window, with full relative-position geometry)
- Parquet/CSV sidecar files in the subject run directory
- `run_manifest.json` capability flag `has_block_native_windows`

See `mndm/README.md` for full YAML reference and `mndm/Output_variables_guide.md` for the complete H5 schema.

## Development Notes

- `requirements.txt` is shared from the repo root.
- `pyarrow` is recommended so feature tables can use parquet cleanly.
- Worker count and memory budget are controlled from the CLI, especially for `mndm.cli features` and `mndm.cli all`.
- OpenNeuro downloads (`openneuro_ingest`) run through `uvx openneuro-py@latest` by default, so the latest released openneuro-py is used in an isolated environment. This avoids breakage from installed openneuro-py versions whose GraphQL metadata query is incompatible with the current OpenNeuro server schema. `uv` (providing `uvx`) is included in `requirements.txt`. Override with `download.use_uvx: false` in config or the `OPENNEURO_PREFER_UVX=0` env var. See `openneuro_ingest/Command_cheat_sheet.md`.


## Read the docs

https://neuralmanifolddynamics.readthedocs.io/en/latest/index.html


## License

Se LICENSE in the root folder.
