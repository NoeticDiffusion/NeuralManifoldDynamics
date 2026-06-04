# NeuralManifoldDynamics

Monorepo for data ingest, feature extraction, MNPS summarization, and downstream artifact generation for EEG and fMRI workflows.

This root README is intentionally high-level. Package-specific usage, schema details, and command references live in each subproject.

## What This Repo Contains

The repository is organized around a shared pipeline:

1. Acquire or locate source datasets.
2. Index and preprocess raw recordings.
3. Compute per-epoch or per-window features.
4. Project features into MNPS spaces.
5. Write subject-level and run-level outputs for analysis and QC.
6. Optionally build derived event-locked sidecars from annotated events.
7. Optionally add config-driven conventional EEG comparator outputs beside MNPS.

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
- per-subject or per-run subdirectories with `summary.json`, QC JSON, and HDF5 outputs

For EEG overlays with `conventional_eeg.enabled: true`, the feature table can
also include `eeg_conventional_*` comparator columns from the generic `tier1`,
`complexity`, and `connectivity` packs, and summarized outputs expose a
separate `conventional_eeg` block plus `/extensions/conventional_eeg/*` in HDF5.

Typical EEG usage is to enable one or more of:

- `packs: ["tier1"]` for relative power, slowing ratios, and peak-frequency comparators
- `packs: ["complexity"]` for spectral entropy, permutation entropy, and Hjorth comparators
- `packs: ["connectivity"]` for summary synchrony metrics from configured EEG channel pairs

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

MNDM 2.1 outputs can also expose explicit coordinate contracts:

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

## Development Notes

- `requirements.txt` is shared from the repo root.
- `pyarrow` is recommended so feature tables can use parquet cleanly.
- Worker count and memory budget are controlled from the CLI, especially for `mndm.cli features` and `mndm.cli all`.


## Read the docs

https://neuralmanifolddynamics.readthedocs.io/en/latest/index.html


## License

Se LICENSE in the root folder.
