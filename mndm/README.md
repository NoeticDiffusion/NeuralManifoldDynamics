# MNDM (Meta-Noetic Diffusion Model)

MNPS computation pipeline that computes per-epoch features and produces MNPS summaries, Jacobians, and derived outputs.

Note: OpenNeuro ingest/download now lives in `openneuro_ingest`. This package covers feature extraction, summarization, packing, and structure checks.

## Overview

This toolkit transforms raw EEG and fMRI data into analysis-ready MNPS trajectories with associated Jacobian meta-dynamics, supporting the Noetic Diffusion Theory framework.

### Pipeline Stages

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Download   │ → │ Preprocess  │ → │  Features   │ → │  Summarize  │
│  + Index    │    │  + Filter   │    │  per-epoch  │    │ MNPS + J_hat│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Key Capabilities

- **Multimodal feature extraction**: EEG (band power, entropy, connectivity), fMRI (regional BOLD, modularity), and peripheral signals (ECG, RESP, EDA, EOG, EMG)
- **MNPS projection**: Maps features to 3D coordinates (m, d, e) representing mobility, diffusivity, and entropy
- **Stratified MNPS**: Optional 9D subcoordinate chart (m_a, m_e, m_o, d_n, d_l, d_s, e_e, e_s, e_m) for mechanistic decomposition
- **Jacobian estimation**: Local linear approximations of MNPS dynamics with meta-indices (trace, rotation, anisotropy)
- **MNPS extensions**: E-Kappa (energetic curvature), RFM (resonant frequency modes), O-Koh (organizational coherence), TIG (temporal integrity grade)
- **Robustness**: Ensemble variance, split-half reliability, PSD multiverse stability, entropy sanity checks
- **MNDM 2.1 anchored coordinates**: explicit subject/session-anchored and cohort-anchored coordinate layers, with versioned feature anchors for clinical group comparisons
- **Resume-friendly**: Interrupted runs can continue from existing artifacts
- **Optional conventional EEG comparator layer**: YAML-driven Tier 1 qEEG outputs (relative power, slowing ratios, alpha peak, median frequency, spectral edge) beside MNPS
- **Optional extras (recent)**:
  - FD censoring of high-motion epochs (framewise_displacement > 0.5 mm, ±1 neighbour)
  - Provisional flag for fMRI modularity when a window has very few volumes
  - Event→MNPS mapping (opt-in) writing binary labels aligned to MNPS time
  - Generic event provenance export (`/events` columnar table + `summary.json.event_provenance`)
  - Config-driven stage-block inference for sparse marker streams (`stage_blocking` policy)
  - Generic event annotation tables, event-window alignment, matched controls, and event-locked sidecar exports for downstream analyses
  - Within-run state labeling for datasets where labels change inside one run, with support for boundary tables, interval tables, and feature-derived stage columns
  - Window start/end (seconds) per MNPS point in HDF5 for clearer time alignment
  - Time Reference v1 for WFDB datasets (clock provenance + anchor-aligned windows under `/extensions/time_reference/*`)

---

## Requirements

- Python 3.11+
- Dependencies: `numpy`, `scipy`, `pandas`, `mne`, `h5py`, `pyyaml`, `tqdm`, `joblib`
- Optional: `openneuro-py` (for dataset downloads)
- Optional: `dandi`, `pynwb` (for DANDI archive access and NWB inputs)
- Optional: `yasa` (for detector-derived sleep-spindle annotations in downstream event-locked workflows)
- Optional but recommended for feature storage: `pyarrow`

---

## Installation

```powershell
# Clone and enter the repository root
cd NoeticDiffusionDataIngest

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

If you run from this source checkout without installing packages editably, set `PYTHONPATH` before invoking `mndm`:

```powershell
$env:PYTHONPATH="H:/SourceRepo2/NeuralManifoldDynamics/mndm/src;H:/SourceRepo2/NeuralManifoldDynamics/core/src;H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/apollo_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/vitaldb_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics"
```

---

## Quick Start

### Step-by-Step

```powershell
# Preflight check: validate paths, config, participants table, and index preview.
python -m mndm.cli prerequisite-check --dataset ds003490

# Download (ingest)
python -m openneuro.cli download --dataset ds003490

# Or list/probe DANDI NWB assets before pointing an MNDM NWB config at the local raw root.
# python -m dandi_ingest.cli list --config dandi_ingest/configs/dandi_000718.yaml
# python -m dandi_ingest.cli probe --config dandi_ingest/configs/dandi_000718.yaml

# Compute per-epoch features (mndm)
python -m mndm.cli features --dataset ds003490

# Project to MNPS and estimate Jacobians
python -m mndm.cli summarize --dataset ds003490

# Or run both in one step:
# python -m mndm.cli all --dataset ds003490

# (Optional) Pack a completed MNPS run (many small H5) into one container H5
# Output: <processed>/<dataset>/<latest mnps_*>/packed.h5
python -m mndm.cli pack --dataset ds003490

# (Optional, MNDM 2.1) Smoke-test cohort anchoring on hard-to-separate EEG cohorts
python -m mndm.cli anchor-smoke --dataset ds003478 ds003944 ds004504 --data-dir M:/datasets/received/openneuro --h5-root <processed_dir>
```

See [Command_cheat_sheet.md](Command_cheat_sheet.md) for complete CLI reference.

---

### Summarize MNPS

```powershell
python -m mndm.cli summarize --dataset ds003490
```

### Run Full MNDM Pipeline

```powershell
python -m mndm.cli all --dataset ds003490 --n-jobs 12
```


## Project Structure

```
mndm/
├── config/
│   └── config_ingest.yaml        # Pipeline configuration
├── src/mndm/
│   ├── cli.py                    # Command-line interface
│   ├── orchestrate.py            # Pipeline orchestration
│   ├── projection.py             # Feature → MNPS mapping
│   ├── jacobian.py               # Local Jacobian estimation
│   ├── extensions.py             # E-Kappa, RFM, O-Koh, TIG
│   ├── robustness.py             # Reliability metrics
│   ├── pipeline/                 # Summarization pipeline
│   │   ├── summary.py            # Runner classes
│   │   ├── context.py            # Configuration resolution
│   │   ├── extensions_compute.py # Extension computation
│   │   ├── robustness_helpers.py # QC summaries
│   │   ├── extractors.py         # Data extraction utilities
│   │   └── regions.py            # Network mapping
│   └── tools/                    # Utilities (pack, aggregate)
└── tests/                        # Test suite
```

---

## Configuration

Edit `config/config_ingest.yaml` or a dataset overlay such as `config/config_ingest_ds004511.yaml` to customize the pipeline.

For new datasets, start from:

- `mndm/config/config_template.yaml` (generic template with optional blocks)
- one common base import:
  - EEG: `mndm/config/config_ingest_common_eeg.yaml`
  - NWB: `mndm/config/config_ingest_common_nwb.yaml`
  - NWB rodent: `mndm/config/config_ingest_common_nwb_rodent.yaml`
  - NWB mouse EEG: `mndm/config/config_ingest_common_nwb_mouse_eeg.yaml`

Example workflow:

```powershell
# 1) Copy template
copy mndm/config/config_template.yaml mndm/config/config_ingest_my_dataset.yaml

# 2) Edit dataset id/path/source + needed overrides
# 3) Run
python -m mndm.cli features --dataset my_dataset --config mndm/config/config_ingest_my_dataset.yaml --n-jobs 1
python -m mndm.cli summarize --dataset my_dataset --config mndm/config/config_ingest_my_dataset.yaml --n-jobs 1
```

### Conventional EEG Comparator Packs

For EEG overlays, you can enable a conventional qEEG comparator layer without
changing the MNPS projection weights:

```yaml
conventional_eeg:
  enabled: true
  packs: ["tier1", "complexity", "connectivity"]
  export:
    per_epoch_columns: true
    summaries: true
  tier1:
    relative_bandpower: true
    ratios: ["theta_alpha", "delta_alpha", "alpha_theta", "beta_alpha", "slowing_index"]
    peak_frequency:
      alpha_peak_frequency: true
      median_frequency: true
      spectral_edge_95: true
  complexity:
    spectral_entropy: true
    permutation_entropy: true
    hjorth_complexity: true
    hjorth_mobility: false
  connectivity:
    roi_pairs:
      - {name: FP, channels: ["F3", "P3"]}
      - {name: FB, channels: ["Fz", "POz"]}
    metrics:
      coherence: true
      plv: true
      pli: true
      wpli: true
    outputs:
      summary_stats: ["mean", "std"]
```

For Parkinson-focused EEG analyses, the most common starting subset is usually
`tier1` plus `complexity`, rather than full connectivity. In practice this means:

- relative bandpower in `delta`, `theta`, `alpha`, and `beta`
- slowing-sensitive ratios such as `theta_alpha`, `delta_alpha`, `beta_alpha`, and `slowing_index`
- spectral peak / edge markers such as `alpha_peak_frequency`, `median_frequency`, and `spectral_edge_95`
- complexity markers such as `spectral_entropy`, `permutation_entropy`, `hjorth_complexity`, and `hjorth_mobility`

That is the default Parkinson-facing comparator profile now enabled for
`ds003490`, while connectivity can still be added explicitly for follow-up
network analyses.

This layer is intentionally separate from the MNPS feature contract:

- it does **not** change `mnps_projection.weights`
- it does **not** change `mnps_9d` mappings
- it does add comparator-friendly `eeg_conventional_*` columns to the feature table

Summarize writes the comparator rollups under:

- `summary.json.conventional_eeg`
- `h5.attrs["manifest"] -> conventional_eeg`
- `/extensions/conventional_eeg/*`

Current generic packs:

- `tier1`: relative bandpower, slowing ratios, alpha peak, median frequency, spectral edge
- `complexity`: spectral entropy, permutation entropy, Hjorth complexity, optional Hjorth mobility
- `connectivity`: summary synchrony features from configured channel pairs, such as coherence, PLV, PLI, wPLI, and optional dPLI/PPC

Common usage patterns:

```yaml
# Tier 1 only
conventional_eeg:
  enabled: true
  packs: ["tier1"]
```

```yaml
# Complexity only
conventional_eeg:
  enabled: true
  packs: ["complexity"]
```

```yaml
# Both packs
conventional_eeg:
  enabled: true
  packs: ["tier1", "complexity", "connectivity"]
```

```yaml
# Connectivity only
conventional_eeg:
  enabled: true
  packs: ["connectivity"]
```

Current naming pattern:

- `eeg_conventional_relative_*`
- `eeg_conventional_ratio_*`
- `eeg_conventional_peak_*`
- `eeg_conventional_complexity_*`
- `eeg_conventional_connectivity_*`

Typical connectivity outputs look like:

- `eeg_conventional_connectivity_alpha_FP_plv_mean`
- `eeg_conventional_connectivity_alpha_FB_coh_mean`

Implementation note:

- `tier1` and `complexity` emit per-epoch comparator columns
- `connectivity` currently emits recording-level synchrony summaries that are
  broadcast across epochs so they can be carried by the same summarize contract

### Time Reference v1 (WFDB clocks)

For WFDB datasets (for example PhysioNet I-CARE), enable:

```yaml
time_reference:
  enabled: true
  schema_version: "time_reference.v1"
  parser: "wfdb_header"
  anchor: "first_recording"
  bins_hours: [0, 24, 48, 72]
  datasets:
    my_dataset:
      enabled: true
      parser: "wfdb_header"
      anchor: "first_recording"
      bins_hours: [0, 24, 48, 72]
```

This keeps canonical `/time`, `/window_start`, `/window_end` unchanged and adds:

- `/extensions/time_reference/run/*`
- `/extensions/time_reference/windows/*`

### Event Provenance and Stage-Blocking Policy

For datasets with sparse event markers (for example onset-only stimulation markers),
you can configure summarize to infer stable per-window stage blocks and emit auditable
event provenance surfaces.

Example YAML under `epoching.datasets.<dataset_id>.sampling`:

```yaml
epoching:
  datasets:
    my_dataset:
      sampling:
        onset_column: "onset"
        duration_column: "duration"
        stage_columns: ["value"]
        # Optional: summarize-only reruns can override stale stage columns
        # in features.csv with fresh event-derived stage.
        prefer_events_stage_in_summary: true
        stage_blocking:
          enabled: true
          stage_event_regex: "(?i)^PHOTO\\s*(\\d+)\\s*Hz$"
          bridge_marker_labels: ["Photo/HV mark"]
          use_bridge_markers: true
          hv_tail_sec: 0.5
          hv_tail_cap_sec: 1.0
          min_block_sec: 2.0
          max_block_sec: 20.0
          preserve_block_assignments: true
          expected_stage_frequencies_hz: [5, 10, 15, 20, 25, 30]
```

Outputs and provenance:

- per-subject `summary.json.stage_mapping_qc`
- per-subject `summary.json.event_provenance`
- run-level `stage_mapping_qc.json`
- `run_manifest.json -> extra.stage_mapping_qc`
- H5 `/events/*` includes both legacy onset arrays and columnar event table when available

Backward compatibility:

- legacy keys remain accepted: `photic_regex`, `hv_mark_labels`, `use_hv_marks`,
  `preserve_photic_blocks`, `expected_frequencies_hz`.

### Normalization (ComBat pilot, post-features)

MNDM summarize can optionally harmonize feature tables before grouping/projection.
Current pilot path is ComBat (`neuroCombat`) in `post_features` scope.

Example config block:

```yaml
normalization:
  enabled: true
  method: "combat"
  scope: "post_features"
  batch_key: "site_or_hospital"      # often "hospital" for I-CARE
  covariates: ["group", "age", "sex"]
  reference_policy: "within_dataset" # currently documented policy surface
  combat:
    chunk_size: 24
    min_batch_size: 2
    min_feature_observations: 16
    winsorize_quantiles: [0.005, 0.995]
  datasets:
    my_dataset:
      enabled: true
      batch_key: "hospital"
```

Runtime notes:

- normalization happens in summarize after QC filtering and before run grouping
- batch/covariate values are resolved from participant metadata tables/sidecars
- chunking limits memory pressure on large datasets
- if covariate fit fails, runtime retries with a batch-only model (non-strict mode)
- normalization provenance is stored in:
  - `run_manifest.json` -> `extra.normalization`
  - `features_snapshot.json` -> `normalization`

Layer-0 invariance context:

- average reference
- ratio features (e.g. alpha/theta, band fractions)
- slope/entropy-style scale-robust features
- regional ensemble averaging

These are useful "free" invariance levers, but they are not statistical harmonization by themselves.

### MNDM 2.1 Anchored Coordinates

MNDM 2.1 separates the coordinate measurement contract from the feature export
contract. The raw feature surface remains the long-lived reanalysis source:

- `/features_raw/*`: empirical feature values in original scale
- `/features_robust_z/*`: strict per-file/session robust-z diagnostic surface

Coordinate layers are now explicit:

- `/coords_3d_subject_anchored/*`: current subject/session-relative 3D geometry; use for within-subject dynamics, local Jacobians, reachability-style summaries, and trajectory shape.
- `/coords_9d_subject_anchored/*`: subject/session-relative stratified coordinates when `mnps_9d` is enabled.
- `/coords_3d_cohort_anchored/*`: externally or cohort-anchored 3D coordinates; use for clinical group comparisons when `mnps_projection.anchor.enabled=true`.
- `/coords_9d_cohort_anchored/*`: cohort-anchored stratified coordinates when both an anchor and `mnps_9d` are available.
- `/feature_anchors/*`: embedded anchor provenance, including `anchor_id`, `anchor_hash`, source policy, and per-feature center/scale statistics.

Example anchor configuration:

```yaml
mnps_projection:
  normalize: "robust_z"
  export_contracts:
    subject_anchored: true
    cohort_anchored: true
  anchor:
    enabled: true
    path: "anchors/ds003944_fep_control_v2_1.json"
    scale_method: "iqr"   # iqr | mad | qn
    min_subjects: 3
```

`export_contracts` lets you choose which anchored H5 surfaces are emitted in a
run. These booleans gate the coordinated export of coordinate layers,
Jacobians, and regional network layers:

- `subject_anchored: true/false`
- `cohort_anchored: true/false`

If omitted, `subject_anchored` defaults to `true`, while `cohort_anchored`
defaults to `true` only when an external or one-shot fitted anchor is active.

Fit and inspect anchors post-hoc from summarized H5 outputs:

```powershell
python -m mndm.cli anchors-fit --h5-root <processed_dir> --dataset ds003944 --config mndm/config/config_ingest_ds003944.yaml --anchor-id ds003944_fep_control_v2_1 --out anchors/ds003944_fep_control_v2_1.json
python -m mndm.cli anchor-smoke --dataset ds003478 ds003944 ds004504 --data-dir M:/datasets/received/openneuro --h5-root <processed_dir> --out anchor_smoke_report.json
python -m mndm.cli anchor-sensitivity --h5-root <processed_dir> --dataset ds003944 --config mndm/config/config_ingest_ds003944.yaml --out anchor_sensitivity_ds003944.json
```

One-shot cohort anchoring is also available directly from merged feature tables,
without a prior subject-anchored summarize pass:

```powershell
python -m mndm.cli summarize --dataset ds003944 --fit-anchor
python -m mndm.cli all --dataset ds003944 --fit-anchor
```

In one-shot mode, summarize:

1. reads `features.parquet` / `features.csv`
2. fits a **subject-balanced frozen anchor artifact**
3. saves it under the run directory (for example `run_dir/anchors/*.json`)
4. applies that frozen anchor during the same summarize pass

This is still a **fit -> freeze -> apply** contract. The anchor is not recomputed
on-the-fly per downstream group comparison.

`run_manifest.json` reports capability flags for these layers, including
`feature_anchors`, `coords_3d_subject_anchored`,
`coords_3d_cohort_anchored`, `coords_9d_subject_anchored`, and
`coords_9d_cohort_anchored`.

### Key Sections

```yaml
# Dataset paths
paths:
  received_dir: "E:/Science_Datasets/openneuro/received"
  processed_dir: "E:/Science_Datasets/openneuro/processed"

# Preprocessing
preprocess:
  eeg_reference: "average"
  eeg_bandpass: [0.5, 45.0]
  resample_hz: 256

# Epoching
epoching:
  length_s: 8.0
  step_s: 4.0

# Feature extraction
features:
  eeg_bands:
    delta: [0.5, 4]
    theta: [4, 8]
    alpha: [8, 12]
    beta: [12, 30]
    gamma: [30, 45]

# MNPS projection weights
mnps_projection:
  weights:
    m: { eeg_theta: 0.5, eeg_alpha: 0.5 }
    d: { eeg_wPLI_theta: 1.0 }
    e: { eeg_sample_entropy: 1.0 }

# Stratified MNPS (optional)
mnps_9d:
  enabled: true
  definition_version: "2.0"
  subcoords:
    m_a: { eeg_alpha: 1.0 }
    m_e: { eeg_theta: 1.0 }
    # ... (9 subcoordinates)

# MNPS extensions
mnps_extensions:
  e_kappa: { enabled: true }
  rfm: { enabled: true, band: "alpha" }
  o_koh: { enabled: true }
  tig: { enabled: true }
```

### fMRI Configuration

```yaml
preprocess:
  fmri:
    atlas_path: "path/to/schaefer_200.nii.gz"
    atlas_labels: "path/to/schaefer_200_labels.txt"
    bandpass: [0.01, 0.1]

features:
  fmri:
    window_sec: 30.0
    step_sec: 15.0
```

### EEG CSD Note

When `preprocess.eeg_csd.enabled=true`, scalp EEG channels transformed by
`mne.compute_current_source_density(...)` are still exported downstream as the
`"eeg"` modality for feature extraction. This prevents CSD-transformed scalp
recordings from disappearing at the modality collection stage simply because
MNE relabels their channel type from `eeg` to `csd`.

### Event-Locked Analysis Sidecars

MNDM now includes generic helpers for event-locked analyses:

- `mndm.pipeline.event_annotations`: load and serialize event tables such as detector-derived sleep spindles
- `mndm.pipeline.event_alignment`: map events to MNPS windows and relative-time bins
- `mndm.pipeline.control_matching`: select matched non-event control windows
- `mndm.pipeline.event_locked_export`: write flat analysis tables with provenance

These helpers are designed as a derived analysis layer. The canonical HDF5 output remains the measurement surface (`/mnps_3d`, `/coords_9d`, explicit MNDM 2.1 coordinate layers, `/jacobian`, `/labels`, feature surfaces, and provenance). Event annotations, event-window mappings, matched controls, and baseline-corrected summaries should be kept as sidecars or clearly marked derived groups unless a future release promotes a stable derived-event schema.

For the sleep-spindle profile, use the event-locked overlay rather than replacing the standard sleep-stage configuration:

```powershell
python -m mndm.cli all --dataset ds005555 --config mndm/config/config_ingest_ds005555_sleep_spindles.yaml --n-jobs 12
```

This overlay uses short windows suitable for spindle-scale event alignment while preserving stage labels as time-aligned annotations.

### Within-Run Labels

Some datasets need two different label layers:

- run-level metadata such as `task`, `run`, and a stable `condition`
- time-varying labels that change within the same run

MNDM supports this through `within_run_labels`, which maps external timing/state information onto the MNPS time axis and writes the result to `payload.stage` and/or `payload.labels[...]`.

Supported source types:

- `boundary_table`: transition points such as LOR/ROR
- `interval_table`: start/stop/label intervals such as sleep scoring windows
- `column_from_features`: stage columns that already exist per epoch in `features.csv` or `features.parquet`

Example config shape:

```yaml
within_run_labels:
  datasets:
    dsExample:
      enabled: true
      output_name: "within_run_state_v1"
      write_to_stage: true
      write_to_labels: true
      codebook:
        wake: 0
        n2: 2
        rem: 4
      rules:
        - id: "sleep_intervals"
          match:
            task: "sleep"
            run: "run-1"
          source:
            type: "interval_table"
            path: "labels/sleep_intervals.csv"
            subject_column: "subject"
            start_column: "start"
            end_column: "end"
            label_column: "label"
            units: "seconds"
```

Dataset-specific logic should live in config, not in core summary logic.

### `ds006623` Special Case

`ds006623` is a concrete example of within-run state labeling:

- run identity remains `task=imagery`
- `run-2` is split at `LOR time (TR in task2)` into `pre_lor` and `unresponsive`
- `run-3` is split at `ROR time (TR in task3)` into `unresponsive` and `post_ror`
- `ROR=N/A` keeps the run labeled `unresponsive` until run end

The dataset overlay `config/config_ingest_ds006623.yaml` now points to:

- `G:/Science_Datasets_longtime_storage/ds006623/LOR_ROR_Timing.csv`
- `G:/Science_Datasets_longtime_storage/ds006623/Participant_Info.csv`

This keeps run-level labels such as `imagery` stable while exposing the clinically relevant state sequence on the MNPS axis.

---

## Output Format

### Directory Structure

```
<processed_dir>/<dataset_id>/
├── file_index.csv                # Indexed BIDS files
├── features.csv / features.parquet
└── neuralmanifolddynamics_<dataset>_<timestamp>/
    ├── features_snapshot.json    # Feature snapshot for the summarized run
    ├── run_manifest.json         # Run-level capabilities and field guide
    ├── normalization_report.json # Present when normalization policy is active
    ├── run_errors.json           # Present when any run/grouping error is captured
    ├── stage_mapping_qc.json     # Present when event-based stage QC is emitted
    └── sub-XXX_<suffix>/
        ├── summary.json          # MNPS manifest
        ├── qc_reliability.json   # Split-half metrics
        ├── qc_summary.json       # Coverage and QC flags
        └── sub-XXX_<suffix>.h5   # MNPS tensors
```

### Session-aware outputs (ds003059)

`ds003059` contains two sessions per subject (`ses-LSD` and `ses-PLCB`) with identical task/run labels. The pipeline maps **BIDS session → H5 `condition`** to prevent overwrites, so you should expect outputs like:

- `sub-001_PLCB_rest_run-01/sub-001_PLCB_rest_run-01.h5`
- `sub-001_LSD_rest_run-01/sub-001_LSD_rest_run-01.h5`

### HDF5 Schema

Canonical regional outputs are written under `/regional_mnps` for both EEG and fMRI.
The `/regions/*` group is optional supporting input data, mainly for raw fMRI regional signals.

| Path | Shape | Description |
|------|-------|-------------|
| `/time` | (T,) | Time index (seconds) |
| `/window_start` | (T,) | Window start times (seconds) |
| `/window_end` | (T,) | Window end times (seconds) |
| `/mnps_3d` | (T, 3) | MNPS coordinates [m, d, e] |
| `/mnps_3d_dot` | (T, 3) | MNPS time derivatives |
| `/features_raw/values` | (T, K) | Raw feature matrix in original scale |
| `/features_raw/names` | (K,) | Feature names aligned to raw values |
| `/features_raw/metadata/*` | (K,) per field | Machine-readable feature provenance and usage flags |
| `/features_robust_z/values` | (T, K) | Strict robust-z feature matrix |
| `/features_robust_z/names` | (K,) | Feature names aligned to strict robust-z values |
| `/features_robust_z/metadata/*` | (K,) per field | Same feature metadata layout as `/features_raw/metadata/*` |
| `/coords_9d/values` | (T, 9) | Stratified subcoordinates |
| `/coords_3d_subject_anchored/values` | (T, 3) | MNDM 2.1 subject/session-anchored 3D coordinates |
| `/coords_3d_subject_anchored/names` | (3,) | Coordinate names `[m, d, e]` |
| `/coords_9d_subject_anchored/values` | (T, 9) | Subject/session-anchored stratified coordinates |
| `/coords_3d_cohort_anchored/values` | (T, 3) | Cohort/external-anchored 3D coordinates when an anchor is configured |
| `/coords_9d_cohort_anchored/values` | (T, 9) | Cohort/external-anchored stratified coordinates when available |
| `/feature_anchors/spec` | attrs | Anchor identity, source policy, scale method, and hash |
| `/feature_anchors/per_feature/*` | (K,) per field | Per-feature anchor center, scale, quantiles, MAD/IQR/Qn, and support counts |
| `/jacobian/J_hat` | (W, 3, 3) | Local Jacobian matrices |
| `/jacobian/J_dot` | (W, 3, 3) | Jacobian time derivatives |
| `/jacobian/centers` | (W,) | Window center indices |
| `/nn/indices` | (T, k) | kNN neighbor indices |
| `/labels/stage` | (T,) | Canonical integer-coded per-window state series, e.g. sleep stages or within-run anesthesia state |
| `/labels/<name>` | (T,) | Additional aligned labels, binary, numeric, or categorical |
| `/events/*` | varies | Legacy event arrays plus optional columnar event provenance (`onset_sec`, `duration_sec`, `raw_event_label`, `mapped_stage_code`, `mapping_mode`, etc.) |
| `/event_windows/*` | (R,) per field | Additive explicit event-to-window join contract for event-locked downstream analysis (`event_id`, `window_id`, `event_label`, onset/window bounds, bin labels, exact containing windows) |
| `/codebooks/stage/*` | (C,) per field | Explicit stage codebook (`codes`, `labels`, `label_keys`) for self-describing H5 labels |
| `/participant/clinical_json` | scalar JSON | Additive richer participant/session metadata embedded into H5 |
| `/coverage/*` | varies | Explicit layer coverage contract, including per-window direct-axis coverage and time-index mappings |
| `/provenance/*` | varies | Structured export-contract, anchoring, normalization, and event/stage provenance blocks |
| `/qc/windows/*` | (T,) per field | Light-weight per-window QC contract (`retained_after_qc`, `coverage_ok`, stage transitions, optional `qc_ok_*`) |
| `/regional_mnps/<network>/mnps` | (Tr, 3) | Canonical regional MNPS output for any modality |
| `/regions/bold` | (R, T') | Optional raw regional fMRI time series |
| `/extensions/time_reference/run/*` | varies | Run-level clock provenance (parser, status, start/end clocks, elapsed offsets, anchor mode) |
| `/extensions/time_reference/windows/*` | (T,) arrays | Window-level references aligned to run and subject anchor, including optional time-bin IDs/labels |
| `/extensions/e_kappa/*` | varies | Energetic curvature |
| `/extensions/rfm/*` | varies | Resonant frequency modes |
| `/extensions/o_koh/*` | varies | Organizational coherence |
| `/extensions/tig/*` | varies | Temporal integrity grade |

### JSON Manifest

```json
{
  "dataset_id": "ds003490:sub-001",
  "meta_indices": {
    "windows": 45,
    "j_hat_mean_trace": -0.023,
    "j_hat_rotation_frob": 0.156
  },
  "robust_summary": {
    "axes": { "summary": {...}, "reliability": {...} },
    "subcoords": { "summary": {...}, "reliability": {...} }
  },
  "extensions": {
    "e_kappa": { "mean_kappa": 0.042 },
    "tig": { "tau": 12.5, "TIG": 0.79 }
  },
  "within_run_labels": {
    "output_name": "within_run_state_v1",
    "assigned_frac": 1.0,
    "matched_rules": [{ "id": "ds006623_lor_task2", "source_type": "boundary_table" }]
  }
}
```

---

## Testing

```powershell
pip install pytest
python -m pytest tests/ -v
```

The test suite covers feature extraction, MNPS projection, Jacobian estimation, schema validation, manifests, and I/O contracts.

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Download failure | Logged to `failed_files.txt`; skipped on resume |
| Feature extraction error | Epoch skipped; logged with traceback |
| Insufficient epochs | Subject skipped with coverage warning |
| QC flag failure | Epoch excluded from MNPS projection |

### Recovery Commands

```powershell
# Re-run mndm features (safe to re-run; skips completed files)
python -m mndm.cli features --dataset ds003490

# Re-run MNPS summarization
python -m mndm.cli summarize --dataset ds003490
```

---

## Performance

| Dataset Size | Typical Runtime | Memory |
|--------------|-----------------|--------|
| ~10 subjects | 5-15 min | 2-4 GB |
| ~100 subjects | 1-2 hours | 4-8 GB |
| ~500 subjects | 6-12 hours | 8-16 GB |

Adjust ingest worker settings in `openneuro_ingest` for your hardware.

---

## Theory Reference

This pipeline implements the data preparation layer for:

- **Noetic Diffusion Theory (NDT)**: Models brain states as rhythmically scheduled denoising on learned manifolds
- **Meta-Noetic Phase Space (MNPS)**: Low-dimensional embedding with mobility (m), diffusivity (d), entropy (e) axes
- **Stratified MNPS**: 9D decomposition revealing mechanistic contributions to each axis
- **Meta-Noetic Jacobian (MNJ)**: Second-order dynamics capturing how the rules of change themselves vary

See `docs/articles/` for theoretical foundations.

---

## License

GNU GENERAL PUBLIC LICENSE v3. Se LICENSE in the root folder.

## Contact

For questions about this pipeline, open an issue or contact the maintainers.
