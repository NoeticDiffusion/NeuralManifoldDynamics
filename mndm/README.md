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
- **Resume-friendly**: Interrupted runs can continue from existing artifacts
- **Optional extras (recent)**:
  - FD censoring of high-motion epochs (framewise_displacement > 0.5 mm, ±1 neighbour)
  - Provisional flag for fMRI modularity when a window has very few volumes
  - Event→MNPS mapping (opt-in) writing binary labels aligned to MNPS time
  - Window start/end (seconds) per MNPS point in HDF5 for clearer time alignment

---

## Requirements

- Python 3.11+
- Dependencies: `numpy`, `scipy`, `pandas`, `mne`, `h5py`, `pyyaml`, `tqdm`, `joblib`
- Optional: `openneuro-py` (for dataset downloads)
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
$env:PYTHONPATH="H:/SourceRepo2/NeuralManifoldDynamics/mndm/src;H:/SourceRepo2/NeuralManifoldDynamics/core/src;H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/apollo_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/vitaldb_ingest/src"
```

---

## Quick Start

### Step-by-Step

```powershell
# Preflight check: validate paths, config, participants table, and index preview.
python -m mndm.cli prerequisite-check --dataset ds003490

# Download (ingest)
python -m openneuro.cli download --dataset ds003490

# Compute per-epoch features (mndm)
python -m mndm.cli features --dataset ds003490

# Project to MNPS and estimate Jacobians
python -m mndm.cli summarize --dataset ds003490

# Or run both in one step:
# python -m mndm.cli all --dataset ds003490

# (Optional) Pack a completed MNPS run (many small H5) into one container H5
# Output: <processed>/<dataset>/<latest mnps_*>/packed.h5
python -m mndm.cli pack --dataset ds003490
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
| `/mnps_3d` | (T, 3) | MNPS coordinates [m, d, e] |
| `/mnps_3d_dot` | (T, 3) | MNPS time derivatives |
| `/features_raw/values` | (T, K) | Raw feature matrix in original scale |
| `/features_raw/names` | (K,) | Feature names aligned to raw values |
| `/features_raw/metadata/*` | (K,) per field | Machine-readable feature provenance and usage flags |
| `/features_robust_z/values` | (T, K) | Strict robust-z feature matrix |
| `/features_robust_z/names` | (K,) | Feature names aligned to strict robust-z values |
| `/features_robust_z/metadata/*` | (K,) per field | Same feature metadata layout as `/features_raw/metadata/*` |
| `/coords_9d/values` | (T, 9) | Stratified subcoordinates |
| `/jacobian/J_hat` | (W, 3, 3) | Local Jacobian matrices |
| `/jacobian/J_dot` | (W, 3, 3) | Jacobian time derivatives |
| `/jacobian/centers` | (W,) | Window center indices |
| `/nn/indices` | (T, k) | kNN neighbor indices |
| `/labels/stage` | (T,) | Sleep stage labels |
| `/regional_mnps/<network>/mnps` | (Tr, 3) | Canonical regional MNPS output for any modality |
| `/regions/bold` | (R, T') | Optional raw regional fMRI time series |
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
