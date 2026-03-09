# Noetic Diffusion Data Ingest

Production-grade pipeline for ingesting BIDS-compliant neuroimaging data from OpenNeuro and computing Meta-Noetic Phase Space (MNPS) representations.

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

---

## Installation

```powershell
# Clone and enter directory
cd NoeticDiffusionDataIngest/openneuro

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
pip install openneuro-py

# (Optional) For private datasets
openneuro-py login
```

---

## Quick Start

### Full Pipeline

```powershell
python -m noetic_ingest.cli all --dataset ds003490
```

### Step-by-Step

```powershell
# Download and index BIDS files
python -m noetic_ingest.cli download --dataset ds003029

# Example: HUP epilepsy iEEG dataset (BIDS iEEG)
python -m noetic_ingest.cli download --dataset ds004100

# Compute per-epoch features
python -m noetic_ingest.cli features --dataset ds003490

# Project to MNPS and estimate Jacobians
python -m noetic_ingest.cli summarize --dataset ds003490

# (Optional) Pack a completed MNPS run (many small H5) into one container H5
# Output: <processed>/<dataset>/<latest mnps_*>/packed.h5
python -m noetic_ingest.cli pack --dataset ds003490
```

See [Command_cheat_sheet.md](Command_cheat_sheet.md) for complete CLI reference.

---

## Project Structure

```
openneuro/
├── config/
│   └── config_ingest.yaml        # Pipeline configuration
├── src/noetic_ingest/
│   ├── cli.py                    # Command-line interface
│   ├── orchestrate.py            # Pipeline orchestration
│   ├── bids_index.py             # BIDS file indexing
│   ├── preprocess.py             # Signal preprocessing
│   ├── features/                 # Feature extractors
│   │   ├── eeg.py                # EEG band power, entropy, connectivity
│   │   ├── fmri.py               # fMRI regional features
│   │   └── {ecg,eda,emg,eog,resp}.py
│   ├── mnps/                     # MNPS computation
│   │   ├── projection.py         # Feature → MNPS mapping
│   │   ├── jacobian.py           # Local Jacobian estimation
│   │   ├── extensions.py         # E-Kappa, RFM, O-Koh, TIG
│   │   ├── robustness.py         # Reliability metrics
│   │   └── schema.py             # Payload dataclass
│   ├── pipeline/                 # Summarization pipeline
│   │   ├── summary.py            # Runner classes
│   │   ├── context.py            # Configuration resolution
│   │   ├── extensions_compute.py # Extension computation
│   │   ├── robustness_helpers.py # QC summaries
│   │   ├── extractors.py         # Data extraction utilities
│   │   └── regions.py            # Network mapping
│   └── io/                       # Output writers
│       ├── h5_writer.py          # HDF5 tensor output
│       └── json_writer.py        # JSON manifest output
└── tests/                        # Test suite
```

---

## Configuration

Edit `config/config_ingest.yaml` to customize the pipeline.

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
mnps_v2:
  enabled: true
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

---

## Output Format

### Directory Structure

```
noetic_output/<dataset_id>/
├── file_index.csv                # Indexed BIDS files
├── features.csv                  # Per-epoch feature matrix
└── mnps_<dataset>_<timestamp>/
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

| Path | Shape | Description |
|------|-------|-------------|
| `/time` | (T,) | Time index (seconds) |
| `/mnps` | (T, 3) | MNPS coordinates [m, d, e] |
| `/mnps_dot` | (T, 3) | MNPS time derivatives |
| `/coords_9d` | (T, 9) | Stratified subcoordinates |
| `/jacobian` | (W, 3, 3) | Local Jacobian matrices |
| `/jacobian_dot` | (W-1, 3, 3) | Jacobian time derivatives |
| `/jacobian_centers` | (W,) | Window center indices |
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

Current coverage: 37 tests across feature extraction, MNPS projection, Jacobian estimation, and I/O.

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
# Resume interrupted run
python -m noetic_ingest.cli continue --dataset ds003490

# Clear failures and retry all
Remove-Item noetic_output\ds003490\failed_files.txt
python -m noetic_ingest.cli all --dataset ds003490
```

---

## Performance

| Dataset Size | Typical Runtime | Memory |
|--------------|-----------------|--------|
| ~10 subjects | 5-15 min | 2-4 GB |
| ~100 subjects | 1-2 hours | 4-8 GB |
| ~500 subjects | 6-12 hours | 8-16 GB |

Adjust `--n-jobs` and `--mem-budget-gb` for your hardware.

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

Proprietary - Noetic Diffusion Research

## Contact

For questions about this pipeline, open an issue or contact the maintainers.
