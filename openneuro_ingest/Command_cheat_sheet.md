# Noetic Ingest CLI Reference

Quick command reference for the OpenNeuro BIDS ingest pipeline.

---

## Installation

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
pip install openneuro-py
```

---

## Command Overview

| Command | Description |
|---------|-------------|
| `download` | Fetch dataset from OpenNeuro and build file index |
| `features` | Compute per-epoch multimodal features |
| `summarize` | Project to MNPS coordinates and compute Jacobians |
| `resummarize` | Re-run summarize step (alias) |
| `all` | Run full pipeline (download → features → summarize) |
| `continue` | Resume from existing artifacts |

---

## Global Options

```
--dataset DATASET [DATASET ...]   Dataset ID(s), e.g., ds003490
--config PATH                     Config file (default: config/config_ingest.yaml)
--out-dir PATH                    Output directory override
--data-dir PATH                   Raw data directory override
--subject ID                      Process single subject, e.g., 001
--n-jobs N                        Parallel workers (default: min(cores, 6))
--h5-mode {dataset,subject}       HDF5 output granularity (default: subject)
```

### MNPS Overrides

```
--mnps-k K                        kNN neighbors for Jacobian estimation
--mnps-super-window N             Super-window length for local Jacobians
--mnps-derivative {sav_gol,central}
--mnps-derivative-window N        Savitzky-Golay window length
--mnps-derivative-poly N          Savitzky-Golay polynomial order
```

---

## Common Workflows

### Full Pipeline (Single Dataset)

```powershell
python -m noetic_ingest.cli all --dataset ds003490
```

### Full Pipeline (Multiple Datasets)

```powershell
python -m noetic_ingest.cli all --dataset ds003490 ds005114
```

### Step-by-Step Execution

```powershell
# 1. Download and index
python -m noetic_ingest.cli download --dataset ds003490

# 2. Compute features
python -m noetic_ingest.cli features --dataset ds003490

# 3. Summarize (MNPS + Jacobians)
python -m noetic_ingest.cli summarize --dataset ds003490
```

### Single Subject Processing

```powershell
python -m noetic_ingest.cli features --dataset ds003490 --subject 001
python -m noetic_ingest.cli summarize --dataset ds003490 --subject 001
```

### Partial Pipeline

```powershell
# Skip download, run features → summarize
python -m noetic_ingest.cli all --dataset ds003490 --start-from features

# Only download and compute features
python -m noetic_ingest.cli all --dataset ds003490 --stop-after features
```

### Resume Interrupted Run

```powershell
python -m noetic_ingest.cli continue --dataset ds003490
```

### Re-summarize with Different Parameters

```powershell
python -m noetic_ingest.cli resummarize --dataset ds003490 --mnps-k 30 --mnps-super-window 5
```

---

## Output Structure

```
noetic_output/
└── <dataset_id>/
    ├── file_index.csv                    # BIDS file index
    ├── features.csv                      # Per-epoch features
    ├── failed_files.txt                  # Failed file log (if any)
    └── mnps_<dataset_id>_<timestamp>/
        └── <sub_id>[_<ses_id>]/
            ├── summary.json              # MNPS manifest + meta-indices
            ├── qc_reliability.json       # Split-half reliability
            ├── qc_summary.json           # Coverage + QC flags
            └── <sub_id>[_<ses_id>].h5    # MNPS tensors
```

### HDF5 Contents

| Dataset | Shape | Description |
|---------|-------|-------------|
| `/time` | (T,) | Time index in seconds |
| `/x` | (T, 3) | MNPS coordinates [m, d, e] |
| `/x_dot` | (T, 3) | MNPS derivatives |
| `/coords_v2` | (T, 9) | Stratified MNPS (if enabled) |
| `/jacobian` | (W, 3, 3) | Local Jacobians |
| `/jacobian_9D` | (W, 9, 9) | Stratified Jacobians (if enabled) |
| `/nn/indices` | (T, k) | kNN neighbor indices |
| `/labels/stage` | (T,) | Sleep stage codes |
| `/events/*` | varies | Event markers |
| `/extensions/*` | varies | E-Kappa, RFM, O-Koh, TIG |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Private dataset access | Run `openneuro-py login` once |
| Slow downloads | Re-run; partial files resume automatically |
| JSON serialization error | Update to latest code version |
| Out of memory | Reduce `--n-jobs` or `--mem-budget-gb` |
| Missing features | Check `failed_files.txt` for errors |

### Fallback: OpenNeuro “curl script” (presigned S3 URLs)

If OpenNeuro provides a text/`.sh` script containing many lines like `curl ... -o <path>`,
you can download it on Windows via:

```powershell
# Downloads into config paths.received_dir\<dataset_id>\...
# and (optionally) writes processed\<dataset_id>\file_index.csv
python src\presigned_fallback.py --script H:\path\to\openneuro_download.sh --dataset ds004504 --build-index
```

Useful options:

```powershell
# Preview what would be downloaded (no network)
python src\presigned_fallback.py --script H:\path\to\openneuro_download.sh --dataset ds004504 --dry-run

# By default, derivatives/ is skipped. Use this only if you really want it.
python src\presigned_fallback.py --script H:\path\to\openneuro_download.sh --dataset ds004504 --include-derivatives

# Override base directories
python src\presigned_fallback.py --script H:\path\to\openneuro_download.sh --dataset ds004504 --data-dir E:\Science_Datasets\openneuro\received --processed-dir E:\Science_Datasets\openneuro\processed
```

### Cleanup Commands

```powershell
# Remove temporary feature files
Remove-Item noetic_output\<dataset>\features_*.csv

# Clear failed files log (to retry all)
Remove-Item noetic_output\<dataset>\failed_files.txt
```

---

## Examples

### EEG-only Dataset (Parkinson's)

```powershell
python -m noetic_ingest.cli all --dataset ds003490 --n-jobs 4
```

### fMRI Dataset (Pixar)

```powershell
python -m noetic_ingest.cli all --dataset ds000228 --n-jobs 2 --mem-budget-gb 8
```

### Batch Processing from Config

```powershell
# Process all datasets listed in config (excluding pca_results)
python -m noetic_ingest.cli all

# Include datasets marked as pca_results
python -m noetic_ingest.cli all --include-pca-results
```

---

## Version

```powershell
python -m noetic_ingest.cli --version
```
