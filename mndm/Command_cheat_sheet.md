# MNDM + OpenNeuro Ingest CLI Reference

Quick command reference for the split pipeline:
- `openneuro` handles download + indexing + features
- `mndm` handles MNPS summarization + Jacobians + exports

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

## OpenNeuro Ingest (download only)

### Commands

```powershell
# Download and index
python -m openneuro.cli download --dataset ds003490

# Download only
python -m openneuro.cli download --dataset ds003490
```

### Common Options

```
--dataset DATASET [DATASET ...]   Dataset ID(s), e.g., ds003490
--config PATH                     Config file (default: config/config_ingest.yaml)
--out-dir PATH                    Processed output directory override
--data-dir PATH                   Raw data directory override
--subject ID                      Process single subject, e.g., 001
--n-jobs N                        Parallel workers (default: min(cores, 6))
--mem-budget-gb N                 Memory budget in GB (default: 4)
```

---

## MNDM (features + summarize + pack + checks)

### Commands

```powershell
# Compute per-epoch features
python -m mndm.cli features --dataset ds003490

# Summarize (MNPS + Jacobians)
python -m mndm.cli summarize --dataset ds003490

# Re-run summarize only
python -m mndm.cli resummarize --dataset ds003490

# Pack a run into a single H5
python -m mndm.cli pack --dataset ds003490

# Validate run structure
python -m mndm.cli check-structure --dataset ds003490
```

### Common Options

```
--dataset DATASET [DATASET ...]   Dataset ID(s), e.g., ds003490
--config PATH                     Config file (default: config/config_ingest.yaml)
--out-dir PATH                    Processed output directory override
--data-dir PATH                   Raw data directory override
--subject ID                      Process single subject, e.g., 001
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

## End-to-End Workflow

```powershell
# 1) Download (ingest)
python -m openneuro.cli download --dataset ds003490

# 2) Features (mndm)
python -m mndm.cli features --dataset ds003490

# 3) Summarize (mndm)
python -m mndm.cli summarize --dataset ds003490

# 4) (Optional) Pack H5
python -m mndm.cli pack --dataset ds003490
```

---

## Output Structure (processed dir)

```
<processed_dir>/
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

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Private dataset access | Run `openneuro-py login` once |
| Slow downloads | Re-run; partial files resume automatically |
| JSON serialization error | Update to latest code version |
| Out of memory | Reduce `--n-jobs` |
| Missing features | Check `failed_files.txt` for errors |

---

## Fallback: OpenNeuro “curl script” (presigned S3 URLs)

If OpenNeuro provides a script containing `curl ... -o <path>`, you can use:

```powershell
python openneuro_ingest\src\presigned_fallback.py --script H:\path\to\openneuro_download.sh --dataset ds004504 --build-index
```

Common options:

```powershell
# Preview only
python openneuro_ingest\src\presigned_fallback.py --script H:\path\to\openneuro_download.sh --dataset ds004504 --dry-run

# Include derivatives (default: skip)
python openneuro_ingest\src\presigned_fallback.py --script H:\path\to\openneuro_download.sh --dataset ds004504 --include-derivatives

# Override base directories
python openneuro_ingest\src\presigned_fallback.py --script H:\path\to\openneuro_download.sh --dataset ds004504 --data-dir E:\Science_Datasets\openneuro\received --processed-dir E:\Science_Datasets\openneuro\processed
```

---

## Cleanup Commands

```powershell
# Remove temporary feature files
Remove-Item <processed_dir>\<dataset>\features_*.csv

# Clear failed files log (to retry all)
Remove-Item <processed_dir>\<dataset>\failed_files.txt
```

