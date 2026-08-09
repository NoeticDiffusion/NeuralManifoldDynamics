# MNDM + OpenNeuro Ingest CLI Reference

Quick command reference for the split pipeline:
- `openneuro` handles dataset download
- `mndm` handles features, summarization, Jacobians, exports, packing, and checks

---

## Installation

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

If you run from this monorepo checkout without installing the packages editably, set `PYTHONPATH` first:

```powershell
$env:PYTHONPATH="H:/SourceRepo2/NeuralManifoldDynamics/mndm/src;H:/SourceRepo2/NeuralManifoldDynamics/core/src;H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/apollo_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/vitaldb_ingest/src"
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
# Run a preflight check before the first pipeline run
python -m mndm.cli prerequisite-check --dataset ds003490

# Compute per-epoch features
python -m mndm.cli features --dataset ds003490

# Force re-extraction — discard cached intermediate JSONs.
# Required after changing preprocessing settings (e.g. enabling ICA,
# bad-channel detection, or notch/bandpass parameters).
python -m mndm.cli features --dataset ds003490 --force-features

# Summarize (MNPS + Jacobians)
python -m mndm.cli summarize --dataset ds003490

# Summarize with one-shot cohort anchor fitting
python -m mndm.cli summarize --dataset ds003944 --fit-anchor

# Run features -> summarize in one command
python -m mndm.cli all --dataset ds003490

# Non-BIDS BioSemi BDF collection: validate mapping/channel policy first.
# Start from mndm/config/bdf_config_ingest_template.yaml.
python -m mndm.cli prerequisite-check --dataset my_bdf_dataset `
  --config mndm/config/config_ingest_my_bdf_dataset.yaml

# Run a mapped subject smoke test, then summarize it.
python -m mndm.cli features --dataset my_bdf_dataset `
  --config mndm/config/config_ingest_my_bdf_dataset.yaml `
  --subject 01 --n-jobs 1 --force-features
python -m mndm.cli summarize --dataset my_bdf_dataset `
  --config mndm/config/config_ingest_my_bdf_dataset.yaml `
  --subject 01 --h5-mode subject --n-jobs 1

# Run the complete BDF archive after the smoke test is accepted.
python -m mndm.cli all --dataset my_bdf_dataset `
  --config mndm/config/config_ingest_my_bdf_dataset.yaml `
  --n-jobs 2 --mem-budget-gb 8 --force-features

# Run features -> one-shot cohort-anchored summarize
python -m mndm.cli all --dataset ds003944 --fit-anchor

# Re-run summarize only
python -m mndm.cli resummarize --dataset ds003490

# Verify MEG physical-unit transform replay against one exported H5.
python -m mndm.cli meg-transform-replay --h5 path/to/sub-001.h5 --dataset ds003645 --config mndm/config/config_ingest_ds003645.yaml

# Optional report-only frozen-sector QC (requires sensor_topography_qc.enabled).
python -m mndm.cli sensor-topography-qc --h5 path/to/sub-001.h5 --dataset ds003645 --config mndm/config/config_ingest_ds003645.yaml --out reports/sub-001_sensor_topography_qc.json

# Pack a run into a single H5
python -m mndm.cli pack --dataset ds003490

# Validate run structure
python -m mndm.cli check-structure --dataset ds003490

# MNDM 2.1: fit a cohort feature anchor from summarized H5 outputs
python -m mndm.cli anchors-fit --h5-root <processed_dir> --dataset ds003944 --config mndm/config/config_ingest_ds003944.yaml --anchor-id ds003944_fep_control_v2_1 --out anchors/ds003944_fep_control_v2_1.json

# MNDM 2.1: smoke-test hard-to-separate OpenNeuro EEG cohorts
python -m mndm.cli anchor-smoke --dataset ds003478 ds003944 ds004504 --data-dir M:/datasets/received/openneuro --h5-root <processed_dir> --out anchor_smoke_report.json

# MNDM 2.1: sensitivity sweep over scale and clipping policies
python -m mndm.cli anchor-sensitivity --h5-root <processed_dir> --dataset ds003944 --config mndm/config/config_ingest_ds003944.yaml --out anchor_sensitivity_ds003944.json
```

### Common Options

```
--dataset DATASET [DATASET ...]   Dataset ID(s), e.g., ds003490
--config PATH                     Config file (default: config/config_ingest.yaml)
--out-dir PATH                    Processed output directory override
--data-dir PATH                   Raw data directory override
--subject ID                      Process single subject, e.g., 001
--h5-mode {dataset,subject}       HDF5 output granularity (default: subject)
--n-jobs N                        Parallel workers (default: min(cores, 6))
--mem-budget-gb N                 Memory budget in GB for worker scaling
--fit-anchor                      Fit and freeze a cohort anchor from features before summarize
--anchor-id ID                    Optional stable id for one-shot anchor fitting
--anchor-scale-method {iqr,mad,qn}
--anchor-min-subjects N
```

### MNDM 2.1 Anchor Commands

`anchors-fit` builds a subject-balanced feature-anchor JSON artifact from
existing H5 `/features_raw` exports. The artifact stores center/scale,
quantiles, MAD/IQR/Qn scale estimates, subject/epoch support counts, and a
stable `anchor_hash`.

Common options:

```text
--h5-root PATH                    H5 file, run directory, processed root, or dataset directory
--dataset DATASET                 Optional dataset id when h5-root is a processed root
--config PATH                     Config used to replay feature_standardization pre-transforms
--anchor-id ID                    Stable anchor identifier
--scale-method {iqr,mad,qn}       Default scale method for downstream projection
--min-subjects N                  Minimum subject support per feature
```

`anchor-smoke` is a lightweight post-hoc report. By default it targets
`ds003478`, `ds003944`, and `ds004504`, checks raw dataset presence under
`M:/datasets/received/openneuro`, and, when `--h5-root` points to summarized H5
outputs, compares subject-anchored vs cohort-anchored separation summaries.

`anchor-sensitivity` runs the standard MNDM 2.1 sensitivity harness:
subject/cohort anchor comparison, scale-method sweep (`iqr,mad,qn`), and
clip-threshold sweep (`4,6,9` by default).

### One-shot frozen-anchor summarize

`summarize --fit-anchor` and `all --fit-anchor` fit a **frozen subject-balanced
anchor artifact from `features.parquet` / `features.csv` at summarize startup**,
save that anchor under the run directory, and then use it during the same
summarize pass.

This removes the old need to:

1. summarize once without anchor
2. run `anchors-fit`
3. summarize again with `mnps_projection.anchor.enabled=true`

Important:

- the one-shot path is still **fit -> freeze -> apply**
- it is **not** a dynamic on-the-fly anchor recomputed per downstream comparison
- resulting H5 outputs still declare `primary_coordinate_contract` and embed
  `/feature_anchors` when cohort anchoring is active

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

# 2) Preflight check: paths, config, participants table, index preview
python -m mndm.cli prerequisite-check --dataset ds003490

# 3) Recommended one-step MNDM run
python -m mndm.cli all --dataset ds003490

# 4) Or run the stages separately
python -m mndm.cli features --dataset ds003490
python -m mndm.cli summarize --dataset ds003490

# 5) (Optional) Pack H5
python -m mndm.cli pack --dataset ds003490
```

---

## Output Structure (processed dir)

```
<processed_dir>/
└── <dataset_id>/
    ├── file_index.csv                    # BIDS file index
    ├── features.csv / features.parquet   # Per-epoch features
    ├── failed_files.txt                  # Failed file log (if any)
    └── neuralmanifolddynamics_<dataset_id>_<timestamp>/
        ├── features_snapshot.json        # Snapshot of feature columns and stats
        ├── run_manifest.json             # Run-level manifest and capability summary
        └── <subject_run_dir>/
            ├── summary.json              # MNPS manifest + meta-indices
            ├── qc_reliability.json       # Split-half reliability
            ├── qc_summary.json           # Coverage + QC flags
            └── <subject_run_dir>.h5      # MNPS tensors
```

MNDM 2.1 runs may also include explicit coordinate-contract groups in each H5:

```text
/feature_anchors/...
/coords_3d_subject_anchored/{values,names}
/coords_9d_subject_anchored/{values,names}
/coords_3d_cohort_anchored/{values,names}
/coords_9d_cohort_anchored/{values,names}
```

`run_manifest.json` reports capability flags for these paths so downstream
analysis can select the intended primary coordinate layer without guessing.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Private dataset access | Run `openneuro-py login` once |
| Slow downloads | Re-run; partial files resume automatically |
| JSON serialization error | Update to latest code version |
| Out of memory | Reduce `--n-jobs` |
| Parquet warnings | Ensure `pyarrow` is installed in `.venv` |
| Missing features | Check `failed_files.txt` for errors |
| First run fails early | Run `python -m mndm.cli prerequisite-check --dataset <ds>` |
| Changed preprocessing config, results unchanged | Re-run `features --force-features` to discard cached intermediate JSONs |
| ICA did not run | Confirm `preprocess.artifacts.method: "ica"` (not `preprocess.ica`); look for `Applied ICA` in logs |
| No components excluded by ICA | Check `eog_proxy_channels` name spelling; lower `ica_eog_threshold` / `ica_ecg_threshold` |
| MNE warning: n_components too high | Reduce `ica_n_components` or use a variance fraction such as `0.995` |
| Bad-channel count unexpectedly high | Tighten `var_high_factor` / `corr_thresh` under `robustness.bad_channels` |
| phi_cardiac_mean / phi_resp_mean missing | Phase anchor disabled by default. Add phase_anchor.enabled: true to YAML and re-run features. |
| Phase anchor R-peak detection very slow | Normal for whole-night ECG. Processed in 5-min chunks; install neurokit2. |
| hep_amplitude all NaN but cardiac phase populated | frontal_eeg_channels mismatch -- verify channel names against EDF/BIDS header. |

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
Remove-Item <processed_dir>\<dataset>\features_*.parquet

# Clear failed files log (to retry all)
Remove-Item <processed_dir>\<dataset>\failed_files.txt
```

