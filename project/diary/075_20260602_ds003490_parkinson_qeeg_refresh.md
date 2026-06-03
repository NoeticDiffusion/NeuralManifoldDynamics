# ds003490 Parkinson qEEG refresh

## Goal

Rerun `ds003490` with a Parkinson-oriented conventional EEG comparator profile,
verify that the new sidecars/H5 groups are present, and document where the fresh
outputs live.

## Config change

Updated `mndm/config/config_ingest_ds003490.yaml` to enable a dataset-specific
`conventional_eeg` override for `ds003490` with:

- `packs: ["tier1", "complexity"]`
- relative bandpower
- ratios: `theta_alpha`, `delta_alpha`, `beta_alpha`, `slowing_index`
- peak markers: `alpha_peak_frequency`, `median_frequency`, `spectral_edge_95`
- complexity markers: `spectral_entropy`, `permutation_entropy`,
  `hjorth_complexity`, `hjorth_mobility`

Connectivity was intentionally left off for this Parkinson-facing pass because it
is less standard for the main manuscript-style comparator layer and increases
runtime materially.

## Run notes

Initial rerun attempt against the default processed root reused the existing
`features.parquet` / `intermediate/*.json` cache, so the new conventional EEG
columns were not regenerated.

To avoid destructive cleanup of the old processed tree, the successful rerun was
executed against a fresh processed root:

`M:/datasets/processed/openneuro_parkinson_qeeg_refresh`

Command used:

```powershell
$env:PYTHONPATH="H:/SourceRepo2/NeuralManifoldDynamics/mndm/src;H:/SourceRepo2/NeuralManifoldDynamics/core/src;H:/SourceRepo2/NeuralManifoldDynamics/openneuro_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/apollo_ingest/src;H:/SourceRepo2/NeuralManifoldDynamics/vitaldb_ingest/src"
python -m mndm.cli all --dataset ds003490 --config "mndm/config/config_ingest_ds003490.yaml" --out-dir "M:/datasets/processed/openneuro_parkinson_qeeg_refresh" --n-jobs 12
```

## Fresh output location

Run directory:

`M:/datasets/processed/openneuro_parkinson_qeeg_refresh/ds003490/neuralmanifolddynamics_ds003490_20260602_125856`

Observed top-level outputs:

- `features.csv`
- `features_snapshot.json`
- `normalization_report.json`
- `run_manifest.json`
- `stage_mapping_qc.json`
- `regional_mnps_subjects_125856.csv`
- `regional_block_jacobians_subjects_125856.csv`
- `stratified_block_jacobians_subjects_125856.csv`
- 75 per-grouping subject bundles with `summary.json`, `qc_summary.json`,
  `qc_reliability.json`, and `.h5`

## Verification

### Conventional comparator columns

The refreshed `features.csv` now contains the Parkinson-facing conventional EEG
columns, including:

- `eeg_conventional_relative_{delta,theta,alpha,beta,gamma}`
- `eeg_conventional_ratio_{theta_alpha,delta_alpha,beta_alpha,slowing_index}`
- `eeg_conventional_peak_{alpha_frequency,median_frequency,spectral_edge_95}`
- `eeg_conventional_complexity_{spectral_entropy,permutation_entropy,hjorth_complexity,hjorth_mobility}`

`features_snapshot.json` reports these columns in the dataset-wide feature
inventory.

### Sidecars / manifests

For `sub-001_OFF_rest`, `summary.json` now includes:

- `conventional_eeg.schema_version = "mndm.conventional_eeg.v1"`
- `packs = ["complexity", "tier1"]`
- `column_count = 16`
- family summaries for `complexity`, `peak`, `ratio`, and `relative`

Dataset-level sidecars confirm:

- `run_manifest.json`: 75 H5, 75 summary JSON, 75 QC summary JSON, 75 QC reliability JSON
- `normalization_report.json`: ComBat config present but status remains `disabled`
- `stage_mapping_qc.json`: dataset-wide aggregate plus per-grouping event/stage audit

### H5 contract

Sample H5 (`sub-001_OFF_rest.h5`) contains the expected additive groups:

- `/event_windows`
- `/codebooks`
- `/coverage`
- `/provenance`
- `/qc/windows`
- `/participant`
- `/extensions/conventional_eeg`

The sample also reports `features_raw` with 80 exported feature columns, which
now includes the conventional EEG comparator set.

## README

Updated `mndm/README.md` in the conventional EEG comparator section to explain
that Parkinson-focused usage typically starts with `tier1` + `complexity`
(relative bandpower, slowing ratios, peak/edge measures, and Hjorth/entropy
markers), with connectivity as an optional follow-up layer.
