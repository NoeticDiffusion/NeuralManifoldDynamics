## Scope

- Ingest stack: `ndt_ingest.ingest_version=1.2`, mapping version `1.2`.
- Config source: `openneuro/config/config_ingest.yaml` (as of 2025-12-05).
- Datasets: fMRI propofol ds003171 (current), planned ds006623 (same code paths; dataset overrides noted below).
- Code paths: base fMRI features in `src/noetic_ingest/features/fmri.py`; advanced fMRI metrics in `features/fmri_connectivity.py`, `features/fmri_dynamic.py`, `features/fmri_phase.py`, `features/fmri_spectral.py`; MNPS projection in `mnps/projection.py`; subject runner in `pipeline/summary.py`; regional MNPS/MNJ in `pipeline/regional_mnps.py` and network grouping in `pipeline/regions.py`.

## 1) Base fMRI feature construction (per window)

Windowing (from `features.fmri.datasets.*`):

| dataset | window_sec | step_sec | notes |
| --- | --- | --- | --- |
| ds003171 | 20 s | 10 s | overrides default 30/15; TR ≈ dataset BOLD TR, indices use left-edge alignment (`t_start = start_idx/sfreq`, `t_end = end_idx/sfreq`). |
| ds006623 | 20 s | 10 s | same as ds003171. |

Sampling: `sfreq = 1/TR` passed from preprocessing. Windows are left-aligned slices; no centering offset.

Base features (computed in `features/fmri.py::compute_fmri_features`):

- `fmri_entropy_global`: mean of per-ROI variance within window. Units: variance; no log transform; NaN-safe mean over regions.
- `fmri_lf_power`: mean of squared amplitude over all ROI×time samples in window. Assumes upstream bandpass (see preprocessing).
- `fmri_lf_power_delta`: absolute change in `fmri_lf_power` relative to the previous window (0.0 for the first window or when either value is NaN); coarse temporal-variability proxy for LF power.
- `fmri_modularity`: greedy community modularity on ROI×ROI Pearson correlation (unthresholded; NaNs → 0). Algorithm: NetworkX `greedy_modularity_communities` + `modularity`.

Advanced metrics (used by MNPS mapping v1.2):

- Static FC/ALFF/fALFF (`features/fmri_connectivity.py`): `fmri_global_FC_mean/std`, `fmri_ALFF_mean/std`, `fmri_fALFF_mean/std`. Uses proportional bandpass (config `alff_band`) and Pearson FC over full run (not windowed).
- Dynamic FC (`features/fmri_dynamic.py`): sliding FC over BOLD with config `dynamic_connectivity.window_length_sec`, `step_sec` (ds003171/ds006623: 30 s / 5 s, retain_matrices=false). Summary keys: `fmri_dFC_variance`, `fmri_dFC_std` (if enabled), `fmri_dFC_mean` (if enabled), `fmri_dFC_entropy`.
- Graph metrics (`metrics/graph.py`): on FC matrix (static or mean of dFC stack). Threshold: proportional density=0.15. Keys: `graph_global_efficiency`, `graph_modularity`, `graph_participation_coeff_mean/std`, `graph_avg_path_length` (hubness disabled by default).
- Phase/Kuramoto (`features/fmri_phase.py`): bandpass 0.01–0.1 Hz (ds003171 overrides 0.008–0.09), regional sets VIS/SMN/FPN/SAL/DMN; outputs `fmri_kuramoto_global_mean/std` plus per-network means/stds.
- Spectral (`features/fmri_spectral.py`): `fmri_spectral_entropy_global` (band 0.01–0.1 default; ds003171/ds006623 use 0.008–0.09).

Preprocessing inputs (from `preprocess.fmri`):

- Atlas: Schaefer2018 200-parcel, 7-network order (`atlas_path`/`atlas_labels` in config).
- Bandpass: default [0.01, 0.1] Hz; ds003171 and ds006623 override to [0.008, 0.09]; atlas resampling to BOLD grid enabled.

## 2) Projection to 3D MNPS (m, d, e)

Weights resolved via `ndt_ingest.mnps_mapping` (version 1.2) override the legacy `mnps_projection` defaults. Feature columns used are median/MAD normalized (`normalize=robust_z`) before weighting; weighted sums are **not** further clipped or rescaled.

Axis weight table (full, fMRI-relevant rows only):

| axis | feature | weight |
| --- | --- | --- |
| m | `fmri_global_FC_mean` | 0.25 |
| m | `fmri_kuramoto_global_mean` | 0.15 |
| m | `fmri_variance_global` | 0.10 |
| d | `fmri_dFC_variance` | 0.25 |
| d | `fmri_kuramoto_global_std` | 0.15 |
| d | `fmri_graph_modularity` | 0.10 (note: current graph metric key is `graph_modularity`; this weight is effectively zero unless renamed) |
| e | `fmri_spectral_entropy_global` | 0.20 |
| e | `fmri_signal_power` | 0.15 |
| e | `fmri_variance_global` | 0.15 |

Effective formulas (with above caveat) for ds003171/ds006623:

- `m_raw = 0.25·fmri_global_FC_mean + 0.15·fmri_kuramoto_global_mean + 0.10·fmri_variance_global` (all inputs robust-z).
- `d_raw = 0.25·fmri_dFC_variance + 0.15·fmri_kuramoto_global_std` (+0.10·graph_modularity if column is renamed/present).
- `e_raw = 0.20·fmri_spectral_entropy_global + 0.15·fmri_signal_power + 0.15·fmri_variance_global`.

Legacy `mnps_projection` weights (`m = -0.40·fmri_entropy_global + 0.60·fmri_lf_power`, `d = 1.0·fmri_modularity`, `e = 1.0·fmri_entropy_global`) are **not used** when `ndt_ingest.mnps_mapping` is present (v1.2 default).

## 3) Normalisation, rescaling, Jacobian

- Input feature normalisation: per-column `robust_z` (median/MAD) on the subset of columns referenced by the weight maps.
- 3D MNPS output: weighted sums as above; no additional min-max or sigmoid rescale.
- Time base: MNPS window `window_sec=8.0`, `overlap=0.5` ⇒ dt = 4.0 s on the MNPS axis (`mnps.fs_out=4.0` denotes intended sampling rate after windowing).
- Derivatives: Savitzky–Golay (`window=7`, `polyorder=3`, auto-adjusted when short).
- kNN: `k=20`, metric `euclidean`, whitening enabled.
- Jacobian: ridge `alpha=1.0`, `super_window=3`, distance-weighted by default.
- Missing/NaN handling: columns used for kNN/Jacobian are median-filled before neighbor search; QC filters drop epochs only via `qc_ok_*` if present (no fMRI-specific QC columns currently).

## 4) Stratified MNPS v2 (global; regional v2 not used)

Enabled for ds003171 and ds006623 (`mnps_v2.datasets.<id>.enabled=true`). Subcoordinates are linear reparameterisations of the same three base fMRI features (`fmri_entropy_global`, `fmri_lf_power`, `fmri_modularity`) with per-subcoord weights; inputs are optionally `normalize=robust_z`, then each subcoordinate is squashed to [0,1] via median/MAD logistic (`_normalize_matrix_unit_interval`).

Global v2 weight table (identical for ds003171 and ds006623):

| subcoord | weights |
| --- | --- |
| m_a | {-0.7·fmri_entropy_global, 0.3·fmri_lf_power} |
| m_e | {-0.3·fmri_entropy_global, 0.7·fmri_lf_power} |
| m_o | {-1.0·fmri_lf_power, 1.0·fmri_lf_power_delta} |
| d_n | {1.0·fmri_modularity} |
| d_l | {0.5·fmri_modularity, 0.5·fmri_entropy_global} |
| d_s | {0.5·fmri_modularity, 0.5·fmri_lf_power} |
| e_e | {1.0·fmri_entropy_global} |
| e_s | {0.5·fmri_entropy_global, -0.5·fmri_modularity} |
| e_m | {0.5·fmri_lf_power, -0.5·fmri_modularity} |

Because only three underlying features are used, the v2 space has effective rank 3.

Jacobian in v2: optional (default enabled); uses the same kNN/derivative parameters as 3D MNPS.

## 5) Regional MNPS / Jacobians (`/regional_mnps`)

- Inputs: regional BOLD stored under `/regions/bold` (atlas-resampled Schaefer parcels). Regions are grouped to networks by name tokens (`regions.py::infer_network_label`), mapping Schaefer 7-network labels to `{VIS, SMN, FPN, SAL, DMN, DAN, VAN, LIM}`. `regional_mnps.networks` whitelist matches this set.
- Per-network signal: mean across parcels in the network (`aggregate_group_timeseries`).
- Windowing: `window_sec=8.0`, `overlap=0.5` → dt = 4.0 s on the regional MNPS axis. Minimum length: `min_segment_length_tr=40`; segments with <3 MNPS windows are dropped.
- Regional coordinates (per network):
  - m: window mean of network-mean BOLD (integration proxy).
  - d: window variance of network-mean BOLD.
  - e: sample entropy of window (`m=2`, `r=0.2·std`), clamped to [0,5].
  - Normalisation: per-axis robust logistic to [0,1] (median/MAD).
- Derivatives: Savitzky–Golay (window=7, poly=3) with dt above.
- kNN/Jacobian: k=10, metric=euclidean, whitening on; ridge α=1.0, `super_window=3`, distance-weighted. Outputs stored per network under `/regional_mnps/<net>`.
- Metrics stored (per network): `m_mean/std`, `d_mean/std`, `e_mean/std`, `trace_mean/std`, `frobenius_mean`, `rotation_norm_mean`, `anisotropy_mean`. Also written to `regional_mnps_subjects.csv`.
- Block-Jacobian summaries (optional CSV via `write_block_jacobians_csv`): ordered (target, source) pairs over the stacked regional state (3D blocks).
- ds006623 differences: none in regional settings; only preprocessing bandpass differs (0.008–0.09 like ds003171).

## 6) Versioning / config linkage

- Ingest version: `ndt_ingest.ingest_version = "1.2"`.
- Mapping version: `ndt_ingest.mnps_mapping.version = "1.2"`, `normalize = robust_z`.
- Stratified v2: enabled per-dataset for ds003171/ds006623; weight table above.
- Regional MNPS: enabled globally (`regional_mnps.enabled=true`); outputs `regional_mnps_subjects.csv` in each run directory.
- Run naming: subject outputs under `processed/<ds>/mnps_<ds>_<timestamp>/`.

## 7) Quick reconstruction recipe (external reader)

Given `features.csv`, `config_ingest.yaml`, and the HDF5 outputs:

1. Use the dataset-specific fMRI windows (20 s, step 10 s) to align `fmri_entropy_global`, `fmri_lf_power`, `fmri_modularity` rows to time; advanced features are run-level (static FC/ReHo/ALFF) or long-window (dFC) scalars joined to each row.
2. Apply `robust_z` to the feature columns referenced by `ndt_ingest.mnps_mapping` (table above), then form weighted sums for m/d/e.
3. Build MNPS timebase: window 8 s, overlap 0.5 → dt = 4 s; derivatives via Savitzky–Golay (7,3).
4. kNN (k=20, euclidean, whiten) and ridge Jacobian (α=1.0, super_window=3) to match `/jacobian` tensors.
5. Stratified v2: apply the 9 weight rows to the three base features, `robust_z` the inputs, then logistic-normalize each subcoordinate column to [0,1]; Jacobian optional (same params).
6. Regional MNPS: mean per network from `/regions/bold`, window 8 s / overlap 0.5, logistic normalise per-axis, Jacobian with k=10, α=1.0.

Any deviations between ds003171 and ds006623 are limited to preprocessing bandpass and identical per-dataset fMRI window overrides; the MNPS projection, normalization, and Jacobian settings are otherwise shared.

