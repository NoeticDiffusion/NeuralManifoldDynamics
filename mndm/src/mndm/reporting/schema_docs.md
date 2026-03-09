# Output Schemas – MNPS, Stratified MNPS v2, and Jacobians

This document describes the **subject-level output schema** for the OpenNeuro ingest pipeline as written by `h5_writer` and `json_writer`. It focuses on MNPS tensors, optional Stratified MNPS v2 subcoordinates, and Jacobians (3D MNPS and higher-dimensional Stratified variants).

## HDF5 Layout (per subject / session)

Top-level attributes:

- **dataset_id**: string (e.g. `"ds003490:sub-001:ses-01"`)
- **fs_out**: float – MNPS sampling rate (Hz)
- **window_sec**: float – analysis window length (seconds)
- **overlap**: float – window overlap fraction (e.g. `0.5`)
- **stage_codebook**: JSON‑encoded mapping from stage labels → int codes
- **participant_meta**: JSON‑encoded participant‑level metadata (raw fields)
- Convenience attrs (best-effort):
  - **group**: normalized diagnostic group (e.g. `AD`, `Parkinson`, `Control`)
  - **condition**: condition / medication status (e.g. `ON`, `OFF`)
  - **meta_*:** flattened scalar entries from `participant_meta`

Datasets and groups:

- `/time` – `float64[T]`  
  - Monotonically increasing time stamps (seconds since session start).
- `/mnps_3d` – `float32[T, 3]`  
  - MNPS coordinates `[m, d, e]` at each time point (standard 3D axes).
- `/mnps_3d_dot` – `float32[T, 3]`  
  - Time derivatives of `mnps_3d` (`ṁ, ḋ, ė`).
- `/features_raw/values` – `float32[T, K]`  
  - Raw per-epoch feature matrix in original scale.
- `/features_raw/names` – `str[K]`  
  - Feature names aligned to `/features_raw/values`.
- `/features_raw/metadata/*` – arrays of length `K`  
  - Machine-readable feature provenance and usage flags.
- `/features_robust_z/values` – `float32[T, K]`  
  - Strict robust-z version of the exported feature matrix.
- `/features_robust_z/names` – `str[K]`  
  - Feature names aligned to `/features_robust_z/values`.
- `/features_robust_z/metadata/*` – arrays of length `K`  
  - Same per-feature metadata layout as `/features_raw/metadata/*`.
- `/z` (optional) – `float32[T, K]`  
  - Embodied signals (e.g. HRV, respiratory phase) aligned to `/time`.
- `/labels/stage` (optional) – `int8[T]`  
  - Stage / task labels with codes described in `stage_codebook`.
- `/events/*` (optional) – `int64[N]` or `float64[N]` per event type  
  - Event indices or timestamps (SO/spindle, task events, etc.).
- `/nn/indices` (optional) – `int32[T, k]`  
  - kNN neighbour indices in MNPS space; used for Jacobian estimation.

Jacobian groups:

- `/jacobian/J_hat` (optional) – `float32[W, D, D]`  
  - Windowed local Jacobian estimates for primary MNPS coordinates (`D=3` in the current pipeline).
- `/jacobian/J_dot` (optional) – `float32[W-1, D, D]`  
  - Temporal differences between successive Jacobians (meta‑plasticity proxy).
- `/jacobian/centers` (optional) – `int32[W]`  
  - Indices into `/time` of the centre of each Jacobian window.

Stratified MNPS v2 (optional):

- `/coords_9d/values` – `float32[T, 9]`  
  - Time series of Stratified subcoordinates in canonical order  
    `[m_a, m_e, m_o, d_n, d_l, d_s, e_e, e_s, e_m]`.
- `/coords_9d/names` – `str[9]`  
  - Names of the subcoordinates (always in canonical order after normalization).
- `/coords_9d` group attrs:
  - `version = "9d"`

Stratified (v2) Jacobians (optional):

- `/jacobian_9D/J_hat` – `float32[W2, K, K]`  
  - Windowed local Jacobian estimates in Stratified subcoordinate space (e.g. `K=9` for full Stratified MNPS).
- `/jacobian_9D/J_dot` – `float32[W2-1, K, K]`  
  - Temporal differences between successive Stratified Jacobians.
- `/jacobian_9D/centers` – `int32[W2]`  
  - Indices into `/time` of the centre of each Stratified Jacobian window.
- `/jacobian_9D/cross_partials/*` (optional) – `float32[W2]`  
  - Selected Jacobian elements `J_{out,in}(t)` saved as time series when enabled via `mnps_v2.save_cross_partials`.
  - When enabled via a preset policy, the *selection rationale* is recorded in the JSON manifest (see below):
    - `preset` (e.g. `ndt_core_v1`, `ndt_core_plus_diag_v1`)
    - `core_pairs` (fixed, theory-driven set)
    - `extra_pairs` (explicit dataset-specific additions)
    - `rationale` (free text; prereg/Methods traceability)

> The 3D Jacobian (`/jacobian`) and 9D Stratified Jacobian (`/jacobian_9D`) can coexist. Both use the same time base, but may have different valid window counts (`W` vs `W2`).

Extended coordinates (optional, EEG‑first in the current pipeline):

- `/extensions/e_kappa/*` – energetic curvature series derived from EEG‑based energy `E(t)`  
  - `time`: `float32[T]` – aligned to `/time`  
  - `energy`: `float32[T]` – scalar energy per MNPS window (e.g. weighted EEG bandpower)  
  - `kappa`: `float32[T]` – energetic curvature `κ_E(t)` for each window.
- `/extensions/rfm/*` – resonant phase modes computed on per‑epoch EEG band trajectories  
  - `times`: `float32[W_r]` – RFM window centres (seconds)  
  - `eigvals`: `float32[W_r, C]` – eigenvalues of the phase‑coherence matrices  
  - `eigvecs`: `float32[W_r, K, C]` – top‑`K` eigenvectors (RFM modes) per window  
  - `dominance`: `float32[W_r]` – relative dominance of the leading mode.
- `/extensions/o_koh/*` – organisational coherence from EEG functional connectivity  
  - `thresholds`: `float32[L]` – filtration thresholds on |C_ij|  
  - `beta0`, `beta1`: `float32[L]` – graph‑based approximations of Betti‑0/1 over thresholds  
  - `OKoh0`, `OKoh1`: `float32[]` – scalar summary indices.
- `/extensions/tig/*` – temporal integrity grade on the MNPS trajectory `x(t)`  
  - `lags_sec`: `float32[L]` – lags used for autocorrelation estimation  
  - `autocorr`: `float32[L]` – normalized autocorrelation C(Δ)  
  - `tau`: `float32[]` – decay time constant (seconds), clipped to `T_max`  
  - `TIG`: `float32[]` – normalized temporal integrity grade `τ / T_max` in `[0, 1]`  
  - `provisional`: `bool` – `true` when `tau` had to be saturated (ill‑conditioned or beyond `T_max`).

---

## JSON Summary (per subject / session)

Each subject/session gets a `summary.json` generated by `json_writer.build_manifest`. The structure is:

```json
{
  "dataset_id": "ds003490:sub-001:ses-01",
  "samples": 1234,
  "mnps": {
    "fs_out": 4.0,
    "window_sec": 8.0,
    "overlap": 0.5
  },
  "meta_indices": {
    "mean_trace": -0.12,
    "mean_rotation_fro": 0.45,
    "windows": 210
  },
  "events": ["so", "spindle", "task"],
  "jacobian": {
    "windows": 210,
    "with_centers": true
  },
  "coords_9d": {
    "names": ["m_a", "m_e", "m_o", "d_n", "d_l", "d_s", "e_e", "e_s", "e_m"],
    "groups": {
      "m": ["m_a", "m_e", "m_o"],
      "d": ["d_n", "d_l", "d_s"],
      "e": ["e_e", "e_s", "e_m"]
    }
  },
  "meta_indices_v2": {
    "mean_trace": -0.08,
    "mean_rotation_fro": 0.37,
    "windows": 205
  },
  "jacobian_9D": {
    "windows": 205,
    "with_centers": true
  },
  "...": "additional fields (participant_meta, group, condition, task, robustness, multiverse, entropy_qc)"
}
```

Key sections:

- **mnps** – basic time base parameters (`fs_out`, `window_sec`, `overlap`).
- **meta_indices** – aggregate indices computed from the primary MNPS Jacobian (`/jacobian/J_hat`):
  - `mean_trace`: mean divergence (expansion vs contraction).
  - `mean_rotation_fro`: mean Frobenius norm of the rotational part.
  - `windows`: number of Jacobian windows.
- **jacobian** – high-level info about the primary Jacobian tensor (window count, presence of centres).
- **coords_9d** (optional) – present when Stratified MNPS v2 is enabled and successfully computed.
- **feature_exports** (optional) – present when H5 embeds `/features_raw/*` and `/features_robust_z/*`.
- **meta_indices_v2** (optional) – same meta-indices as above, but computed from the Stratified Jacobian (`/jacobian_9D/J_hat`).
- **jacobian_9D** (optional) – high-level info about the Stratified Jacobian tensor.
- **jacobian_9D_cross_partials** (optional) – present when `mnps_v2.save_cross_partials.enabled=true`:
  - `preset`: optional preset name used to expand a stable “core” set
  - `core_pairs`: list of core `[out, in]` pairs (dataset-agnostic)
  - `extra_pairs`: list of dataset-specific extra pairs (hypothesis-driven)
  - `rationale`: optional free text justification (recommended for prereg/Methods)
  - `pairs`: the resolved final pairs list actually saved to HDF5 under `/jacobian_9D/cross_partials/*`
  - `items`: summary stats per extracted series (mean/std/abs_mean)
- **stage_codes** (optional) – added when stage labels are present:
  - `unique`: list of unique int codes in `/labels/stage`.
  - `codebook`: mapping from stage labels → codes.
- **robust_summary / ensemble_robustness / multiverse_psd / entropy_qc** – optional blocks providing robustness and multiverse diagnostics for MNPS and Stratified MNPS subcoordinates.

---

## Notes

- Keep keys stable across datasets to simplify downstream analysis.
- Subject-level JSON manifests are designed so that analysis code can:
  - Discover the presence/absence of Stratified MNPS v2 and Stratified Jacobians.
  - Read basic MNPS configuration and meta-indices without opening the HDF5 file.
  - Access robustness and QC metadata for filtering and weighting subjects in downstream MNPS/MNJ analyses.


