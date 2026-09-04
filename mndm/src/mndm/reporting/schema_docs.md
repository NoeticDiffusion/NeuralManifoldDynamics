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
- **schema_version / mndm_version**: `mnps_tensor_spec_v2_1` / `2.1` when explicit anchored coordinate layers or embedded feature anchors are present.
- **primary_coordinate_layer**: H5 group to use as the primary 3D coordinate layer, e.g. `coords_3d_cohort_anchored`.
- **primary_coordinate_contract**: `subject_anchored` or `cohort_anchored`.
- **anchor_id / anchor_hash**: set when cohort/external anchoring is active.
- **geometry_invalidity_policy / geometry_contract_status**: always-on canonical validity contract for MNPS/MNJ geometry.

## Standard Geometry Invalidity Contract

Canonical MNDM exports now distinguish between:

- **`geometry_contract`**: always-on machine-readable contract for mathematically invalid rows/windows that were dropped or retained as degraded surfaces
- **`geometry_contract.time_grid`**: always-on realized time-base audit (`dt`, `window_len`, finite/non-positive bound checks)
- **`mnps_mnj_sanity`**: optional reviewer-facing QA block under `review_qc`, including derivative self-consistency between finite differences and `/mnps_3d_dot`

Use `geometry_contract` to decide whether a subject/run is mathematically valid for downstream MNPS, `coords_9d`, Jacobian, or reachability-style analyses. Invalid geometry is not clamped; rows or Jacobian windows are removed from canonical export and counted under this contract.

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
  - Guarded exports include `robust_z_valid`, `robust_z_invalid_reason`, and
    `robust_z_finite_count`. With the default `nan` policy, insufficient
    support or a MAD-degenerate scale yields an all-NaN strict robust-z column.
    `features_projection_z` and coordinate contracts remain separate.
- `/coords_3d_subject_anchored/values` – `float32[T, 3]`
  - Subject/session-anchored 3D coordinates for within-subject geometry.
- `/coords_3d_subject_anchored/names` – `str[3]`
  - Coordinate names, usually `[m,d,e]`.
- `/coords_9d_subject_anchored/values` – `float32[T, K]`
  - Subject/session-anchored stratified coordinates when available.
- `/coords_3d_cohort_anchored/values` – `float32[T, 3]`
  - Cohort/external-anchored 3D coordinates for group comparisons when an anchor is configured.
- `/coords_9d_cohort_anchored/values` – `float32[T, K]`
  - Cohort/external-anchored stratified coordinates when available.
- `/feature_anchors/spec` – attrs
  - Anchor identity/source/hash and scale policy.
- `/feature_anchors/per_feature/*` – arrays of length `K`
  - Per-feature center/scale/quantile/support statistics used by cohort-anchored coordinates.
- `/anchor_state/values` (optional) – `float32[T, Qa]`
  - Noetic Anchoring Dynamics aligned anchor-state matrix on the MNPS time grid.
- `/anchor_state/names` (optional) – `str[Qa]`
  - Column names for `/anchor_state/values`.
- `/anchor_state_dot/values` (optional) – `float32[T, Qa]`
  - Derivative or finite-difference anchor-state matrix aligned to `/time`.
- `/anchor_quality/values` (optional) – `float32[T, Qq]`
  - Anchor signal-quality / support matrix aligned to `/time`.
- `/anchor_quality/names` (optional) – `str[Qq]`
  - Column names for `/anchor_quality/values`.
  - The backward-compatible `mndm.anchor_quality.v1` contract may have
    `quality_surface = "v2"` attrs and additive `<anchor>_eligible`,
    `<anchor>_valid`, and `anchor_valid_fraction` columns. Invalid guarded
    components/composites are represented as `NaN` in `/anchor_state/values`.
- `/anchor_coupling/*` (optional)
  - Additive body-brain coupling diagnostics kept separate from canonical `/jacobian/*`.
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
- `/jacobian/derived_metrics/v1` (optional)
  - `mndm.jacobian_metrics.v1` per-window semantic measurements plus support-counted summaries and provenance. It derives from the exported continuous-time Jacobian and does not change the estimator.

Stratified MNPS v2 (optional):

- `/coords_9d/values` – `float32[T, 9]`  
  - Time series of Stratified subcoordinates in canonical order  
    `[m_a, m_e, m_o, d_n, d_l, d_s, e_e, e_s, e_m]`.
- `/coords_9d/names` – `str[9]`  
  - Names of the subcoordinates (always in canonical order after normalization).
- `/coords_9d` group attrs:
  - `version = "9d"`

MNDM 2.1 coordinate contract:

- `subject_anchored` layers preserve the current within-subject/session geometry and are the appropriate input for local trajectory-shape interpretation.
- `cohort_anchored` layers are computed with a frozen feature anchor and are the preferred layer for clinical group contrasts.
- `feature_anchors` means cohort/external feature scaling only. It is not the same concept as `/anchor_state`, which represents embodied/interoceptive state aligned to the MNPS grid.
- The root `primary_coordinate_layer` attr tells downstream code which layer the run declares as primary.
- `run_manifest.json` exposes capability flags for `feature_anchors`, `anchor_state`, `anchor_quality`, `anchor_coupling`, `coords_3d_subject_anchored`, `coords_3d_cohort_anchored`, `coords_9d_subject_anchored`, and `coords_9d_cohort_anchored`.

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

Local-dynamics groups (all optional) are `/finite_time_response/v1`,
`/residual_covariance_proxy/v1`, `/transition_residuals/v1`, and
`/stochastic_reachability/v1`. They record their capability schema,
coordinate-layer provenance, validity status, and time semantics; computed
response or reachability is not evidence of predictive or perturbational
validity by itself. Round 2 adds sibling certificate fields
`computation_status`, `measurement_validity`, and `claim_status`.
`validation_level` is a method-validation tag, not regime validity. New
writes set `claim_status` to `no_biological_claim` and do not emit
`ndt_licensed`. Jacobian metrics that are finite remain
`measurement_validity=not_assessed`. They are provenance on `J_hat`, not
S3-licensed empirical NDT \(\alpha/\omega\). FTR `g_peak_over_horizons` is
the serialized \(G_{\mathrm{peak}}\) analogue; it is not a jacobian_metrics
field and is not licensed empirical NDT. I-CARE / analysis-repo coma
reachability is not `/stochastic_reachability/v1`.

Round 3 adds nested `grain/` (`native`, `parent`, `biological_unit`,
`repeated_measure`, `direct_between_subject_inference`). New writes set
`biological_unit=subject` and `direct_between_subject_inference=forbidden`.
A window is not a participant. Grain is present even when the object is not
computed. Legacy readers fill missing grain fields as `not_recorded` and must
not infer `window` from series.

Round 4 adds `/support_signature/v1/` (`mndm.support_signature.v1`): per-9D
coordinate `source` (`direct` / `fallback`) from existing metric policies
and recorded substitutions, plus one modality capability row
(`chart_3d`/`mnj_*`/`spread=gated`/`resilience=perturbational_only`). This
is metadata, not a coverage estimator and not `geometry_contract`.
`coords_9d` present is not the same feature support and not the same
modality capability. Legacy missing fields are `not_recorded`; do not infer
`chart_3d=yes` from `/mnps_3d` or `spread=gated` from absent reachability.

Standard non-MNPS measurement families, when explicitly requested, are
written only under `/dynamical_families/{diffusion,destination,resilience}/v1`.
They do not alter `/mnps_3d`, `/coords_9d`, or `/jacobian`. Schema IDs remain
`mndm.diffusion_geometry.v1`, `mndm.committor.v1`, and
`mndm.finite_amplitude_resilience.v1`. Pre-v3 files may still contain
`/orthogonal_dynamics/*`; readers load that tree only when the canonical
group is missing. The diffusion, committor, and finite-amplitude-resilience
contracts each carry their own support/status, dataset-eligibility, and
translation-qualification requirements. Diffusion, when enabled, is computed
if the estimator has support and is written `not_assessed`; OD-TQ1 id/hash
are provenance method tags, not a compute gate. Ingest diffusion uses
C1 defaults (`drift=None`, `residualize_increments=False`): `computed` is
raw increment covariance (`a_semantics=raw_increment_covariance`), not
testable \(A_{bD}\) / \(R_{b/a}\) (`summary.A_bD_computation_status=not_testable`,
`summary.R_b_over_a_computation_status=not_testable`,
`independent_drift_not_supplied`; NaN is not zero alignment). Library C1
may consume an externally qualified chart \(b\) without changing `a_hat`.
Ingest does not estimate \(b\) (SL-005; nmd-analysis). C2
residualization is not authorized.
`contract_status=standard` is the schema
class, not an empirical license. Destination and resilience
are `measurement_validity=translation_qualified` only when computed with a
TQ id and contract hash already recorded; otherwise new writes use
`not_assessed` or `not_applicable`. Production destination is 1-D O2b;
neither O2b nor the first-hit estimator serializes \(V_{1/2}\) or
\(\lvert\nabla q\rvert\). Spread as a **family YAML / `/dynamical_families` key**
(`stochastic_reachability.v1`) remains registry `gate_closed` and is not
written there. Opt-in ingest `W_Q` uses `/stochastic_reachability/v1` after
the Gate F freeze (Gate E Q + crossfit \(\Phi\)). Legacy readers must fill missing
certificate fields as `not_recorded` and must not infer
`translation_qualified`. Missing `claim_status` may be copied from
`provenance/claim_status` when that dataset exists; otherwise it is
`not_recorded`. Do not default a missing claim to `no_biological_claim`.

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
  Includes `robust_z_guard` with the strict-export policy, minimum support,
  scale floor, and policy version.
- **anchor_state_validation** (optional) – per-anchor finite count, IQR,
  maximum absolute magnitude, max/IQR, invalid count, warnings, thresholds,
  and the guard-policy version. It is warning-only unless configured as
  blocking for smoke/CI.
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

## Cross-Modal Interpretation Note (MEG vs EEG)

MEG and EEG share the NMD coordinate contract, but individual subcoordinates may not
have identical sign semantics across modalities.

In ds003645 (Wakeman-Henson face perception, simultaneous MEEG, 18 subjects):

- **Hjorth mobility** (`d`-family): MEG shows a fixed face < scrambled direction across
  all 18 subjects. EEG polarity is variable — 14/18 subjects show the opposite sign to
  MEG, and the inversion is consistent across MAG sensors, GRAD sensors, EEG-from-FIF,
  and EEG-from-.set. It survives QC filtering.

- **Hjorth complexity** (`d`-family): MEG shows a fixed face > scrambled direction across
  all 18 subjects. EEG polarity is again variable.

The d-family inversion has been validated as a **modality-specific signal-space
difference**, not an artifact. The most likely cause is the contrast between
reference-free MEG field measurement and scalp-referenced EEG potentials, combined
with volume conduction mixing in EEG.

**Therefore, cross-modal comparison should:**

1. Prioritize **modality-internal validity** (does MEG separate the task within itself?).
2. Use **family-level diagnostics** (m-, d-, e-family separately) rather than a single
   aggregate cosine.
3. Apply **null-controlled convergence** (label-shuffle and wrong-run nulls) rather than
   raw sign equality.
4. Not assume that Hjorth mobility/complexity have the same directional semantics in MEG
   and EEG.

**Current status (ds003645, `meg_shadow_v0_usable`):**

| Claim | Status |
|-------|--------|
| MEG engineering readiness | 0.9419 — production-ready |
| MEG-internal task separation | 15/18 subjects (p < 0.10) |
| EEG-MEG convergence | 0.4786 — weak; d-family diverges |
| d-family inversion | Validated modality-specific finding |

---

## Notes

- Keep keys stable across datasets to simplify downstream analysis.
- Subject-level JSON manifests are designed so that analysis code can:
  - Discover the presence/absence of Stratified MNPS v2 and Stratified Jacobians.
  - Read basic MNPS configuration and meta-indices without opening the HDF5 file.
  - Access robustness and QC metadata for filtering and weighting subjects in downstream MNPS/MNJ analyses.

---

## Sleep-EAP Phase 2 extension contracts

The RichSleep Phase 2 overlay may add two independent, versioned extension
groups. Neither is aligned to the MNPS epoch grid or part of the coordinate
contract.

- `/extensions/phase_continuous_v1`: columnar continuous phase data with
  `timestamp_sec`, `phi_cardiac`, `phi_resp`, and quality/validity columns.
  Manifest fields under `phase_continuous` report the source sidecar, contract,
  sample count, and embedding status.
- `/extensions/non_event_risk_v1`: columnar risk timestamps with subject,
  stage, time-of-night quartile, phase values, selection seed, and exclusion
  margin. Manifest fields under `non_event_risk` record the contract, source
  sidecar, row count, and embedding status.

`event_phase_v3` is a Parquet sidecar contract rather than a mandatory H5
group. It augments the catalog-filtered v2 event-phase rows with raw-EEG sigma
strength, YASA provenance, and SO-spindle relationship columns. Missing SO
partners are explicitly represented by NaN metrics plus a missing-partner flag.

`event_phase_n3_so_v1` and `event_phase_rem_theta_v1` are separate Parquet
contracts. The N3-SO table has one N3-gated detector event per row and samples
SO, cardiac, and respiratory phases at trough and up-state reference points.
The REM-theta table has one scored REM epoch per row and samples theta and
autonomic phase at its 30-s midpoint. Both retain NaN for unavailable
autonomic phase and record explicit validity/QC flags. Neither schema permits
spindle-strength, sigma, or SO-spindle-pairing columns.

These products make phase and endogenous event-strength variables available for
downstream qz(t)/bqg and hazard analyses; they do not themselves test or
validate those models.


