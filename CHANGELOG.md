# Changelog

---

## Unreleased — DF-DRIFT-C1 ingest freeze and analysis handoff (SL-005)

Documentation only. No measurement change. MNPS `[m,d,e]` / 9D, `J_hat`,
and family schema IDs are unchanged. Ingest still writes `drift=None`.

- SL-004 RATIFY of C1 M1–M2 stands. SL-005 freezes the ingest line at
  consume-external-`b` and moves M3+ to nmd-analysis.
- Handoff: `project/mnps_v3/df_drift_c1_analysis_handoff.md`
  (**RATIFIED** 2026-09-03). MNPS chart trajectory, not “raw”; analysis
  supplies auditable split hashes; ingest records an externally governed
  qualification without adjudicating M3 PASS; Gate F
  `input_semantics=discrete_transition_residual_covariance`.
- C2 residualization, empirical C1 overlays, common-profile enablement,
  and ds004100 OD-EPI-001 modification remain not authorized.

---

## Unreleased — DF-DRIFT-C1 M1–M2 alignment-only estimator split

Science-lead SL-003: C1 `alignment_only` is authorized for synthetic
M1–M2 qualification only. C2 residualization, empirical C1 overlays,
common-profile enablement, and ds004100 OD-EPI-001 modification are not
authorized. MNPS `[m,d,e]` / 9D, `J_hat`, and family schema IDs are
unchanged.

- `estimate_local_diffusion_geometry` now splits alignment `drift` from
  `residualize_increments` (default false). Supplying truth-known chart
  \(b\) for alignment leaves `a_hat` / `D_total` / `d_diff` / `c_diff`
  byte-equal to the no-drift path (Gate C1-A). The C2 flag is
  `invalid` / `c2_residualize_increments_not_authorized`.
- Ingest still does not supply a vector and never residualizes. Forbidden
  sources (`mnps_xdot`, Jacobian intercept, same-sample increment mean)
  fail closed; `a_hat` stays on the no-drift path.
- Under C1, `a_semantics=raw_increment_covariance` and
  `ratio_semantics=chart_velocity_to_increment_spread`. \(R_{b/a}\) is
  secondary; it is not an Itô drift-to-diffusion ratio.
- FAR-EXT-002E is parked `BLOCKED_EXTERNAL_CUSTODY`. No live 0.2, freeze,
  \(R(\rho)\), or FAR-003 until complete E1+E2 XML arrives.

---

## Unreleased — Gate F admissible Q and opt-in \(W_Q\)

Gate F freeze: the only admissible Q for computed discrete \(W_Q\) is the
Gate E recording-level transition-residual covariance. MNPS `[m,d,e]` / 9D,
`J_hat`, and family schema IDs are unchanged.

- Admissible tags: `q_time_semantics=one_step_transition_covariance`,
  `q_units=state_squared`, `conversion_model=not_applicable`.
- Derivative-residual covariance remains refused
  (`q_contract_not_admissible`). Irregular \(dt\) keeps Q and \(W_Q\)
  `unavailable` / `materially_irregular_dt`.
- Propagators are Gate E `series/phi_one_step` =
  \(\mathrm{expm}(J_{\mathrm{crossfit}}\Delta t)\), not canonical `J_hat`.
- Opt-in `local_dynamics.stochastic_reachability.enabled` (default false)
  writes `/stochastic_reachability/v1`. Family YAML `spread` stays
  `gate_closed`. Registry forbid `gate_e_proxy_as_process_noise` stays.
- Computed writes: `measurement_validity=not_assessed`,
  `claim_status=no_biological_claim`, grain `recording_horizon`.
- Common EEG/fMRI/ephys profiles do not enable reachability or FTR.
  I-CARE `tube_d_eff_median` is not this schema.

---

## Unreleased — v3 B/D docs: names, \(A_{bD}\) status, claim ceilings

Documentation catch-up plus explicit ingest status for drift-alignment
scalars. MNPS `[m,d,e]` / 9D, `J_hat`, and family schema IDs are unchanged.

- `contract_status=standard` is the schema class, not an experimental vs
  licensed scientific tag.
- Theory \(G_{\mathrm{peak}}\) serializes as FTR `g_peak_over_horizons`.
- Ingest diffusion remains `drift=None`. `computed` increment covariance is
  not testable \(A_{bD}\) / \(R_{b/a}\). Those series stay NaN and now carry
  `summary.A_bD_computation_status=not_testable` /
  `R_b_over_a_computation_status=not_testable` with
  `drift_alignment_failure_reason=independent_drift_not_supplied`.
- Registry forbids `mnps_xdot_as_sde_drift` in addition to Jacobian-residual
  substitution as \(a\).
- O2b and first-hit committor do not serialize \(V_{1/2}\) or
  \(\lvert\nabla q\rvert\).
- v2.6 jacobian_metrics / FTR are provenance, not S3-licensed empirical NDT.
  I-CARE analysis reachability is not `mndm.stochastic_reachability.v1`.

---

## Unreleased — diffusion compute vs OD-TQ1 method tag

When `dynamical_families.diffusion` is enabled, ingest computes increment
covariance if the estimator has support. YAML OD-TQ1 id/hash are provenance
method tags. They are not an on/off gate and do not set
`measurement_validity=translation_qualified`.

- Write: `computed` + `not_assessed` + `no_biological_claim` when support holds.
- Estimator refusals (gaps, irregular \(dt\), insufficient samples) remain
  `not_testable` / `insufficient_support`.
- Destination and FAR still require protocol inputs plus an adapter stamp.
- MNPS `[m,d,e]` / 9D, `J_hat`, and family schema IDs are unchanged.

---

## Unreleased — v3 Round 4 support / capability

Additive metadata schema `/support_signature/v1/`
(`mndm.support_signature.v1`). MNPS `[m,d,e]` / 9D, `J_hat`, family schema
IDs, and `geometry_contract` are unchanged.

- Per-coordinate `source` / `fallback_feature` / `semantic_equivalence`
  from existing `metric_policies` and recorded fallback metadata.
- File-level capability row by modality (`chart_3d=yes` for EEG/iEEG/fMRI;
  `spread=gated`; FAR `resilience=perturbational_only`; fMRI MNJ `limited`).
- Legacy readers fill missing fields as `not_recorded` and do not infer
  capability from HDF5 presence.

---

## Unreleased — v3 Round 3 inferential grain

Additive nested `grain/` on existing v1 groups. MNPS `[m,d,e]` / 9D, `J_hat`,
and schema IDs are unchanged.

- Fields: `native`, `parent`, `biological_unit`, `repeated_measure`,
  `direct_between_subject_inference`.
- New writes set `biological_unit=subject` and
  `direct_between_subject_inference=forbidden`.
- Grain is schema metadata and is written even when the object is not
  computed. Windows are not participants.
- Legacy readers fill missing grain fields as `not_recorded` and do not infer
  `window` from series.

---

## Unreleased — v3 Round 2 validity certificate

Additive certificate fields on existing v1 groups. MNPS `[m,d,e]` / 9D,
`J_hat`, and family schema IDs are unchanged.

- Sibling fields `computation_status`, `measurement_validity`, and
  `claim_status` on dynamical-family and local-dynamics exports.
- New writes set `claim_status` to `no_biological_claim`; `ndt_licensed` is
  not emitted.
- `measurement_validity=translation_qualified` only when a family TQ id and
  contract hash are already recorded. Jacobian metrics and FTR remain
  `not_assessed` when computed.
- `validation_level` is not `measurement_validity`.
- Legacy readers fill missing fields as `not_recorded` and do not infer
  regime validity or NDT licensing.

---

## Unreleased — v3 Round 1 semantic kernel (dynamical families)

Container and language rename only. MNPS `[m,d,e]` / 9D order, the Jacobian
estimator, and family schema IDs are unchanged.

- Python package `mndm.orthogonal_dynamics` → `mndm.dynamical_families`.
- YAML root `orthogonal_dynamics` → `dynamical_families` (old key refused).
- HDF5 write path `/dynamical_families/{diffusion,destination,resilience}/v1`.
- Legacy `/orthogonal_dynamics/` is read-only; new files do not dual-write.
- Schema IDs remain `mndm.diffusion_geometry.v1`, `mndm.committor.v1`,
  `mndm.finite_amplitude_resilience.v1`.
- Historical protocol IDs (`OD-TQ*`, `OD-EPI-*`, `OD-SLP-*`, `FAR-*`) kept.
- Spread / Gate F `W_Q` remains closed.

---

## v2.6.0 — Local Jacobian metrics and finite-time response

This release adds a versioned local-dynamics interpretation layer on top of the
unchanged MNPS Jacobian estimator. It remains a measurement/export release:
cohort comparisons, retrospective audits, and biological claims stay in the
analysis repository.

### Major additions

**Jacobian Metrics v1**

- Added `mndm.jacobian_metrics.v1` for 3D and directly estimated 9D Jacobian
  fields, including spectral/numerical abscissae, reactivity, deformation,
  rotation, support counts, and explicit stable-reactive flags.
- Added `/jacobian/derived_metrics/v1/` and
  `/jacobian_9D/derived_metrics/v1/`, carrying coordinate-contract provenance.

**Finite-Time Response v1**

- Added opt-in `mndm.finite_time_response.v1` with time-ordered exponentials,
  original-center and observed-time continuity gates, actual horizon duration,
  support counts, and distinct computation/validation status.
- Added 9D family-transfer summaries only for truly 9D fields; no 3D fallback
  is synthesized.

**Residual covariance and reachability boundary**

- Added residual-covariance and stochastic-reachability library contracts,
  including PSD/QC and explicit time semantics.
- The summarize pipeline does not yet emit a one-step transition residual
  covariance, so it does not claim computed stochastic reachability.

### Validation and claim boundaries

| Stream | Evidence | Claim ceiling |
|---|---|---|
| Local-dynamics unit and writer contracts | Analytic matrices, continuity and Q-gate tests | Mathematical/export contract only |
| ds003944 / ds003947 reruns | Model-derived local measurements | Internal descriptive output; not biological inference |
| Finite-time response | `validation_level = model_derived` | Not held-out predictive or perturbational validation |

### Release boundary

v2.6.0 is identified by its final commit and package version, not by generated
dataset outputs. The Jacobian estimator semantics are unchanged, and
unavailable Q/reachability inputs remain explicit gates rather than inferred
noise models.

---

## v2.5.0 — Multimodal ingest expansion, phase-aware features, and contract hardening

This release consolidates the post-v2.4.0 working-tree additions into the next
versioned measurement-contract line. The detailed MEG, HRV v0.1, block-native
v2, geometry-contract, anchoring, and validation notes retained under v2.4.0
below remain part of the cumulative contract history. This section records the
additional functionality promoted into the v2.5.0 release.

### Major additions

**BDF / Figshare infant EEG**

- Added BDF adapters and privacy-safe metadata extraction for Figshare infant
  EEG workflows.
- Added BAD_ masking, coverage-aware cohort-anchor export, and a repair for
  connectivity features becoming all-NaN after masked segments.
- The full internal run processed 71 BDF files, retained 5,273 epochs, and
  produced 70 coverage-passing HDF5 summaries. These are ingest/QC results;
  infant behavioural condition labels were not established by the available
  Status field.

**NWB / Neuropixels / ephys and LFP**

- Added DANDI/NWB ElectricalSeries and Units-to-rate paths with probe discovery,
  streaming QC, state/stimulation annotations, and explicit geometry limits.
- Added Neuropixels ephys feature extraction and smoke validation for DANDI
  000006.
- Added LFP channel selection, contact/reference sensitivity, and state-aware
  QC for DANDI 000458. These are one-session descriptive foundations, not
  cross-probe geometric or circuit-level validation.

**Phase anchor and sleep-EAP extensions**

- Added optional cardiac/respiratory phase, RR, HR, respiratory-rate, inhale
  fraction, and HEP-related feature extraction through `phase_anchor`.
- Added the sleep-EAP phase-2 contract and associated quality/provenance
  surfaces.
- These features are now release-bound as optional v2.5.0 capabilities; their
  dataset-scale scientific interpretation remains modality- and cohort-
  dependent.

**Contract, provenance, and QC hardening**

- Added `/epoch_id` as an explicit window join key where available.
- Added stricter simultaneous-MEEG row-lineage checks and expanded provenance.
- Added robust-z degenerate-scale safeguards and documented the
  `degenerate_scale_policy = "nan"` compatibility boundary.
- Added configuration overlay replacement semantics and additional source
  adapters, tests, and QC artifacts.

### Validation and claim boundaries

| Dataset or stream | Evidence in v2.5.0 | Claim ceiling |
|---|---|---|
| ds003838 | 130 completed HDF5 exports and 27,670 block-native windows; corrected stage statistics use `N = 62`, with `vagal_index` listen--mem13 `d = 1.995` | Internal task-contrast validation; no independent replication |
| ds006848 | ECG polarity and contamination audits; encoding `m`/`d` analyses on `n = 30` | Working-memory HRV claims withheld because 87.7% of 60 s windows overlap retrieval |
| ds003645 | Five-subject MEG pilot, readiness `0.7879` | Exploratory pilot; not full-cohort production validation |
| DANDI 000006 / 000458 | Units and LFP smoke/QC paths | Transport and descriptive sensitivity only |
| Figshare infant EEG | BDF ingest, masking, connectivity repair, and cohort outputs | No infant behavioural condition or clinical inference |

### Release boundary

The v2.5.0 release must be identified by the final commit and package version,
not by the presence of local generated data. Exploratory, negative, and
dataset-specific findings remain labelled as such. The v2.4.0 section below
retains the historical release surface that v2.5.0 extends.

---

## v2.4.0 — MEG support, HRV v0.1, block-native v2, geometry contract

### Major additions

**MEG ingest (exploratory/beta)**

- `meg_mag_*`, `meg_grad_*`, `meg_*` feature columns extracted from Neuromag FIF files via MNE-Python.
- Shadow mapping routes MEG features through the existing 9D coordinate contract — each `meg_*` type maps to the same subcoordinate slot as its `eeg_*` counterpart; no changes to the projection machinery.
- For simultaneous MEEG recordings: explicit row-source provenance under `row_source/` (schema: `mndm.row_source.v1`), replacing the implicit positional half-split assumption.
- New `features_projection_z` HDF5 export surface applies the configured transform pipeline (log10 → robust-z → clip) before export. Required for MEG spectral features, where raw physical power (~10⁻²⁵ W) collapses to near-zero under raw-space robust-z.
- Validated on a 5-subject pilot from ds003645 (readiness score 0.7879). Labeled **exploratory** — not yet confirmed at full 18-subject scale.

**Embodied Anchoring — concrete modality implementations** *(v2.3 introduced the principle)*

- *ECG / HRV v0.1*: Superwindow time-domain surface (`ecg_hrv_hr_mean_bpm`, `ecg_hrv_ibi_mean_ms`, `ecg_hrv_sdnn_ms`, `ecg_hrv_rmssd_ms`, `ecg_hrv_pnn50`, `ecg_hrv_nn_count`, plus artifact/coverage/quality flags) via configurable centered window (default 60 s). Optional complexity columns (`ecg_hrv_sampen`, `ecg_hrv_dfa_alpha1`) when enabled. Manifest tag: `anchor_hrv_v0_1`. Note: frequency-domain metrics (HF power, LF/HF) are not part of the v0.1 surface.
- *PPG surface*: Per-epoch rate, amplitude, variability, and quality flags when PPG channels are present; feeds `vascular_index` in `anchor_state`.
- *Pupillometry surface*: Per-epoch diameter, volatility, blink-rate proxy, and quality score when pupil traces are present.
- Automatic ECG polarity correction (validated on ds006848: 92.7% of epochs had inverted polarity; after correction, population median HR 76.2 bpm, RMSSD 40.7 ms).
- HRV contamination gating: `ecg_hrv_*` columns carry contamination flags when the superwindow overlaps task events.

**Block-native v2 sidecar ecosystem**

- `block_native_qc.json`, named window profiles, `source_window_index` provenance.
- Built-in parquet/CSV sidecars alongside HDF5.
- Inter-network Jacobian coupling columns (`coupl_*`) in block-native sidecars; stage-level pooling fallback for short-trial datasets.

### Other additions

- `anchor_auto_fit`: one-shot per-run cohort anchor fitting — resolves most `cohort_anchored` skip cases without manual anchor preparation.
- `standard_invalidity_v1` geometry contract: versioned policy for `coords_9d` duplicate-subcoordinate tolerance with per-subject diagnostics; always-on time-grid contract auditing.
- `participants.extra_tables`: generic clinical TSV join (UPDRS items, longitudinal tables) embedded into per-subject H5 output. Demonstrated on ds007526.
- Conventional EEG coma pack extended: suppression ratio, burst-suppression proxy, spectral ratios, and reactivity proxies for clinical ICU datasets.
- `openneuro_ingest` downloads now run through `uvx openneuro-py@latest` by default (`download.use_uvx`, default true; `OPENNEURO_PREFER_UVX` env override). Works around installed openneuro-py versions broken by upstream OpenNeuro GraphQL schema changes (e.g. 2026.3.0 `Cannot query field "key" on type "DatasetFile"`). `uv` added to `requirements.txt`; `openneuro-py` pin bumped to `>=2026.4.1`.

### Production validation — six additional cohorts

| Dataset | n | Notes |
|---------|---|-------|
| ds003838 | 130 subjects | HRV v0.1 + block-native; 27,670 block windows; vagal_index listen–mem13 Cohen's d = 1.995 (Wilcoxon p = 1.2×10⁻¹⁰) |
| ds006036 | 88 subjects | Block-native |
| ds007526 | 277 recordings | Parkinson gait/rest + clinical TSV join |
| ds003490 | 75 subjects | Dual-anchor rerun |
| ds003506 | 84 subjects | Dual-anchor rerun |
| ds003509 | 75 subjects | Dual-anchor rerun |

---

## v2.3.0 — Embodied Anchoring Principle, event/block layers

### Major additions

- **Embodied Anchoring Principle (EAP)**: additive body-state surface (`anchor_state`, `anchor_state_dot`, `anchor_quality`) aligned to the same epoch grid as `mnps_3d`, without redefining the canonical chart. Optional `anchor_coupling` for downstream body-brain covariation diagnostics. Four index slots: `vagal_index`, `sympathetic_index`, `anchor_index`, `vascular_index`.
- **`geometry_contract`**: always-on mathematical validity reporting for canonical geometry exports.
- **`event_locked`**: generic derived analysis layer for short-event-centered questions.
- **`block_native`**: generic derived analysis layer for sustained-block / task-segment questions.
- **Explicit coordinate anchoring** (formalized from the 2.1 line):
  - `coords_3d_subject_anchored` / `coords_9d_subject_anchored`: preserves within-subject/session-relative geometry.
  - `coords_3d_cohort_anchored` / `coords_9d_cohort_anchored`: uses a frozen feature anchor for cross-subject and cross-group comparisons.
  - `/feature_anchors/*`: per-feature center/scale statistics with release-bound `anchor_id` / `anchor_hash` provenance.

### Other additions

- DANDI and PhysioNet ingest/download support.
- Sleep-spindle detection support (YASA-based annotation alignment).
- NWB and WFDB source format support.
- Conventional EEG comparator packs alongside the MNPS contract.
- HRV-oriented embodied-anchor features and task-segment-driven block-native export (demonstrated on ds003838).
- Regional EEG via channel-group trajectories (frontal, central, parietal-occipital, temporal) with optional CSD preprocessing (λ² = 1e-5, stiffness = 4.0).

---

## v2.1.0 — Explicit coordinate layers, MNDM contract formalization

Major change: MNDM 2.1 explicitly separates exported coordinate layers:

- `subject_anchored`: preserves subject/session-relative geometry.
- `cohort_anchored`: uses a frozen feature anchor for cross-subject and cross-group comparisons.

### Additional functionality

- Added DANDI and PhysioNet ingest/download support.
- Added sleep-spindle detection support.
- Added support for NWB and WFDB source formats.
