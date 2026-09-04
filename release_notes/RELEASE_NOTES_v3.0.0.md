# NeuralManifoldDynamics v3.0.0 — Dynamical families and measurement certificates

v3.0.0 is a measurement-contract generation on top of the frozen MNPS chart.
Canonical `mnps_3d = [m, d, e]`, stratified `coords_9d`, the Jacobian estimator
`J_hat`, and family schema IDs (`mndm.diffusion_geometry.v1`,
`mndm.committor.v1`, `mndm.finite_amplitude_resilience.v1`,
`mndm.jacobian_metrics.v1`, `mndm.finite_time_response.v1`,
`mndm.stochastic_reachability.v1`) are unchanged.

This release serializes non-MNPS family objects, validity/grain/support
certificates, and an opt-in discrete chart-space \(W_Q\) surface. It does not
license NDT \(\alpha/\omega/G_{\mathrm{peak}}\), claim a latent SDE, enable
families on common EEG/fMRI/ephys profiles, or move empirical C1-M3, observational
FAR, or committor overlays into ingest. Those remain in `nmd-analysis`.

Package version: `mndm.__version__ = "3.0.0"`.

## Highlights

**Dynamical families write surface (Round 1)**
- Python package `mndm.orthogonal_dynamics` → `mndm.dynamical_families`.
- YAML root `orthogonal_dynamics` → `dynamical_families` (old key refused; no alias).
- New writes land at `/dynamical_families/{diffusion,destination,resilience}/v1`.
- Legacy `/orthogonal_dynamics/` is read-only. New files do not dual-write.
- Historical protocol IDs (`OD-TQ*`, `OD-EPI-*`, `OD-SLP-*`, `FAR-*`) are kept.
- Common EEG/fMRI/ephys profiles do not import or enable this YAML root.

**Validity certificate (Round 2)**
- Sibling fields `computation_status`, `measurement_validity`, and `claim_status`
  on dynamical-family and local-dynamics exports.
- New writes set `claim_status=no_biological_claim`. `ndt_licensed` is not emitted.
- Jacobian metrics and finite-time response remain `not_assessed` when computed.
- Legacy readers fill missing fields as `not_recorded`.

**Inferential grain (Round 3)**
- Additive nested `grain/` on existing v1 groups.
- New writes set `biological_unit=subject` and
  `direct_between_subject_inference=forbidden`.
- Windows are not participants. Grain is schema metadata even when the object is
  not computed.

**Support / capability signature (Round 4)**
- Additive `/support_signature/v1/` (`mndm.support_signature.v1`).
- Per-coordinate source / fallback / semantic-equivalence metadata.
- File-level capability rows by modality (`chart_3d=yes` for EEG/iEEG/fMRI;
  `spread=gated`; FAR `resilience=perturbational_only`; fMRI MNJ `limited`).

**Gate F freeze and opt-in \(W_Q\)**
- The only admissible Q for computed discrete \(W_Q\) is the Gate E
  recording-level transition-residual covariance
  (`q_time_semantics=one_step_transition_covariance`,
  `q_source=transition_residual_covariance_proxy`). This is discrete
  chart-space predictive spread, not process noise.
- Opt-in `local_dynamics.stochastic_reachability.enabled` (default false) writes
  `/stochastic_reachability/v1`. Family YAML `spread` stays `gate_closed`.
- Derivative-residual covariance remains refused. Irregular \(dt\) keeps Q and
  \(W_Q\) unavailable. Common profiles do not enable reachability or FTR.
- I-CARE / analysis-repo coma tubes are not `mndm.stochastic_reachability.v1`.

**Diffusion compute vs method tags**
- When `dynamical_families.diffusion` is enabled, ingest computes increment
  covariance if the estimator has support (`computed` + `not_assessed` +
  `no_biological_claim`).
- YAML OD-TQ1 id/hash are provenance method tags, not an on/off gate and not
  `measurement_validity=translation_qualified`.
- Destination and FAR still require protocol inputs plus an adapter stamp.

**DF-DRIFT-C1 ingest freeze (consume-external-`b`)**
- Alignment-only estimator split: supplying an externally governed chart \(b\)
  for alignment leaves `a_hat` / `D_total` / `d_diff` / `c_diff` on the no-drift
  path. Ingest still writes `drift=None` and never residualizes.
- Forbidden sources (`mnps_xdot`, Jacobian intercept, same-sample increment mean)
  fail closed. C2 residualization is not authorized.
- Empirical C1-M3, observational FAR \(R(\rho)\), and committor overlays are
  analysis-owned. FAR-EXT-002E remains `BLOCKED_EXTERNAL_CUSTODY`.

## Validation and claim boundaries

| Dataset / stream | Evidence in v3.0.0 | Claim ceiling |
|---|---|---|
| MNPS / 9D / `J_hat` | Frozen from v2.6 | Chart coordinates and chart Jacobian; not a latent neural SDE |
| Dynamical families | Fail-closed serialization + certificates | Schema objects, not licensed NDT \(\alpha/\omega/G_{\mathrm{peak}}\) |
| ds004100 diffusion | 319/319 `a_hat` serialization | Chart-space increment covariance; not biology |
| Gate F \(W_Q\) | Opt-in discrete write when Gate E Q is admissible | Discrete chart-space predictive spread, not process noise or \(a(x)\) |
| C1 M1–M2 | Synthetic alignment-only split (SL-004) | Can consume external \(b\) without changing `a_hat`; not empirical \(A_{bD}\) |
| C1 M3 / C2 / FAR \(R(\rho)\) | Not in this ingest release | Downstream / blocked |

**Safe claims**

- MNPS is a chart; MNJ is a chart Jacobian.
- Families serialize fail-closed with explicit validity, grain, and support metadata.
- ds004100 can emit chart-space `a_hat` without a biological claim.
- Gate F \(W_Q\) is discrete chart-space predictive spread from
  Gate E `one_step_transition_covariance`, not Itô process noise.
- C1 can consume externally governed \(b\) without changing `a_hat`.

**Not established in v3.0.0**

- A latent SDE or unique manifold ontology.
- Licensed NDT \(\alpha/\omega/G_{\mathrm{peak}}\).
- Gate E Q as process noise or diffusion \(a(x)\).
- Observational FAR / \(R(\rho)\).
- MNPS \(\dot x\) as drift \(b\).
- C2 residualization.
- Empirical \(A_{bD}\) as NDT alignment.
- I-CARE tubes as `stochastic_reachability.v1`.
- Chart-robustness certificate as a product surface.

## Upgrading

- Existing MNPS, 9D, anchor, Jacobian, jacobian_metrics, and FTR paths remain
  compatible. Sub-schema identifiers such as `mnps_tensor_spec_v2_1` are retained.
- New HDF5 files write families under `/dynamical_families/...`, not
  `/orthogonal_dynamics/`. Legacy readers should keep the old path read-only.
- Replace YAML `orthogonal_dynamics:` with `dynamical_families:`. The old key is
  refused.
- Do not enable `dynamical_families` or `local_dynamics.stochastic_reachability`
  on common EEG/fMRI/ephys profiles. Use an explicit qualification overlay.
- Enable Gate F \(W_Q\) only together with Gate E transition residuals. See
  [`mndm/CONFIG_GUIDE.md`](../mndm/CONFIG_GUIDE.md).
- HDF5 `mndm_version` / package version now report `3.0.0`.
- Analysis-owned C1-M3, FAR, and committor work stays in `nmd-analysis`
  (`project/mnps_v3/df_drift_c1_analysis_handoff.md` in this repo).

## Full changelog

See [`CHANGELOG.md`](../CHANGELOG.md) for cumulative history.
See [`mndm/CONFIG_GUIDE.md`](../mndm/CONFIG_GUIDE.md) for YAML,
[`mndm/Output_variables_guide.md`](../mndm/Output_variables_guide.md) for HDF5
paths, and [`RELEASE_NOTES_v2.6.0.md`](RELEASE_NOTES_v2.6.0.md) for the preceding
local-dynamics release.
