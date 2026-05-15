#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
  numbering: "1",
)

#set text(
  font: "Times New Roman",
  size: 12pt,
)

#set par(
  justify: true,
  first-line-indent: 0.5in,
  leading: 1.8em,
)

#set heading(numbering: none)
#set math.equation(numbering: "(1)")

#show table: it => {
  set text(size: 9pt)
  set par(
    first-line-indent: 0pt,
    leading: 1.2em,
  )
  it
}

#align(center)[
  #text(size: 10pt)[Article type: Methods]
  #v(1em)
  #text(size: 18pt, weight: "bold")[
    NeuralManifoldDynamics: A Versioned Measurement Contract for Low-Dimensional Neural-Manifold Trajectories
  ]
  #v(1em)
  #text(size: 10pt)[Short title: NeuralManifoldDynamics]
  #v(1em)
  #text(size: 11pt)[Robin Langell]
  #v(0.5em)
  #text(size: 9pt, style: "italic")[Noetic Diffusion Project/Langell Konsult AB]
  #v(1em)
  #text(size: 10pt)[Corresponding author: Robin Langell]
  #v(1em)
  #text(size: 10pt)[Keywords: NeuralManifoldDynamics, MNPS, MNJ, EEG, fMRI, NWB, event annotations, regional dynamics, CSD, robustness]
]

#v(2em)

= Abstract
#par(first-line-indent: 0pt)[
  NeuralManifoldDynamics (v 2.0) is a versioned ingest-layer measurement contract for constructing and serializing low-dimensional and stratified neural-manifold proxy trajectories from EEG, fMRI, and selected NWB/ecephys feature tables, together with optional local Jacobian-based summaries. In the current release, the contract is NDT-aligned but operational rather than definitional: it fixes a canonical 3D chart (`mnps_3d = [m, d, e]`), an optional stratified 9D chart (`coords_9d`), and optional Jacobian exports as release-bound measurement objects rather than direct measurements of the full theoretical constructs. Relative to the older MNPS 1.2 generation, the updated contract introduces stricter coverage and estimator hygiene, explicit feature-standardization pipelines, improved handling of missing and non-finite support, self-describing HDF5 outputs, and regional manifold dynamics that now include EEG through channel-group trajectories in addition to fMRI network trajectories. For EEG, regionalization is coupled to optional Current Source Density preprocessing and topology-based electrode ensembles, making local regional summaries and, where enabled, block-Jacobian exports possible. The active reference configs are intentionally asymmetric at higher order: regional stratified and block-Jacobian exports are enabled for the EEG reference path but disabled for the main fMRI reference path. The repository also supports derived event-locked sidecar analyses, illustrated by sleep-spindle annotations, while keeping such event layers separate from the canonical ingest-time HDF5 measurement surface. The primary contribution of this release is therefore a stable, auditable, modality-aware measurement contract for downstream analysis and reuse, not a claim of comparative superiority over alternative latent-state or clustering frameworks.
]

= Author Summary
#par(first-line-indent: 0pt)[
  NeuralManifoldDynamics (v 2.0) describes how this repository turns EEG and fMRI feature tables into auditable manifold-proxy measurements for downstream analysis. The system no longer stops at a single 3D coordinate summary: it now supports a canonical 3D trajectory, an optional stratified 9D chart, regional manifold outputs for fMRI networks and EEG channel groups, stricter epoch-quality controls, explicit provenance, and self-describing HDF5 outputs. The purpose is not to interpret cognition at ingest time, infer diagnosis, or claim that this chart family is already the best available state-space representation. The purpose is to provide a stable, reproducible, and inspectable measurement contract that downstream analysis can compare, filter, and reinterpret.
]

#v(2em)

= Introduction
NeuralManifoldDynamics is the current name for the ingest-layer measurement system implemented in this repository for constructing NDT-aligned neural-manifold proxy trajectories from EEG and fMRI feature tables. It is grounded in the broader Noetic Diffusion Theory framework and its formalization of low-dimensional coordinates, stratified coordinates, and local dynamical summaries [@langell2025_6; @langell2025_stratified; @langell2025_mnj], but the contribution of this manuscript is primarily infrastructural and methodological rather than a claim of superior latent-state discovery. The central design choice is that ingest defines a fixed and reproducible measurement contract: it standardizes signals, extracts features, applies release-bound coordinate mappings, optionally estimates local dynamics, and exports auditable artifacts for downstream analysis. It does not adapt its behavior to contrasts of interest, and it does not interpret diagnoses, phenomenology, or conditions.

The three NDT background references used here are DOI-backed Zenodo records rather than peer-reviewed journal articles. They are cited to define the current terminology and layered conceptual background, while the present manuscript is intended to remain readable as a standalone methods/software description.

This new version replaces the older (internal) MNPS 1.2 style in which the primary emphasis was a weighted low-dimensional trajectory plus limited summary exports. The current implementation is more explicit about measurement support, coverage, failure modes, regionalization, provenance, and export semantics. It separates three related objects: the canonical 3D trajectory, the stratified 9D coordinate system, and the optional family of Jacobian-derived summaries built on top of these trajectories. Within this manuscript, these outputs should therefore be read first as a versioned software and data contract: a stable, inspectable interface between modality-specific preprocessing and downstream statistical or theoretical analysis. The intended contribution is not to benchmark a new state-discovery algorithm against all competing latent-variable methods, but to define and document a reproducible contract for producing auditable manifold-proxy measurements across supported modalities.

For neuroimaging readers, it is important to distinguish this role from several adjacent method families that are often applied after preprocessing. Common alternatives include modality-native preprocessing and feature ecosystems (`fMRIPrep`, `MNE-BIDS`, `Nilearn`, `mne-features`), dynamic-connectivity and chronnectomic workflows (`dyconnmap`, LEiDA-style analyses), latent state-space models such as HMM and DyNeMo in `OSL-Dynamics`, EEG microstate methods (`Pycrostates`), co-activation pattern approaches (`NeuroCAPs`), and learned latent embeddings such as `CEBRA` [@Esteban2019FMRIPrep; @Appelhoff2019MNEBIDS; @MNEBIDSPipelineDocs; @MNEFeaturesPackage; @Abraham2014Nilearn; @Marimpis2021Dyconnmap; @Cabral2017LEiDA; @Gohil2024OSLDynamics; @Ferat2022Pycrostates; @Smith2025NeuroCAPs; @Schneider2023CEBRA]. These are often stronger choices when the primary aim is adaptive state discovery, discrete state segmentation, or predictive performance. NeuralManifoldDynamics addresses a narrower problem: providing a fixed, auditable, multimodal measurement contract whose outputs remain comparable across runs and datasets.

#block(
  width: 100%,
  breakable: true,
  inset: 10pt,
  fill: luma(245),
  stroke: 0.5pt + luma(120),
  radius: 4pt,
)[
  *Claims and Non-Claims* \
  *Claims:* fixed export naming, deterministic feature preprocessing, explicit coverage handling, global and regional trajectory construction, optional local Jacobian serialization, and manifest-based self-description. \
  *Non-claims:* no diagnosis inference, no consciousness-level inference, no claim that ingest-level proxies exhaust the theoretical meaning of `[m, d, e]`, and no claim that EEG channel groups are direct homologues of fMRI networks. The chart labels should be read as operational labels within the current release contract.
]

#block(
  width: 100%,
  breakable: true,
  inset: 10pt,
  fill: luma(245),
  stroke: 0.5pt + luma(120),
  radius: 4pt,
)[
  *Chart Definition in Current Release* \
  `coords_9d` subcoordinates are fixed for this release. \
  `mnps_3d` is a derived canonical export from a fixed weighted projection of `coords_9d`. \
  Projection weights, axis names, and serialization paths are versioned and auditable. \
  They should not be read as claims of unique biological identifiability.
]

#figure(
  image("figures/neuralmanifolddynamics_flow.svg", width: 100%),
  caption: [
    Operational flow of the current NeuralManifoldDynamics ingest-layer measurement contract. Raw datasets are indexed, preprocessed, and converted into per-epoch feature tables before projection-time standardization is applied to weighted features for `coords_9d` and derived `mnps_3d`. Optional Jacobians, regional outputs, and self-describing HDF5 artifacts are then serialized for downstream analysis.
  ],
)

== Minimal NDT Notation Used Here
For readers encountering NDT here before the theory papers, only a small amount of notation is needed. In the broader NDT framework, neural dynamics are written as latent trajectories on a manifold:

$ d X_t = f(X_t, t) d t + sigma(t) d W_t $

with observed modality-specific signals generated through an observation map:

$ Y_t = g(X_t) + epsilon_t $

Here $X_t$ denotes a latent NDT state, $Y_t$ the measured EEG or fMRI signal family, $f$ the local drift field, and $epsilon_t$ measurement noise [@langell2025_6]. In the chart used throughout this manuscript, the coarse state coordinates are:

$ x_t = [m_t, d_t, e_t] $

where $m$ denotes a metastability / mobility-aligned coordinate, $d$ a deviation-from-optimal-balance coordinate, and $e$ an entropy / energetic-complexity coordinate [@langell2025_6; @langell2025_stratified]. Stratified NDT extends this to a finer chart:

$ x_t^(9) = [m_a, m_e, m_o, d_n, d_l, d_s, e_e, e_s, e_m] $

This manuscript does not claim to learn $X_t$ directly. Instead, ingest computes an empirical feature vector $z_t = phi(Y)_t$ from sliding-window EEG or fMRI features and applies a fixed release-bound mapping:

$ x_t^(9) = W_(9D) z_t, quad x_t = P x_t^(9) $

where $W_(9D)$ denotes the configured feature-to-subcoordinate map and $P$ the fixed 9D-to-3D projection used in the current release contract. When local dynamics are exported, the corresponding chart-level Jacobian is:

$ J(x_t, t) = frac(partial f(x, t), partial x) |_(x = x_t) $

For this paper, the key boundary is simple: NDT supplies the notation, while NeuralManifoldDynamics supplies one auditable empirical realization of that chart for ingest-time serialization rather than claiming to identify the latent manifold uniquely.

= Model Definition
Using the notation introduced above, the primary exported 3D trajectory is the canonical chart `x_t = [m_t, d_t, e_t]`, serialized in HDF5 as `mnps_3d`. The exported labels *m*, *d*, and *e* remain aligned to the theoretical MNPS axes of metastability, deviation from optimal integration-segregation balance, and entropy / entropic energy in the broader literature. In this ingest manuscript, however, they should be read as release-fixed operational proxy families rather than as direct redefinitions of those constructs:

- *m* is the current release's metastability-aligned proxy family, implemented through macrostate- and low-frequency morphology features under the active contract.
- *d* is the current release's deviation-aligned proxy family, implemented through dispersion and network-binding features under the active contract.
- *e* is the current release's entropy-aligned proxy family, implemented through nonlinear complexity, low-order energy, and auxiliary arousal-related features under the active contract.

Its temporal derivative is exported as `mnps_3d_dot`. The canonical axis order is fixed as `[m, d, e]` and is written explicitly into file-level metadata. Conceptually, this is an ingest-layer proxy realization of the low-dimensional Meta-Noetic Phase Space used in the Noetic Diffusion Theory program [@langell2025_6].

The stratified coordinate system extends this to the 9D chart introduced above, exported as `coords_9d/values` together with `coords_9d/names`. The purpose of the 9D system is not to replace the 3D manifold, but to provide a finer-grained operational decomposition of these proxy families within the current chart version. The primary 3D trajectory and the stratified 9D chart therefore coexist as distinct measurement objects, matching the rationale of Stratified Meta-Noetic Phase Space while remaining operational and version-bound in this ingest contract [@langell2025_stratified].

When Jacobian estimation is enabled, local dynamics are represented by chart-level Jacobian estimates. The primary Jacobian is written under `jacobian/J_hat`, while stratified dynamics are written under `jacobian_9D/J_hat` in the current codebase. Regional network-specific Jacobians are exported under `regional_mnps/<network>/jacobian`. This follows the role assigned to the Meta-Noetic Jacobian (MNJ) as the local second-order dynamical layer on top of MNPS coordinates [@langell2025_mnj].

== Rationale for the Stratified 9D Contract
The current `coords_9d` configuration is not presented here as a claim of uniquely privileged latent neurobiological ontology. Rather, it should be read as the current release's NDT-aligned measurement contract: a fixed, auditable decomposition chosen to balance increased resolution beyond coarse 3D composites, preserved recomposability into the canonical `mnps_3d` export, modality-level measurability in EEG and fMRI, and estimator-aware robustness at ingest time [@langell2025_stratified].

The main methodological motivation is dimensional masking. In a composite 3D summary, compensatory redistributions among subcoordinates can produce near-zero movement along a canonical axis even when the underlying signal family changes substantially. The stratified 9D chart exposes those redistributions directly and therefore reduces false-null behavior in the canonical 3D summary.

The nine subcoordinates were deliberately restricted to three families aligned with the canonical `[m, d, e]` topology. This grouping allows deterministic recomposition through a fixed weighted 9D->3D projection while keeping naming, provenance, and downstream serialization stable across datasets. The current weight values should therefore be read as release-fixed operational priors encoded in configuration during contract design, chosen to preserve sign consistency, recomposability, modality-level measurability, and estimator robustness across the reference paths. They are not learned dataset-specific optima, and they are not presented as claims of unique biological correctness. Full chart stability under feature substitutions, weighting perturbations, and alternative projection families remains a future validation target rather than an established property of the current release.

== Why this release uses this subcoordinate configuration
The current 9D chart was selected as the release contract for four practical reasons. First, the base 3D chart is sometimes too coarse: compensatory subcoordinate shifts can cancel in the composite and produce false-null behavior. Second, the chosen 9D grouping preserves recomposability into the canonical `[m, d, e]` export rather than creating nine unrelated free dimensions. Third, the selected subcoordinates remain measurable in the EEG and fMRI feature families actually supported by the current repository. Fourth, the chart had to remain versionable, auditable, and numerically usable under the coverage, finite-support, and Jacobian-validity constraints of the ingest layer, including a local dynamical regime that was neither so aggressive that small feature perturbations were amplified into unstable Jacobian estimates nor so flat that anisotropy and block-level summaries became uninformative.

This is therefore best read as a release-bound operational choice rather than as a claim that these are the uniquely correct latent primitives. The current configuration is intended to balance finer-grained decomposition against export stability: enough stratification to expose masking effects, but still constrained enough that the canonical 3D export can remain fixed across runs and datasets. Stronger claims about uniqueness or invariance belong to future validation rather than to the present contract-definition paper.

= Key Methodological Advances in this version
NeuralManifoldDynamics introduces several changes relative to the older MNPS 1.2 implementation.

== Stronger Measurement Robustness
The updated measurement model enforces explicit bounds on estimator support. Epoch inclusion is no longer a minimal pass/fail step. Instead, the pipeline tracks coverage in terms of available seconds, available epochs, and direct axis support. Missing weighted features are handled by per-axis renormalization rather than silent zero-filling. Windows or trajectories with insufficient support, all-non-finite stratified coordinates, or inconsistent dimensionality are now surfaced explicitly rather than silently propagated.

Feature preprocessing is also deterministic rather than ad hoc. In the current reference contracts, projection-time standardization defaults to `robust_z -> clip`, while selected power or bandpower features use explicit `log10 -> robust_z -> clip` overrides, as configured in `mnps_projection.feature_standardization`. Entropy-like, Hjorth-derived, and similar metrics are not subjected to blind `log10` compression unless explicitly configured. Separately, the exported `/features_robust_z/*` surface is a strict robust-z view of the raw feature matrix and does not bake in projection-only `log10` or clipping steps; those remain represented in provenance metadata. The untransformed baselines used to produce projection-time normalized values are retained as per-feature metadata (`abs_median`, `abs_mad`, and applied transformation string), so absolute scale is preserved for audit rather than destroyed by preprocessing.

This release also now records explicit reproducibility provenance for the exported manifold and Jacobian surfaces, including stable hashes for `mnps_3d`, neighbor indices, and primary and stratified Jacobian tensors. In a reference replay on open neuro dataset `ds003059`, two full summarization runs over the same regenerated feature table, executed with identical configuration and seed but different parallel worker counts (`n_jobs = 1` versus `n_jobs = 4`), produced matching subject-level provenance hashes across all `90` summarized runs for `x_hash_saved`, `nn_indices_hash_saved`, `jacobian_hash_saved`, `jacobian_dot_hash_saved`, `coords_9d_hash_saved`, `jacobian_9d_hash_saved`, and `jacobian_9d_dot_hash_saved`. This should be read as an implementation-level reproducibility check within one environment rather than as a claim of cross-machine floating-point identity under arbitrarily different BLAS, OS, or dependency stacks.

== Improved Epoch Quality and Support
The current pipeline is designed to retain more usable epochs while improving quality control. Coverage policy is now computed explicitly, including effective coverage after masking and quality-control drops. This allows the system to preserve high-quality support where possible while rejecting windows that would otherwise degrade the Jacobian fit or distort anisotropy-related summaries.

The result is not only more data, but more defensible data. This matters because anisotropy, condition numbers, and local Jacobian estimates are highly sensitive to unstable or poorly supported neighborhoods.

== Derivative and Time-Base Contract
`mnps_3d_dot` is not an unspecified symbolic derivative. In the active contracts it is estimated with a Savitzky-Golay derivative on the epoch-time series, with EEG default `window = 7`, `polyorder = 3`, and fMRI default `window = 5`, `polyorder = 2`. When the sequence is too short for a valid Savitzky-Golay fit, the implementation falls back to central differences; when large jumps segment a trajectory, the robust segmented derivative path prevents smoothing across discontinuities. In the current implementation, Savitzky-Golay derivatives are evaluated with interpolation-based edge handling and are not post-trimmed at segment boundaries before export. Accordingly, boundary derivatives remain part of the serialized contract and should be treated as lower-confidence near short or recently split segments if a downstream analysis requires stricter edge control. These choices are part of the measurement contract because downstream Jacobian estimation depends directly on derivative stability.

== Formal 3D and 9D Separation
The older pipeline used naming that blurred low-dimensional and stratified outputs. The new version now makes the distinction explicit:

- `mnps_3d` is the canonical 3D trajectory.
- `coords_9d` is the stratified coordinate chart.
- `mnps_3d_dot` is the derivative of the canonical trajectory.

This separation makes the output contract more interpretable for human readers and for downstream tooling.

= Regional NeuralManifoldDynamics
Regional manifold dynamics are now a core part of the implemented measurement contract rather than an external post-processing idea. In theoretical terms, this extends the MNPS and MNJ framing from a single global chart to a set of network- or group-specific charts that can be compared within one measurement contract [@langell2025_stratified; @langell2025_mnj].

== Regional fMRI
The repository already supported regional fMRI through ROI-based or network-based aggregation. Version 2.0 preserves that path and continues to export regional manifold summaries without changing the basic contract. Canonical derived regional outputs are written under `regional_mnps/*` for both EEG and fMRI, while `/regions/*` is reserved for optional supporting raw regional signals, mainly on the fMRI side. The code keeps modality-specific safeguards where stratified block Jacobians are not empirically justified for fMRI; in the active `ds000228` reference path, regional 3D summaries are enabled but regional stratified and block-Jacobian exports remain disabled.

== Regional EEG via Channel Groups
The major new addition is regional EEG support. EEG features can now be grouped using topology-based channel ensembles, producing per-group feature columns with `__g_<group>` suffixes. These grouped feature tables are then converted into per-group manifold trajectories and optional stratified regional trajectories.

In practical terms, a channel group such as `frontal`, `central`, `parietal_occipital`, or `temporal` is treated as a topology-based regional surrogate used to approximate a regional decomposition under shared naming and export patterns, while preserving modality-specific interpretive limits. Each region can therefore produce:

- a regional 3D trajectory,
- a regional 3D Jacobian,
- an optional regional stratified trajectory,
- regional CSV-style summaries that are now also embedded into HDF5.

This brings EEG regional processing closer to the fMRI regional path while still keeping modality differences explicit.

== CSD / Surface Laplacian for EEG
Regional EEG is coupled to an optional *Current Source Density* preprocessing step. This is a critical methodological change because direct channel averaging in sensor space is otherwise vulnerable to volume conduction. In the active EEG reference configuration, the CSD transform uses `lambda2 = 1e-5`, `stiffness = 4.0`, `n_legendre_terms = 50`, and `min_eeg_channels = 16`, with failure behavior controlled by `on_error` (the current `ds004511` overlay uses `warn`). These parameters are written into preprocessing metadata so the exact spatial filter is auditable.

The important design point is that the measurement contract now acknowledges spatial filtering as the preferred safeguard before inter-regional EEG dynamical summaries are interpreted. In the current implementation, CSD is an optional supported preprocessing path rather than a hard requirement, and failure-tolerant configurations can continue without it while recording the chosen behavior in provenance.

= Modality Coverage
The current reference configurations, `ds000228` for fMRI [@richardson2023_ds000228] and `ds004511` for EEG [@ds004511:1.0.2], support the following entities:

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, center, center),

  [*Data entity*], [*fMRI*], [*EEG/iEEG*],
  [Global `mnps_3d`], [Yes], [Yes],
  [Global `coords_9d`], [Yes], [Yes],
  [Global MNJ], 
  [
    Yes \
    `jacobian/J_hat` on `mnps_3d` \
    `jacobian_9D/J_hat` on `coords_9d`
  ],
  [
    Yes \
    `jacobian/J_hat` on `mnps_3d` \
    `jacobian_9D/J_hat` on `coords_9d`
  ],

  [Regional `mnps_3d`],
  [
    Yes \
    network-level 3D from regional fMRI features
  ],
  [
    Yes \
    channel-group 3D from `__g_<group>` EEG features
  ],

  [Regional `coords_9d`],
  [
    No in `ds000228` \
    code path exists, but `regional_mnps.stratified.enabled = false`
  ],
  [
    Yes in `ds004511` \
    `regional_mnps.stratified.enabled = true`
  ],

  [Regional `mnps_3d` + MNJ],
  [
    Yes \
    `regional_mnps/<network>/jacobian`
  ],
  [
    Yes \
    `regional_mnps/<network>/jacobian`
  ],

  [Regional `coords_9d` + MNJ / block structure],
  [
    No in `ds000228` \
    regional 9D disabled by config
  ],
  [
    Yes in `ds004511` \
    regional stratified trajectories enabled \
    regional block Jacobians enabled
  ],
)

#par(first-line-indent: 0pt)[
  Code check: this table reflects the active reference configs rather than only abstract code capability. For fMRI, `ds000228` keeps global `coords_9d` enabled because `mnps_3d` is derived from 9D, but disables regional 9D in config. For EEG, `ds004511` enables both global and regional stratified trajectories, together with regional block-Jacobian summaries. In the current repository, iEEG datasets are routed through the electrophysiology path using `modality: eeg`, so the rightmost column should be read as the current EEG-family implementation path.
]

== DANDI / NWB ingest path
In addition to BIDS-style OpenNeuro EEG/fMRI workflows, the repository includes a DANDI/NWB ingest path for selected neurophysiology assets [@dandi2022; @nwb2022]. This path is implemented as a constrained adapter layer rather than as a claim of full NWB ecosystem coverage. DANDI helper configs create asset manifests and probe local `.nwb` files, while MNDM configs such as `config_ingest_dandi_000718.yaml` and `config_ingest_dandi_000458.yaml` route NWB electrical-series data through the same downstream feature, MNPS, and HDF5 writer contracts used by the rest of the package. Where NWB interval tables expose behavioral or state labels, these can be mapped onto the MNPS time axis through `state_labels` / `within_run_labels` and serialized alongside the trajectory as `/labels/stage` or named labels. This makes DANDI/NWB support part of the source-format and label-alignment surface of the measurement contract, not a separate theoretical model.

== PhysioNet / WFDB ingest path and clock provenance
In parallel with the DANDI/NWB path, the repository now includes a PhysioNet ingest utility for cohort-level selection, checksum-aware transfer, resumable downloads, and manifest logging. On the MNDM side, WFDB headers (`.hea`) with paired signal files (`.mat`/`.dat`) are routed through the same canonical feature, MNPS, and HDF5 measurement contract used by other modalities. For WFDB overlays where `time_reference.enabled = true`, the export layer preserves the canonical relative-time surfaces (`/time`, `/window_start`, `/window_end`) and adds explicit clock-provenance extensions under `/extensions/time_reference/run/*` and `/extensions/time_reference/windows/*`. This extension is metadata-oriented: it improves temporal auditability and cross-run alignment without redefining the underlying MNPS coordinate contract.

= Axis Construction
The same reference configurations construct global `mnps_3d` and `coords_9d` as follows. The rows below should be read as modality-specific operationalizations under a shared chart family, not as claims of one-to-one physiological homology between EEG and fMRI features and not as a redefinition of the underlying theoretical axes.

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, left, left),

  [*Subcoordinate / entity*], [*fMRI (`ds000228`)*], [*EEG (`ds004511`)*],
  [
    `mnps_3d`
  ],
  [
    `mnps_3d.mode = from_v2` \
    `x = coords_9d @ P_fixed` \
    `P_fixed` from `mnps_projection.v1_mapping` \
    weights resolved against `mnps_9d.subcoords` \
    runtime: L2-normalized columns + coverage-aware renormalization
  ],
  [
    `mnps_3d.mode = from_v2` \
    `m <- 0.62*m_a + 0.55*m_e + 0.45*m_o` \
    `d <- 0.50*d_n + 0.82*d_l + 0.28*d_s` \
    `e <- 0.85*e_e + 0.62*e_s + 0.03*e_m` \
    runtime: L2-normalized columns + coverage-aware renormalization
  ],

  [`m_a`], [`fmri_FC_mean`], [`-0.5*eeg_delta - 0.5*eeg_theta`],
  [`m_e`], [`fmri_gradient_ratio`], [`-1.0*eeg_alpha`],
  [`m_o`], [`fmri_modularity`], [`eeg_beta_alpha`],
  [`d_n`], [`fmri_variance_global`], [`eeg_gamma`],
  [`d_l`], [`fmri_dFC_variance`], [`eeg_hjorth_mobility`],
  [`d_s`], [`fmri_kuramoto_global`], [`eeg_alpha_theta`],
  [`e_e`], [`fmri_signal_power`], [`eeg_permutation_entropy`],
  [`e_s`], [`fmri_slow4_slow5_ratio`], [`eeg_hjorth_complexity`],
  [`e_m`], [`fmri_ar1_coefficient`], [`ecg_rmssd -> eog_blink_rate -> eeg_highfreq_power_30_45`],
)

#par(first-line-indent: 0pt)[
  These rows summarize the active 9D-to-3D construction used by the current reference configs. The active runtime path is a fixed weighted projection from `coords_9d` to `mnps_3d`, not a trivial equal-weight mean over all subaxes. The projection weights should be read as release-fixed operational priors chosen during contract design to preserve recomposability, interpretability, modality-level measurability, and estimator robustness within the ingest contract; they were not obtained by supervised optimization against one benchmark objective. Part of that robustness criterion was dynamical rather than purely semantic: the selected weighting had to support a usable Jacobian layer that was neither excessively aggressive under small feature perturbations nor trivially flat in ways that would collapse local anisotropy and block-level summaries. The EEG `e_m` slot is especially operational: the current code resolves it empirically from `ecg_rmssd`, else `eog_blink_rate`, else `eeg_highfreq_power_30_45`, while also storing the chosen source. The manuscript therefore treats this slot as an empirical fallback family rather than as a direct embodiment variable. This improves coverage but weakens strict inter-dataset identity for `e_m`, so cross-dataset comparisons involving that slot should be interpreted with added caution and with the recorded source provenance in view. For fMRI, regional stratified construction is present in the dataset file but explicitly disabled. For EEG, the dataset overlay enables regional stratified MNPS and associated regional block-Jacobian summaries.
]

#par(first-line-indent: 0pt)[
  When subcoordinate support is incomplete, the implementation renormalizes over the weights that remain present and records per-axis coverage. This yields a degraded support class of `mnps_3d` estimates rather than a geometry that is automatically identical to the full-support case. Accordingly, scale-sensitive geometric summaries derived from full-support and degraded-support trajectories should be treated as support-conditioned and, where necessary, adjusted downstream using the exported coverage and provenance rather than assumed to be directly interchangeable.
]

= Jacobians, Block Jacobians, and Anisotropy
NeuralManifoldDynamics 2.0 extends the dynamical output family beyond a single primary Jacobian. This is directly aligned with the idea that first-order position in manifold space and second-order transformation structure should be reported separately rather than collapsed into one scalar summary [@langell2025_mnj].

When enabled, the current implementation exports:

- primary 3D Jacobians on `mnps_3d`,
- stratified Jacobians on `coords_9d`,
- regional Jacobians for each network or EEG channel group,
- block-Jacobian summaries for stratified and regional outputs,
- embedded tabular exports of those summaries inside HDF5.

Anisotropy is now treated as a first-class quality and geometry descriptor. It appears in regional summaries and in block-Jacobian summaries, alongside Frobenius norms, trace-like quantities, and symmetric or rotational cross-block metrics where applicable. The practical effect is that version 2.0 provides a more discriminating description of local geometry than a pure trace-based summary.

== Jacobian Validity Domain
Jacobian export is conditional on support and numerical validity rather than guaranteed by name alone. The active implementation enforces or records at least the following constraints:

- minimum coverage in seconds and epochs before a segment is processed,
- minimum direct-axis support after projection renormalization (`min_axis_coverage`, default `0.3`),
- finite-valued `mnps_3d` rows before kNN/Jacobian estimation,
- finite-valued `coords_9d` rows before stratified Jacobian estimation,
- conditioning and anisotropy diagnostics in downstream summaries, including Jacobian condition-number summaries and regional `strat9_condition_number`,
- withholding or skipping of regional/block Jacobians when the modality/configuration is not empirically supportable, most notably regional 9D block Jacobians for fMRI, which are disabled by default because per-network trajectories are typically rank-deficient at available window counts.

Accordingly, Jacobian-derived exports should be read as valid only within these support constraints. The contract serializes the resulting diagnostics and provenance; it does not imply that every requested Jacobian is estimable for every dataset, modality, or regional decomposition.

== Chart Stability as Future Validation Target
The present manuscript defines the current release contract, not full embedding-family invariance. In particular, it fixes one NDT-aligned chart family, one set of subcoordinate definitions per release, and one auditable 9D->3D projection contract. Future validation should therefore assess chart stability under reasonable feature substitutions, weighting perturbations, and projection changes, so that release stability can be distinguished from calibration dependence.

= Output Contract and Self-Describing Artifacts
The export layer has also changed substantially.

== Run Directory and Naming
Runs are now written into directories named:

`neuralmanifolddynamics_<dataset>_<timestamp>`

This replaces the older `mnps_*` naming convention and makes the run purpose clearer.

== HDF5 Naming
The HDF5 contract is more explicit than before. Important paths now include:

- `mnps_3d`
- `mnps_3d_dot`
- `coords_9d/values`
- `coords_9d/names`
- `jacobian/J_hat`
- `window_start` and `window_end`
- `extensions/time_reference/run/*`
- `extensions/time_reference/windows/*`
- `labels/stage`
- `regional_mnps/<network>/mnps`
- `regional_mnps/<network>/jacobian`
- `extensions/tabular_exports/*`

The previous ambiguity of short names such as `x` has been removed.

== Self-Description Through Manifests
Each run writes `run_manifest.json`, which now includes a field guide describing the meaning of key HDF5 paths. Capability probes in the same manifest can also report extension presence (including `time_reference`) so that downstream tooling can branch on available temporal metadata without schema guessing. This is important because the export contract should be legible to both human users and automated readers without requiring separate source-code inspection.

Selected summary tables that were previously emitted only as CSV files are now also embedded into HDF5 as columnar exports under `extensions/tabular_exports`. This makes the HDF5 file a more self-contained artifact.

== Reference Run Snapshot
To make the contract more concrete, the documentation bundle accompanying this manuscript includes a reference EEG metadata snapshot for `ds004511` under `metadata/`. In that example run, `run_manifest.json` reports 132 subject-level HDF5 outputs, together with 132 `summary.json`, 132 `qc_summary.json`, and 132 `qc_reliability.json` files across 45 subjects and 3 tasks. The same manifest records that all probed HDF5 files contained `mnps_3d`, `coords_9d`, 3D and 9D Jacobian groups, regional outputs, and both raw and strict-robust-z feature exports.

The accompanying `features_snapshot.json` shows the feature-layer side of the contract for the same run, including 55,772 rows and explicit columns for grouped EEG features, entropy provenance fields, and `embodied_arousal_proxy_source`. Per-file QC artifact JSONs further illustrate how preprocessing safeguards are serialized. For example, the `sub-S210317 ... Rest` and `... CC` records both document 3000 Hz -> 250 Hz resampling by integer-ratio policy, identified bad channels, and an EEG CSD path that was enabled but not applied because digitization was unavailable; the chosen failure reason is retained in provenance rather than being hidden. These snapshots are illustrative rather than inferential, but they show that the measurement contract described in this manuscript is realized as concrete, inspectable artifacts.

A refreshed sleep-dataset run for `ds005555` further illustrates the newer reviewer-facing QA exports. In the `sub-1_Sleep_acq-psg` example, `qc_summary.json` reports 160 retained epochs (640 s) together with a `baseline_comparisons` block spanning raw entropy, smoothed entropy, variance/bandpower summaries, and a simple sliding-window FC baseline via `eeg_dfc_variance`. In the same record, `eeg_permutation_entropy` aligns most strongly with the exported `e` axis (`r = 0.924`), while the null/sanity export shows that time-shuffling collapses the mean axis autocorrelation length from 24-32 s in the original trajectory to 4 s across all three axes. White-noise surrogates similarly inflate total path length by about `2.82x` relative to the observed trajectory. These `ds005555` outputs are still QA artifacts rather than a full benchmark figure panel, but they demonstrate that simple baseline contrasts and null perturbations can now be serialized in the same auditable output surface as the main MNPS summaries.

A separate event-locked sleep-spindle workflow was developed as a derived layer on top of the same `ds005555` measurement outputs. In that workflow, YASA-derived N2 spindle annotations [@vallat2021_yasa] are aligned to short-window MNPS trajectories, matched to same-subject N2 control windows, and exported as sidecar tables for downstream statistical analysis. The key contract point is architectural rather than biological: spindle annotations, baseline-corrected bins, and event-window mappings are treated as derived event-layer artifacts with their own detector and QC provenance, while the canonical subject HDF5 files continue to store the measurement surface (`mnps_3d`, `coords_9d`, Jacobians, labels, features, and manifests). This prevents a special-purpose sleep analysis from becoming implicit primary output, but still makes event-locked analyses reproducible and joinable to the canonical HDF5 trajectory through subject IDs and window indices. Appendix A.6 gives the recommended layout.

= Relation to Existing Frameworks
NeuralManifoldDynamics sits adjacent to several established software ecosystems rather than replacing them. For standardized preprocessing and data organization, the closest precedents are BIDS-oriented pipelines such as `fMRIPrep`, `MNE-BIDS`, and the `MNE-BIDS-Pipeline`, together with feature- and ROI-oriented toolkits such as `mne-features` and `Nilearn` [@Esteban2019FMRIPrep; @Appelhoff2019MNEBIDS; @MNEBIDSPipelineDocs; @MNEFeaturesPackage; @Abraham2014Nilearn]. These systems remain stronger choices when the primary goal is modality-native preprocessing maturity, broad BIDS interoperability, or extraction of clean modality-specific time series. NeuralManifoldDynamics instead occupies a narrower layer: it standardizes feature-level inputs, fixes a versioned proxy-chart contract, serializes auditable outputs, and records provenance and capability metadata for downstream use.

The manuscript should also be read against data-adaptive state-space and brain-state software. Methods such as `CEBRA` optimize latent embeddings from data [@Schneider2023CEBRA], while packages such as `Pycrostates` and `NeuroCAPs` operationalize states through discrete microstate segmentation or co-activation pattern clustering [@Ferat2022Pycrostates; @Smith2025NeuroCAPs]. More general neurodynamics and dynamic-connectivity toolboxes, including HMM- and DyNeMo-based workflows in `OSL-Dynamics`, `dyconnmap`, and LEiDA-style state analyses, emphasize generative latent-state inference, time-varying connectivity, recurrent connectivity states, or broader dynamic integration analyses of neural activity [@Gohil2024OSLDynamics; @Marimpis2021Dyconnmap; @Cabral2017LEiDA; @shine2016]. NeuralManifoldDynamics should not be read as claiming empirical superiority over these families. Rather, it makes a different trade-off: less data-adaptive flexibility in exchange for release-bound coordinates, explicit serialization paths, reproducible defaults, and more direct cross-run auditability.

Accordingly, the main claim of NeuralManifoldDynamics is not that every ingredient is novel in isolation, nor that the repository currently establishes a new performance baseline against clustering-based, HMM-family, or latent-embedding methods. The narrower and more defensible claim is that the repository combines multimodal ingest, a fixed 3D/9D proxy-chart family, optional first-layer Jacobian exports, self-describing HDF5 outputs, and manifest-level provenance into one auditable measurement contract. In publication terms, this places the work closer to a methods-oriented software/resource contribution than to a full comparative benchmark paper.

= Relationship to the Older MNPS 1.2 Generation
The appropriate way to think about version 2.0 is not as a cosmetic rename, but as a stricter measurement model.

Compared with MNPS 1.2, the new system:

- formalizes the distinction between 3D and 9D coordinates,
- improves robustness and coverage handling,
- makes feature standardization explicit and auditable,
- supports regional EEG in addition to regional fMRI,
- couples EEG regionalization to optional CSD preprocessing,
- exports richer Jacobian and anisotropy-oriented summaries,
- writes more self-describing HDF5 and run-manifest outputs.

The theoretical object remains a manifold-based description of neural dynamics, but the implementation is now better aligned with estimator hygiene, provenance, and reproducible export semantics.

= Methods-Oriented Discussion
The most important conceptual shift in NeuralManifoldDynamics 2.0 is methodological rather than rhetorical. The ingest layer is no longer treated as a lightweight staging area before “real” analysis begins. Instead, it is treated as the place where the measurement contract is fixed.

This has several consequences. First, naming matters, because ambiguous path names lead to ambiguous downstream assumptions. Second, coverage matters, because local linear estimators fail silently when support is poor. Third, regional EEG cannot be justified merely by averaging channels; it must be coupled to a defensible preprocessing pathway. Fourth, regional fMRI and regional EEG should share a common export logic where possible, while still preserving their modality-specific limits.

Under this design, NeuralManifoldDynamics 2.0 is best understood as an auditable, NDT-aligned measurement contract. Downstream analysis may compare groups, estimate clinical effects, or test theoretical predictions, but those later steps should inherit a stable coordinate system rather than redefine it.

= Limitations
The current manuscript has several important limitations that should be read as part of the contract definition rather than as post hoc caveats.

- The chart family is release-bound and auditable, but not yet validated as invariant under feature substitutions, weighting perturbations, or alternative projection families.
- The present paper does not provide a comparative benchmark against HMM-family models, CAP methods, LEiDA-style workflows, microstate methods, or learned latent embeddings such as `CEBRA`; those methods may be stronger choices when adaptive state discovery or predictive performance is the primary aim.
- The reference configurations are intentionally asymmetric across modalities: higher-order regional stratified and block-Jacobian exports are enabled for the EEG reference path but disabled for the main fMRI reference path.
- The EEG `e_m` slot uses a recorded fallback family (`ecg_rmssd -> eog_blink_rate -> eeg_highfreq_power_30_45`) when preferred inputs are unavailable. This preserves coverage and provenance, but it weakens strict inter-dataset comparability for that subcoordinate.
- Jacobian-derived outputs remain support- and estimator-limited. In particular, higher-dimensional regional fMRI Jacobians are withheld by default where available window counts are typically insufficient for stable estimation.
- Reviewer-oriented `baseline_comparisons` and `null_sanity_tests` are now emitted in subject-level QA JSONs for reference runs, but these should be read as artifact-level sanity checks rather than as a completed comparative benchmark against external methods.
- The present manuscript is organized as a measurement-contract and software paper and therefore does not yet include a dedicated figure panel of reference-run trajectories, QC distributions, or subject-level output examples.

= Conclusions
NeuralManifoldDynamics 2.0 is the current implementation name for the manifold measurement system in this repository. It supersedes the older MNPS 1.2-style ingest contract by making the coordinate hierarchy, robustness logic, regionalization strategy, provenance surface, and output semantics substantially more explicit.

The present release is best understood as a methods-oriented software and data resource with four defining properties:

1. a canonical 3D manifold trajectory, `mnps_3d`;
2. a stratified 9D coordinate chart, `coords_9d`;
3. regional trajectory and Jacobian outputs for both fMRI and EEG channel groups;
4. self-describing exports designed for auditability and downstream reproducibility.

Its primary contribution is therefore not a claim that the current chart family is uniquely biologically identified or already benchmarked as the best available state-space representation. The contribution is that supported datasets can be processed into a stable, inspectable, versioned measurement object that downstream analyses can compare, filter, reinterpret, and test without having to reconstruct ingest assumptions from source code.

This makes the system better suited for public release, external inspection, downstream scientific reuse, and future comparative validation against simpler or more data-adaptive alternatives.


= Acknowledgements


Large language model assistants were used under human supervision for literature synthesis, drafting support, peer-review simulation, and editorial refinement. These tools are acknowledged as writing assistants rather than as authors or collaborators.

= Contact

Robin Langell — hello(at)noeticdiffusion.com

= Licensing

GNU GENERAL PUBLIC LICENSE v3. See LICENSE in the root folder.

= Data and Code Availability

Reference implementations and analysis pipelines are available at
`https://github.com/NoeticDiffusion` 

= Trademark Notice

Certain terms used in this manuscript (including Noetic Diffusion, Noetic
Diffusion Theory, Noetic Diffusion Mapping, Noetic Diffusion Health Index,
and Noetic Atlas) are used as project names and are the subject of ongoing
trademark applications. A concise description of the stewardship rationale,
current registration status, and how this interacts with open scientific use is
available at `https://noeticdiffusion.com/license.html`. These trademark aspects
do not affect the scientific content, reproducibility, or licensing of the
methods described here.

= Appendix A: Running the Reference Implementation

The current repository is available at:

`https://github.com/NoeticDiffusion/NeuralManifoldDynamics`

The commands below describe a practical way to run the released code from a
source checkout on Windows PowerShell. The same logic applies on other
platforms with shell-specific path adjustments.

== A.1 Environment setup

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

When running directly from the monorepo source tree without editable package
installation, the package roots must also be exposed through `PYTHONPATH`:

```powershell
$repo_root="C:/path/to/NeuralManifoldDynamics"
$env:PYTHONPATH="$repo_root/mndm/src;$repo_root/core/src;$repo_root/openneuro_ingest/src;$repo_root/apollo_ingest/src;$repo_root/vitaldb_ingest/src"
```

`pyarrow` is recommended so feature tables can be written and read cleanly in
parquet format, although the pipeline can fall back to CSV/JSON-oriented paths
when necessary.

== A.2 Typical execution pattern

For datasets already present on disk, the main entry point is `mndm.cli`.
The examples below use the two reference configurations discussed in the main text: `ds004511` for EEG [@ds004511:1.0.2] and `ds000228` for fMRI [@richardson2023_ds000228].

A direct end-to-end EEG example is:

```powershell
python -m mndm.cli all --dataset ds004511 --config mndm/config/config_ingest_ds004511.yaml --n-jobs 12
```

A corresponding fMRI example is:

```powershell
python -m mndm.cli all --dataset ds000228 --config mndm/config/config_ingest_ds000228.yaml --n-jobs 12
```

This runs:

1. file indexing and feature extraction
2. MNPS summarization and optional Jacobian estimation
3. HDF5, JSON, and manifest writing

The stages can also be run separately:

```powershell
python -m mndm.cli features --dataset ds004511 --config mndm/config/config_ingest_ds004511.yaml --n-jobs 12
python -m mndm.cli summarize --dataset ds004511 --config mndm/config/config_ingest_ds004511.yaml --n-jobs 12
```

Optional post-processing utilities include:

```powershell
python -m mndm.cli pack --dataset ds004511 --config mndm/config/config_ingest_ds004511.yaml
python -m mndm.cli check-structure --dataset ds004511 --config mndm/config/config_ingest_ds004511.yaml --run-selector latest
```

== A.3 Data locations and configuration

Runtime behavior is controlled through YAML overlays under `mndm/config/`.
In practical use, dataset-specific files such as:

`mndm/config/config_ingest_ds004511.yaml`

or

`mndm/config/config_ingest_ds000228.yaml`

override shared defaults from the common EEG or fMRI configurations. Local or
nonstandard dataset roots can be specified through
`paths.dataset_received_dirs.<dataset_id>`.

== A.4 Output layout

Processed outputs are typically written under a dataset-specific processed
directory. Summarized runs appear in directories named:

`neuralmanifolddynamics_<dataset>_<timestamp>`

These runs typically contain:

- `run_manifest.json`
- `features_snapshot.json`
- per-subject or per-run subdirectories with:
  - `summary.json`
  - `qc_summary.json`
  - `qc_reliability.json`
  - subject-level HDF5 outputs

The HDF5 contract described in this manuscript includes canonical `mnps_3d`
exports, optional `coords_9d`, optional Jacobian groups, and feature surfaces such as
`/features_raw/*` and `/features_robust_z/*`.

== A.5 DANDI/NWB execution path

DANDI/NWB workflows use two layers. The `dandi_ingest` package creates manifests,
probes available assets, and records local triage information for selected
DANDI dandisets. MNDM then processes local NWB files through dataset overlays such
as:

```powershell
python -m dandi_ingest.cli list --config dandi_ingest/configs/dandi_000718.yaml
python -m dandi_ingest.cli probe --config dandi_ingest/configs/dandi_000718.yaml
python -m mndm.cli all --dataset dandi_000718 --config mndm/config/config_ingest_dandi_000718.yaml
```

The current NWB path is intentionally conservative. It selects supported
electrical-series data, applies the configured preprocessing and epoching rules,
and emits the same MNDM feature/MNPS/HDF5 contract used for other modalities.
NWB interval tables can also be used as time-varying state labels when enabled,
so that states such as behavioral epochs or anesthesia intervals are written as
time-aligned labels rather than forced into a single run-level condition.

== A.6 Derived event-locked sidecars

Event-locked analyses should be treated as derived layers unless and until they
become part of a stable release schema. For sleep spindles, the recommended
short-window workflow is:

```text
canonical H5 measurement run
  -> detector/imported event annotations
  -> event-to-window alignment
  -> matched same-stage controls
  -> derived event-locked sidecar tables
```

For the `ds005555` sleep-spindle track, the canonical HDF5 files remain the
source of MNPS, 9D, Jacobian, stage, feature, and provenance data. Detector
outputs and event-locked summaries are kept as sidecars such as:

```text
<processed>/<dataset>/baseline_corrected_all_psg_c3.csv
<processed>/<dataset>/baseline_corrected_all_psg_f3.csv
```

Those CSV files are sufficient for subject-level sign tests and baseline
robustness checks because they contain one row per subject/bin with
baseline-corrected `m`, `d`, and `e` summaries. They are not sufficient for
event-level analyses of `coords_9d`, Jacobians, or future reachability measures.
For publication-grade event reuse, the preferred next release format is a
versioned derived sidecar directory containing per-subject event tables,
event-window mappings, baseline-corrected bin tables, and QC JSON, for example:

```text
<run>/derived_events/
  spindles_yasa_0_7_0_psg_c3/
    sub-001_events.parquet
    sub-001_event_locked_windows.parquet
    sub-001_baseline_corrected_bins.parquet
    sub-001_qc.json
```

This keeps the canonical HDF5 measurement contract narrow while preserving the
information needed to rejoin event-level analyses to `/mnps_3d`, `/coords_9d`,
`/jacobian/*`, and time-aligned labels.

= Appendix B: Consolidated NDT Reference Notation and Its Relation to Ingest

This appendix consolidates the notation introduced briefly in the Introduction
so that readers can find the main formulas in one place. The summary does not
change the main claim of this paper: NeuralManifoldDynamics implements a
measurement contract, not a full latent-manifold identification procedure.

== B.1 Latent dynamics and observation layer

In the broader NDT formalism, neural dynamics are modeled as a stochastic
process on a latent manifold:

$ d X_t = f(X_t, t) d t + sigma(t) d W_t $

with observed signals generated through:

$ Y_t = g(X_t) + epsilon_t $

Here $X_t$ denotes the latent NDT state, $f$ the drift field, $sigma(t)$ a
time-varying diffusion scale, $W_t$ Brownian motion, $Y_t$ the observed EEG or
fMRI measurement family, $g$ the observation map, and $epsilon_t$ observation
noise [@langell2025_6].

== B.2 Canonical and stratified charts

The canonical NDT chart is:

$ x_t = [m_t, d_t, e_t] $

with the conventional interpretation:

- $m$: metastability / mobility / rhythmic-coherence-aligned coordinate,
- $d$: deviation from optimal integration-segregation balance,
- $e$: entropy / energetic-complexity-aligned coordinate.

Stratified NDT refines this chart to:

$ x_t^(9) = [m_a, m_e, m_o, d_n, d_l, d_s, e_e, e_s, e_m] $

where the three families preserve the $m$-, $d$-, and $e$-aligned grouping while
making within-family redistributions visible [@langell2025_stratified].

== B.3 Ingest-time operationalization

NeuralManifoldDynamics does not estimate the latent manifold $cal(M)$ from
scratch in this manuscript. Instead, it computes a windowed empirical feature
vector:

$ z_t = phi(Y)_t $

and maps that feature vector into a release-fixed chart:

$ x_t^(9) = W_(9D) z_t $

followed by a fixed 9D-to-3D projection:

$ x_t = P x_t^(9) $

In this paper, $W_(9D)$ and $P$ are not learned per contrast; they are versioned
configuration objects recorded in provenance. That is why the output should be
read as an auditable measurement contract rather than as a data-adaptive
embedding benchmark.

== B.4 Local dynamics and Jacobian layer

When local dynamics are exported, the corresponding chart-level Jacobian is:

$ J(x_t, t) = frac(partial f(x, t), partial x) |_(x = x_t) $

This Jacobian summarizes local deformation structure in chart coordinates. It
is not presented as direct access to a unique biophysical state equation
[@langell2025_mnj]. In the ingest implementation, Jacobians are estimated only
when support and numerical validity are sufficient under the active coverage and
conditioning rules.

== B.5 Interpretation boundary

For a reader seeing NDT first through this manuscript, the practical reading
rule is therefore:

1. NDT supplies the layered notation: latent state, chart coordinates, and local
   dynamical summaries.
2. NeuralManifoldDynamics supplies one release-bound empirical realization of
   those objects for EEG and fMRI ingest.
3. Downstream analysis remains responsible for group contrasts, falsification,
   baseline comparison, and any stronger theoretical interpretation.

#pagebreak()

= References
#bibliography("NeuralManifoldReferences.bib")


= Technical Terms
#par(first-line-indent: 0pt)[*NeuralManifoldDynamics*: The ingest-layer measurement system in this repository for constructing and serializing manifold-proxy trajectories and optional Jacobian-based summaries from neural data.]
#par(first-line-indent: 0pt)[*mnps_3d*: The canonical three-dimensional trajectory exported with fixed axis order [m, d, e].]
#par(first-line-indent: 0pt)[*coords_9d*: The stratified coordinate matrix containing nine subcoordinates that refine the canonical 3D chart.]
#par(first-line-indent: 0pt)[*Meta-Noetic Jacobian (MNJ)*: A local Jacobian estimate defined on chart trajectories and, when enabled, used to summarize local dynamical structure.]
#par(first-line-indent: 0pt)[*Anisotropy*: A summary of directional imbalance in a Jacobian or block-Jacobian field.]
#par(first-line-indent: 0pt)[*Current Source Density (CSD)*: A spatial filtering transform for EEG intended to reduce broad field spread and emphasize more local cortical activity.]
#par(first-line-indent: 0pt)[*Regional MNPS*: A regionalized trajectory representation in which trajectories and Jacobians are computed separately for networks or channel groups.]
#par(first-line-indent: 0pt)[*Block Jacobian*: A Jacobian summary restricted to a block or family of coordinates, such as m-to-m or e-to-d interactions.]
#par(first-line-indent: 0pt)[*Coverage policy*: The rule set that determines whether an epoch or trajectory has sufficient support to be retained for measurement.]
#par(first-line-indent: 0pt)[*NWB*: Neurodata Without Borders, a neurophysiology data format supported here through a constrained adapter path for selected electrical-series assets.]
#par(first-line-indent: 0pt)[*Derived event layer*: A sidecar or derived HDF5 group containing event annotations, event-window mappings, matched controls, and event-locked summaries that are joinable to, but separate from, the canonical MNPS measurement output.]
#par(first-line-indent: 0pt)[*Measurement contract*: The fixed export definition that specifies what is measured, how it is named, and how it is serialized for downstream use.]

