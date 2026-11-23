#import "template_eLife.typ": *
#set page(numbering: "1")

#show: netn-template.with(
  title: "NeuralManifoldDynamics: A Versioned Measurement System with Modality-Specific EEG, MEG, and LFP Branches",
  short-title: "Modality-specific neural measurements",
  subtitle: "An EEG measurement contract with separate MEG and LFP pilots",
  authors: (
    (name: "Robin Langell", affil: 1),
  ),
  affiliations: (
    "Langell Konsult AB, Vallentuna, Sweden",
  ),
  corresponding-author: "Robin Langell, hello@noeticdiffusion.com",
  keywords: ("EEG", "MEG", "LFP", "measurement contract", "signal processing", "robust scaling", "data provenance", "reproducibility"),
  article-type: "Methods",
  abstract: [
    NeuralManifoldDynamics is an experimental measurement system that records modality-specific neural measurements with explicit support and provenance. We specify MNPS 3.4, its flat nine-coordinate EEG definition, participant-balanced robust anchors, HDF5 representation, and reference-cohort exports from OpenNeuro ds005385. We also report fixed-anchor, three-session repeatability in 60 participants from ds004148: across 54 EC/EO coordinate-level session comparisons, descriptive ICC(A,1) point estimates ranged from 0.041 to 0.954; across 27 EO−EC contrast comparisons, they ranged from 0.249 to 0.864. Separate engineering pilots demonstrate a MEG derived-export smoke on one ds003645 recording and LFP contact-feature computation plus support audits using DANDI 000458 and metadata-only DANDI 000232. EEG repeatability varied by coordinate and session; the LFP pilot yielded no supported real-data local dynamics estimates or paired held-out predictions. The separate modality branches do not establish a shared cross-modal coordinate system, measurement equivalence, artifact-free signals, biological stability, diagnostic value, or predictive performance.
  ],
  author-summary: [
    EEG processing can produce many numbers while leaving unclear which channels, filters, time windows, reference values, and missing-data rules created them. This paper describes an experimental nine-coordinate EEG definition and reports its extraction, scaling, support, provenance, and a three-session assessment in 60 participants using one fixed external EC reference for both EC and EO recordings. It also summarizes separate MEG and LFP engineering branches. Each modality keeps its own measurement realization; the examples show selected computations and support limits, not cross-modal equivalence or clinical usefulness.
  ],
  data-availability: [
    The core EEG reference and repeatability analyses use OpenNeuro ds005385, version 1.0.3 @openneuro_ds005385_v103, and ds004148, version 1.0.1 @openneuro_ds004148_v101. Separate engineering checks use OpenNeuro ds004511, version 1.0.2, for a single-recording EEG dynamics smoke; ds003645 for one MEG FacePerception run; and DANDI 000458, version 0.240510.2038, for an LFP contact-feature and support pilot. DANDI 000232 / EF0147 was reviewed using metadata only; no signal values were read and no features were extracted. No new recordings were collected. The v3.4 implementation and frozen anchor manifests are in mndm/src/mndm/mnps/v3_4 and mndm/config/v.3.4. The ds005385 cohort outputs and measurement-quality audit are in project/mnps2_audit/outputs/20261111_mnps_v3_4_ds005385_baseline_cohort and project/mnps2_audit/outputs/20261116_mnps_v3_4_measurement_quality_audit. The version-controlled ds004148 full-cohort repeatability package is in mndm/docs/ds004148_session_repeatability_fullcohort; the ds004511 smoke is in project/mnps2_audit/outputs/20261119_mnps_v3_4_ds004511_dynamics_integration_test. Generated HDF5 files and source EEG binaries remain in local output/data locations and are not included in those documentation packages. The v3.4 definition is experimental and the repository package metadata reports 3.1.0; this manuscript does not assign a new package version or archival DOI.
  ]
)

= Introduction

Electroencephalography (EEG) is often reduced to spectral and temporal summaries before those values reach a downstream study. A number alone does not identify its signal-processing path. Channel set, reference, filtering, window boundaries, estimator parameters, missingness, and scaling reference can all change what a value represents. An auditable *measurement contract* must therefore describe not only a numeric result but also the realization and support that produced it.

NeuralManifoldDynamics (NMDM) is an ingest and measurement system. Its scientific boundary separates *computability* (whether an estimator has its declared inputs), *comparability* (whether two values share compatible realizations), and *qualification* (whether they meet a study's requirements for a stated use). These are separate decisions. A measurement can be computable while its support is incomplete, its sensor realization differs from another recording, or its scientific qualification remains unresolved.

Here we specify MNPS 3.4, an experimental EEG measurement definition implemented in NeuralManifoldDynamics. The term *flat* means that nine ordered components are emitted side by side: the software does not group them into higher-level families, project them to a lower-dimensional chart, or declare a default distance geometry. Each component retains its own source, unit, estimator, validity, support, and provenance. The purpose of this paper is to state that implementation as a versioned, inspectable contract and report the reference-cohort export used to freeze its current technical anchors.

Within the broader Noetic Diffusion Theory (NDT) research programme, NeuralManifoldDynamics provides an empirical measurement layer: NDT frames questions about neural states and their possible futures, while NMD specifies reproducible observations, time support, scaling, validity, and provenance. MNPS 3.4 is one candidate observation chart for that work, not a mathematical prerequisite for neural-dynamics analysis in general; another adequately characterized chart could serve the same role @langell2026_ndt.

This article presents an experimental measurement system through a primary EEG definition and separate modality-specific engineering branches. It does not re-present the historical NMDM model or imply that EEG, MEG, and LFP values share a coordinate frame. Their version-specific constructs and changes are summarized in Supplementary Changelog S6. The paper specifies the nine EEG measurements and their extraction order; documents condition-specific robust references and the output contract; reports dataset-specific EEG repeatability; and describes bounded MEG/LFP engineering checks with their distinct support and evidence limits.

= Results

== The contract emits nine ordered window-level measurements

The payload is named mndm.mnps_v3_4.experimental_flat_9d.v1 and is written under #raw("/measurements/v3_4/"). The order is fixed and is stored with the values. Table 1 gives each component's source rule before frozen reference scaling.

#figure(
  clean-table(
  columns: (auto, 1fr, 1fr),
  header: ([*Order*], [*Measurement*], [*Source rule and raw unit*]),
  inset: 4pt,
  [1], [#raw("delta_theta_level")], [0.5 log10(P_delta / 1 μV²) + 0.5 log10(P_theta / 1 μV²); dimensionless log-power],
  [2], [#raw("alpha_level")], [log10(P_alpha / 1 μV²); dimensionless log-power],
  [3], [#raw("beta_alpha_ratio")], [Common EEG beta-to-alpha ratio; dimensionless],
  [4], [#raw("gamma_level")], [log10(P_gamma / 1 μV²); dimensionless log-power],
  [5], [#raw("hjorth_mobility")], [sqrt(var(Δx) / var(x)); sample-difference ratio, without division by Δt @hjorth1970],
  [6], [#raw("alpha_theta_ratio")], [Common EEG alpha-to-theta ratio; dimensionless],
  [7], [#raw("permutation_entropy")], [Normalized order-5, delay-1 permutation entropy; dimensionless],
  [8], [#raw("alpha_centroid_hz")], [7-13 Hz power-weighted spectral centroid; Hz],
  [9], [#raw("alpha_envelope_cv")], [Median across channels of the alpha-envelope coefficient of variation; dimensionless],

  ),
  kind: table,
  caption: [Nine ordered measurement components and source definitions before frozen feature and coordinate reference scaling.],
)

The first seven components reuse named outputs from the common EEG feature path @gramfort2013_mne. Before window-level feature calculation, this path takes a pointwise median across the available EEG channels. Two candidates are calculated on the native channel realization: a band-limited spectral centroid and a channel-wise alpha-envelope coefficient of variation. A third candidate, spatial log-variance dispersion, is exported as an auxiliary measurement and is not one of the nine coordinates. The source values and transformed values remain distinguishable in the output. Power is expressed in μV² before the logarithm, so each log-power is defined relative to 1 μV².

== Rationale and expected dependencies among the nine measurements

The components cover several descriptive summaries of the same multichannel recording: band-limited power levels (delta/theta, alpha, and gamma), relative spectral balance (beta/alpha and alpha/theta), temporal sample-to-sample structure (Hjorth mobility and permutation entropy), and two alpha-specific summaries (a band power-weighted centroid and envelope variability). The intent is to preserve these named measurement views in a versioned export, not to claim that they are sufficient statistics or orthogonal dimensions of neural activity.

Dependence is expected. Delta/theta level, alpha level, beta/alpha ratio, alpha/theta ratio, and the alpha centroid reuse related spectral information. Band summaries can also combine periodic peaks and aperiodic background power, so they are descriptive spectral measurements rather than isolated oscillatory components @donoghue2020_specparam. Hjorth mobility and permutation entropy also depend on the time-series structure of the same aggregated trace. The flat nine-coordinate interface therefore does not imply nine statistically independent degrees of freedom. Correlation diagnostics are reported below using participant-level summaries. Correlated summaries are retained because they are separately defined measurements with distinct formulas, raw units, and support behavior; downstream analyses can select or model them for a stated question.

== Spatial aggregation is part of the measurement definition

For the first seven components, the inherited EEG path forms a pointwise median signal across the available, average-referenced channels and then computes window features. This preserves that established extraction path as an explicit v3.4 realization. It is not equivalent to computing each feature per channel and taking a channel median afterward: nonlinear feature extraction and spatial aggregation generally do not commute. The pointwise median is reference- and topography-dependent, and signed channel values can cancel or change with spatial configuration @bigdelShamlo2015_prep. The current audit does not compare this global trace against per-channel feature aggregation or a posterior-channel summary. Accordingly, the paper claims implementation of this specific realization; it does not claim that the aggregation is optimal or interchangeable with those alternatives.

No component is automatically interpreted as a biological axis. The flat coordinate payload does not aggregate the nine values or declare one cross-coordinate distance. Separate optional v3.4 Jacobian and dynamical-family exports can operate on a declared coordinate realization; they are additive measurements, not part of the nine-coordinate vector definition.

== Explicit references define the scaled coordinate frame

Scaling uses two distinct stages. For source feature q_j, the declared transform g_j produces u_j, the frozen feature anchor produces z_j, fixed coefficients a_kj assemble coordinate c_k, and the frozen coordinate anchor produces x_k:

$ u_j = g_j(q_j), quad z_j = (u_j - μ_j) / s_j, quad c_k = sum_j a_(k j) z_j, quad x_k = (c_k - ν_k) / r_k. $

Here μ_j and s_j are feature-reference center and scale; ν_k and r_k are coordinate-reference center and scale. For delta/theta, the raw source coordinate is 0.5 log10(P_delta / 1 μV²) + 0.5 log10(P_theta / 1 μV²), while the assembled feature-scaled coordinate is 0.5 z_delta + 0.5 z_theta. Those formulas are not generally equal because the feature anchors can have different centers and scales. The anchor manifest records both stages, the measurement definition, units, estimator identity, preprocessing surface, channel realization, reference, and implementation hashes.

For a one-feature coordinate and identical supported reference observations and participant weights at both stages, the second median/MAD normalization is algebraically identity-like, up to floating-point error. A synthetic test checks this property. The coordinate anchor remains explicit as a uniform output interface, but it does not add information for those one-to-one mappings. The multi-feature delta/theta coordinate is not covered by that identity result.

The robust center is a participant-balanced median. The robust scale is 1.4826 times the median absolute deviation. Within each participant, finite supported windows share that participant's total weight, so a participant with more usable windows does not receive more total weight. A zero, missing, or otherwise invalid scale leaves the affected anchored values unsupported; the implementation does not substitute an epsilon. Non-positive power inputs to log10 coordinates are unsupported. Coordinates are not clipped or winsorized, and no recording-local refit is performed.

A target run can use a registered external reference or an explicitly configured reference manifest. Strict compatibility checks the declared measurement definition, estimator, units, preprocessing surface, channel set, and reference identity. Known channel-order differences remain QC when the channel set matches. An exploratory transfer requires a reason and records the mismatch; it does not convert units or establish equivalence between modalities.

#figure(
  image("figures/mnps_v3_4_measurement_flow_draft.svg", width: 100%),
  caption: [
    The implemented sequence from window features through frozen feature and coordinate references to the support-aware flat output. The flat coordinate payload preserves raw values and provenance and emits no family aggregation or lower-dimensional projection. Separate optional Jacobian and dynamics groups may coexist under `/measurements/v3_4/` without changing the coordinate order or values.
  ],
)

== The reference run produced two separate technical frames

The reference cohort was selected from session-1, pre-task recordings in OpenNeuro ds005385 version 1.0.3 @openneuro_ds005385_v103. The metadata-based selection included 608 participants with one eyes-closed and one eyes-open recording each, for 1,216 recordings. Feature extraction produced 55,183 window rows: 27,651 eyes-closed and 27,532 eyes-open. Preprocessing used all 64 EEG channels, average reference, a 50 Hz notch, a 1-45 Hz bandpass, resampling to 250 Hz, and 8 s windows advanced by 4 s. The alpha-envelope estimator additionally required a 10 s segment-edge guard.

One anchor was fitted for each condition using the 608 participants in that condition. Each anchor was then applied to those same 608 recordings, with no anchor identity failures reported. These are transductive fits, not held-out evaluation. The eyes-open and eyes-closed anchors have different centers and scales and therefore define different coordinate frames. Their scaled values cannot be pooled as if they shared one origin and unit. Applying one common anchor to both target conditions is possible, but it is a separate, explicitly selected analysis design.

A descriptive inventory summarized existing numeric HDF5 outputs by full measurement path and realization. It reported mean, population standard deviation, median, median absolute deviation, scaled median absolute deviation, quartiles, interquartile range, minimum, maximum, skewness, excess kurtosis, and support counts. The broader inventory also included other measurement paths in the enclosing files; each path and realization retains its own estimator identity and is not relabeled as a v3.4 measurement. No inferential eyes-open versus eyes-closed test was performed.

== Coordinate-specific computability and contract validity

The v3.4-only audit separated raw estimator support, coordinate validity under the contract, and whole-vector completeness. Table 2 reports numerically supported-window counts from the selected HDF5 exports. For all nine measurements, each condition included supported values from all 608 participants. The first eight measurements passed their coordinate validity rules in every exported window. The only observed support loss was for `alpha_envelope_cv`, where windows inside the declared edge guard were unsupported. All participants had at least one complete nine-coordinate window. Contract validity does not indicate artifact-free EEG.

#figure(
  clean-table(
    columns: (1.2fr, auto, auto, auto, 2fr),
    header: ([*Measurement*], [*Eyes closed*], [*Eyes open*], [*Participants*], [*Observed support loss*]),
    inset: 4pt,
    [#raw("delta_theta_level")], [27,651 / 27,651], [27,532 / 27,532], [608 / 608 each], [None in exported support mask],
    [#raw("alpha_level")], [27,651 / 27,651], [27,532 / 27,532], [608 / 608 each], [None in exported support mask],
    [#raw("beta_alpha_ratio")], [27,651 / 27,651], [27,532 / 27,532], [608 / 608 each], [None in exported support mask],
    [#raw("gamma_level")], [27,651 / 27,651], [27,532 / 27,532], [608 / 608 each], [None in exported support mask],
    [#raw("hjorth_mobility")], [27,651 / 27,651], [27,532 / 27,532], [608 / 608 each], [None in exported support mask],
    [#raw("alpha_theta_ratio")], [27,651 / 27,651], [27,532 / 27,532], [608 / 608 each], [None in exported support mask],
    [#raw("permutation_entropy")], [27,651 / 27,651], [27,532 / 27,532], [608 / 608 each], [None in exported support mask],
    [#raw("alpha_centroid_hz")], [27,651 / 27,651], [27,532 / 27,532], [608 / 608 each], [None in exported support mask],
    [#raw("alpha_envelope_cv")], [24,361 / 27,651], [24,215 / 27,532], [608 / 608 each], [3,290 EC and 3,317 EO windows inside the 10 s segment-edge guard],
  ),
  kind: table,
  caption: [Valid windows by condition, participants with any valid window, and observed support loss. Each window is counted once in its recording; overlapping windows are not independent participants.],
)

Across the 55,183 windows, coordinate-specific anchor application was reported as `computed` for all 1,216 recordings. The complete-nine mask was true for 24,361/27,651 eyes-closed windows (88.1%) and 24,215/27,532 eyes-open windows (88.0%). All recordings remained unqualified, and artifact QC was unassessed (`qc_ok_eeg = -1`) for every window. These results describe data availability and application behavior, not signal cleanliness or scientific suitability.

== The nine participant-level measurements are correlated

To avoid treating overlapping windows as independent participants, the dependency audit summarized one median per participant, coordinate, and condition under two window-support policies. Available-coordinate medians use every valid window for each coordinate; matched medians use only windows where all nine coordinates are supported. Both policies retained the same 608 participants per condition. Effective ranks of the Spearman correlation matrices were 4.3649 (available) and 4.3655 (matched) for eyes-closed, and 4.7141 and 4.7185 for eyes-open. Mean absolute matrix differences were 0.0024 and 0.0025; maximum absolute differences were 0.0091 and 0.0101. Participation ratios changed from 3.1687 to 3.1651 and from 3.4594 to 3.4575. These small support-policy differences describe this export; they are not feature-importance tests or estimates of intrinsic neural-state dimension. The matrices are Spearman correlation matrices, not covariance matrices. Positive affine anchor scaling does not change within-condition Spearman ranks, so the separate EO and EC anchors do not explain differences between these within-condition matrices.

#figure(
  image("figures/mnps_v3_4_dependency_heatmap.svg", width: 100%),
  caption: [Spearman correlations among participant-level medians under available-coordinate and matched complete-nine window support, shown separately for eyes-closed and eyes-open (608 shared participants per condition). Abbreviations map to the Table 1 names: δ/θ, alpha level, β/α, gamma level, Hjorth mobility, α/θ, permutation entropy, alpha centroid, and alpha envelope CV.],
)

== Output groups retain support and provenance

In a combined HDF5 file, the v3.4 payload is separate from other measurement groups. The core arrays include raw features, transformed feature values, raw coordinate inputs, anchored coordinates, ordered names, window start and end times, per-coordinate validity, and a *complete-nine-coordinate mask*. Per-channel candidate values are preserved when supported. Support metadata records the ordered labels and *channel-set hash*, channel count, reference and reference-set identity, source surface, code implementation hashes, and QC. Current writes also record the realized PSD method ID and fallback reason per window under `/measurements/v3_4/support/`; these fields were absent from the frozen cohort exports audited here. Anchor selection, compatibility mode, hashes, transfer reason, qualification, and failure status are separately recorded.

Finite values do not imply successful QC or scientific qualification. If an anchor cannot be applied, raw measurements and QC remain available while the anchored values are marked invalid with a reason. Partial coordinates are retained. The complete-nine-coordinate mask answers whether all nine components are valid for that window; it does not establish comparability with a second recording.

The summarize pipeline can emit the default-on 9D v3.4 local Jacobian at `/measurements/v3_4/jacobian`; this does not replace the older `/jacobian` or `/jacobian_9D` groups. A separate opt-in dynamical-family export writes to `/measurements/v3_4/dynamics`. Both are independently versioned from the flat coordinate arrays. Other Jacobian paths retain independent switches, so disabling the v3.4 Jacobian candidate alone does not suppress every Jacobian path. Supplementary Methods S1 and Changelog S6 describe the output identities and the bounded ds004511 test.

== Evidence supports implementation, not biological qualification

The ds005385 reference work completed feature extraction across the selected cohort, fitted both registered anchors, and produced a descriptive HDF5 statistics inventory. This establishes that the reference artifacts and their recorded cohort summaries exist. It does not establish raw-file byte identity: the cohort matching used metadata and unique manifest basenames, and raw EDF content hashes were not computed. Artifact-level EEG quality was also not established by this metadata-only reference selection.

The ds005385 reference-cohort work is complemented by a full-cohort, fixed-anchor repeatability assessment in OpenNeuro ds004148, a separate dataset whose 60 selected participants were not used to fit the reference. The selection fixed all session 1–3 resting EC/EO recordings from metadata before coordinate inspection. Session-1 HDF5/QC outputs were reused; sessions 2–3 were re-extracted. All 360 recordings received the same frozen ds005385 EC anchor, with no refitting, and QC matched 26,640 windows. The prespecified support floor was met by 2,858/3,240 participant-session-condition-coordinate summaries; complete nine-coordinate support passed in 306/360 participant-session-condition summaries. Across 54 eligible EC/EO coordinate-level session comparisons, ICC(A,1) ranged from 0.041 to 0.954 (43–52 participants per row). Across 27 EO−EC comparisons using the same anchor, it ranged from 0.249 to 0.864 (36–42 participants per row). Alpha-centroid level ICC point estimates ranged from 0.762 to 0.954; alpha-envelope-CV EO−EC estimates ranged from 0.793 to 0.864. All anchored coordinates, including `alpha_centroid_hz`, are dimensionless reference-relative values. These are descriptive, coordinate- and session-dependent estimates, not qualification thresholds or evidence that one coordinate is superior. The support/QC rule is an operational eligibility screen and does not prove artifact absence. The assessment establishes technical fixed-anchor application and dataset-specific repeatability behavior; it does not establish biological stability, cross-dataset equivalence, or clinical or predictive utility.

== Separate MEG and LFP branches demonstrate bounded modality-specific computation

The LFP pilot demonstrates contact-level 3–5 Hz centroid, envelope-CV, and bandpower computation in one DANDI 000458 Probe F/MOs session. Its observed chart keeps two depth groups as separate, session-locally calibrated 3D trajectories. The finite-lag analysis produced no supported real-data local estimates or paired held-out predictions because the eligible transition rows came from too few distinct source batches under the frozen project heuristic. A metadata-only audit of DANDI 000232 / EF0147 was deferred under the same protocol; this does not establish that the data are unsuitable for other LFP measurements. These are LFP-specific results, not coordinates in the EEG 9D definition or evidence of cross-modal transfer. Full methods and support limits are in Supplementary Methods S1.8.1.

The MEG pilot used one OpenNeuro ds003645 FacePerception run and kept its magnetometer (MAG) and gradiometer (GRAD) realizations separate. Across 121 windows, raw coordinate values and support masks for three variants matched the cached feature table to an absolute tolerance of `1e-12`; no EEG anchor or recording-local normalization was applied. This smoke demonstrates derived-export integration for that recording. The MEG variants have their own schema and sensor identity; overlapping component labels do not make their values interchangeable with EEG. Mixed source units and the separate sensor realizations do not establish a qualified metric, MAG/GRAD comparability, cross-modal equivalence, repeatability, or predictive validity. Full results are in Supplementary Methods S1.8.2.

= Discussion

== A measurement contract is useful only with its realization

The central result is an explicit accounting surface: a fixed coordinate order is paired with declared feature transforms, two stages of robust references, per-window support, and recording-level sensor and implementation identity. This structure permits a later analysis to select a subset, stratify by realization, or exclude unsupported windows with the relevant evidence in hand. It does not determine which measurements should answer a biological question.

The two reference presets illustrate why a label such as eyes-open or eyes-closed is not enough to make values comparable. Different condition-specific centers and scales create separate frames. The shared participant cohort also means the reference values contain information from target recordings. A study that requires a common frame or held-out normalization must predeclare that reference and apply it consistently.

The broader system view also does not imply a shared multimodal chart. The EEG, MEG, and LFP branches preserve distinct sensors, units, references, and support rules. The MEG and LFP examples establish selected computation and export behavior only; no cross-modal coordinate comparison was performed or qualified.

== Relationship to NDT and downstream dynamical inference

NMD is intended to make downstream dynamical questions empirically inspectable, rather than to supply a new feature table alone. NDT treats a measured state chart as an observation interface to hypotheses about local transitions, fluctuation spread, and context-dependent future laws. MNPS 3.4 contributes a candidate interface with explicit temporal support and realization metadata. It does not make NDT true, and NDT does not depend on this particular chart; other qualified measurement representations could be used.

Local dynamical analyses require a fitted transition or flow field in a declared chart. A spatial Jacobian $J = partial f / partial x$ describes the local derivative of that fitted chart-level field. When estimated in an MNPS chart it is termed a Meta-Noetic Jacobian (MNJ). It is distinct from temporal acceleration $d^2 x / (d t)^2$ and from the time derivative $dot(J)$. The optional v3.4 Jacobian is a local estimate on the declared nine-coordinate layer. The separate v3.4 dynamics export estimates local drift and increment covariance on raw or actually anchored layers, and offers a lag-1 one-step map only for an applied anchored layer. These estimators do not turn the flat coordinate definition into a validated transition model. In the bounded ds004511 Rest smoke, raw drift and diffusion were computable, but no anchor was applied and no one-step result was emitted. The smoke establishes export integration for one recording, not local identifiability, predictive validity, comparability, or scientific qualification. Temporal resolution, observation noise, neighborhood support, estimator fidelity, coordinate dependence, and held-out validation remain separate requirements; details and counts are in Supplementary Methods S1.

The longer-term research sequence is therefore conditional on evidence at each layer:

#figure(
  clean-table(
    columns: (1.05fr, 2.2fr, 1.45fr),
    header: ([*Layer*], [*Question*], [*Status in this paper*]),
    inset: 4pt,
    [Measurement], [Can the declared coordinates be computed and reproduced with visible support and provenance?], [Characterized for the selected exports.],
    [Representation], [Are summaries stable and informative under declared temporal, sensor, and reference choices?], [Partly described; robustness remains open.],
    [Local dynamics], [Can a transition model predict held-out evolution, and can a chart-level MNJ be estimated reliably?], [A 9D local Jacobian and raw drift were emitted in one smoke; no anchored one-step or held-out evaluation.],
    [Higher-order dynamics], [Can changes in local operators or stochastic spread be estimated and validated?], [Raw local increment covariance was computed in one smoke; validation remains open.],
    [NDT hypotheses], [Do dynamical summaries add future information beyond current state (C2), or vary across contexts at matched state (C3)?], [Not tested.],
  ),
  kind: table,
  caption: [Evidence layers from measurement to downstream tests; completion of one layer does not establish the next.],
)

The contract also differs in scope from neighboring EEG infrastructure. EEG-BIDS standardizes data organization and metadata, and MNE-Python supplies widely used preprocessing and feature-estimation tools; MNPS 3.4 builds on such tools rather than replacing them. Its specific contribution is the integrated contract for frozen participant-balanced feature and coordinate references, per-coordinate support, sensor/reference realization identity, and a separate qualification state. An ordinary feature table can hold the same numerical features, but those additional distinctions must be defined and retained explicitly for downstream reuse @Appelhoff2019MNEBIDS @gramfort2013_mne. In the frozen cohort exports examined here, implementation hashes are present but the runtime choice between multitaper PSD and its Welch fallback was not serialized; the current writer now records that choice for future exports, while the existing values cannot be retrospectively assigned to a backend from these HDF5 files.

#figure(
  clean-table(
    columns: (1.15fr, 2.65fr),
    header: ([*Adjacent infrastructure*], [*What the MNPS 3.4 export makes explicit*]),
    inset: 4pt,
    [EEG-BIDS], [MNPS does not replace standardized source organization; it records the measurement realization and output support for an analysis built from selected recordings.],
    [MNE-Python and related estimators], [MNPS declares transforms, frozen references, support, and code identity. A current cohort cache lacks the runtime PSD backend field, so its exact PSD branch is not retrospectively known.],
    [Feature table], [MNPS preserves raw and transformed values alongside masks, sensor/reference identity, failure reasons, and qualification state.],
    [Dynamical analysis], [The flat 9D definition does not include these estimators. Optional v3.4 Jacobian, drift, diffusion, and anchored lag-1 one-step groups are separate outputs; the bounded smoke does not establish predictive validity or an NDT claim.],
  ),
  kind: table,
  caption: [MNPS 3.4 is an additive measurement contract that complements data standards and analysis software.],
)

== A fixed feature representation does not establish a neural manifold

The EEG-specific MNPS 3.4 contract emits nine ordered operational measurement components for downstream analyses of windows and trajectories. The use of *coordinate* here denotes an ordered component in the versioned output, not an identified latent dimension or a validated manifold chart. The flat coordinate payload alone does not establish a biologically meaningful distance, identify latent state, or validate a dynamical system. Optional local Jacobian and dynamical-family estimates remain chart-dependent measurements with their own support and validation requirements. The versioned output contract and relationship to earlier software generations are summarized in Supplementary Changelog S6.

== Limits and next validation

The ds005385 reference-cohort selection did not establish artifact QC, and raw EDF hashes and header calibration were not verified for that audit. In ds004148, predeclared QC and support rules determine which participant summaries enter repeatability estimates; passing those rules does not establish artifact-free signals. Repeatability is uneven across coordinates and session pairs, and the analysis does not establish biological stability, measurement equivalence across datasets, or qualification for a downstream use. Frozen ds005385 derived exports retain per-channel candidates for centroid and envelope summaries but not raw waveforms or per-channel values for the first seven common features; the source EDFs were unavailable in that audit, so its spatial-aggregation sensitivity cannot be reconstructed. Further work should target a declared measurement question and source-data availability rather than tune coordinates to a preferred contrast.

Future validation should assess measurement-specific reliability, sensitivity to declared preprocessing choices, sensor/reference compatibility, and missingness under a predeclared use case. EEG reliability can differ by estimator and summary property; alpha-asymmetry results across electrode pairs and EEG systems do not establish reliability for a global centroid or envelope summary @metzen2022_alpha_asymmetry. Prior microstate work likewise illustrates that reliability must be measured rather than assumed @kleinert2024_microstate. Alpha-frequency studies report condition-dependent estimator stability, but their peak estimators are distinct from the v3.4 band centroid @han2026_alpha. Preprocessing standardization efforts provide adjacent infrastructure rather than biological validation of this contract @boettcher2026_clean_eeg. Such work should not tune estimators to create a preferred condition contrast. Until that evidence exists, v3.4 should be treated as an experimental measurement definition with transparent technical references, not as a qualified neural state or a clinical or predictive instrument.

= Materials and Methods

== Software and definition identity

The repository package metadata is 3.1.0. The core EEG coordinate definition and its reference-cohort evidence are frozen for further evaluation; separate optional Jacobian and dynamical-family exports are described in Supplementary Methods S1. A distinct experimental MEG surface and the LFP contact-feature/dynamics branch are modality-specific engineering paths summarized in S1.8.1–S1.8.2; neither is part of the EEG nine-coordinate schema or evidence of cross-modal coordinate equivalence. MNPS 3.4 is a measurement definition under the package, not a package release. Its flat EEG coordinate schema is `mndm.mnps_v3_4.experimental_flat_9d.v1` with definition version 3.4. The common EEG and ds005385 configurations, registered anchor JSON files, and source implementation identify that coordinate realization. No public release tag or Zenodo record for this measurement definition is asserted here.

== Dataset selection and preprocessing

The reference-cohort source was OpenNeuro ds005385 version 1.0.3 @getzmann2024_ds005385. Its dataset README warns that physical minimum/maximum specifications in some EDF headers may be invalid and should be ignored @openneuro_ds005385_readme. The source EDFs were unavailable in the derived-export directory, so header calibration and raw-signal identity were not independently verified. Metadata and required sidecars selected session-1 pre-task recordings in both resting conditions, with one recording per condition for 608 participants. The 64-channel EEG was average referenced without common-montage interpolation. Core preprocessing used the versioned filters, resampling to 250 Hz, and 8 s windows advanced by 4 s; exported start/end times remain authoritative, and windows do not cross declared discontinuities. Full selection and preprocessing details are in Supplementary Methods S1.2.

The session-repeatability source was OpenNeuro ds004148 version 1.0.1 @openneuro_ds004148_v101, described as a test-retest resting and cognitive-state EEG dataset. The metadata-only selection fixed 60 participants and all session 1–3 resting EC/EO pairs before inspecting features or coordinates. The registered ds005385 EC anchor was applied unchanged to every recording; no ds004148 data were used to refit it. Session-1 HDF5/QC outputs were reused, and sessions 2–3 were processed with the versioned EEG configuration. Eligibility required at least 30 QC-eligible windows and 120 unique seconds per participant, session, condition, and coordinate. Comparisons additionally required compatible declared sensor and signal-processing realizations. These criteria define the analyzed support; they do not certify artifact absence.

Separate engineering checks used one ds004511 EEG recording for the optional raw dynamics smoke, one ds003645 FacePerception MEG run for derived-export integration, and one DANDI 000458 LFP session for contact-level features and support-limited dynamics. The DANDI 000232 / EF0147 assessment used metadata only. These pilots were not used to fit the EEG anchors or estimate EEG repeatability, and their modality-specific outputs are not pooled or compared as a shared chart. Supplementary Methods S1.8 describes each pilot.

The 50 Hz notch and 1–45 Hz EEG bandpass are separate MNE operations; no equivalence between their effects is assumed. The alpha-envelope candidate adds a continuous-segment 8–12 Hz zero-phase filter and Hilbert transform with a 10 s edge guard. These offline operations and their candidate-specific support are detailed in S1.2–S1.3; the guard does not remove other computable coordinates.

== Feature estimators and coordinate assembly

The first seven measurements use the common EEG path, which takes a pointwise median across available average-referenced channels before window-level feature calculation @gramfort2013_mne. This reference- and topography-dependent reduction is not equivalent to computing each channel's feature before aggregation. Exact estimator parameters are in S1.3. The frozen cohort exports do not record whether each PSD used the MNE multitaper method or its SciPy Welch fallback, so those values cannot be retrospectively assigned to a backend. The alpha centroid is a band summary, not a peak detector; alpha-envelope CV measures envelope variability, not bursts. Spatial log-variance dispersion is auxiliary and is not part of the nine coordinates.

Table 1 lists the nine source definitions. Log-power uses μV² and a 1 μV² reference; non-positive log inputs or ratio denominators are unsupported, with no floor substituted. Hjorth mobility is a sample-difference ratio without division by Δt, not a physical derivative. Gamma power from 30–45 Hz is computable under the declared estimator but is not necessarily a clean neural-gamma measure without independent muscle-artifact assessment. Feature transforms, candidate support rules, and units are detailed in S1.3.

== Robust reference fitting

Feature and coordinate anchors use participant-balanced weighted medians and 1.4826-scaled median absolute deviations on finite supported observations. The reference manifests record fit counts and status; insufficient or degenerate fits remain unsupported rather than imputed. The ds005385 eyes-closed and eyes-open anchors were fitted separately and applied to the same 608 participants in their respective condition, making each fit transductive. They define technical reference frames, not normative population parameters or one shared frame. Full weighting, fit, and identity details are in S1.4–S1.5.

== Validity, support, and output schema

The v3.4 group stores values, per-coordinate validity, complete-nine support, QC, and provenance separately at the window level. Failed anchor applications preserve raw measurements and QC while invalidating anchored values with a reason; unsupported values are not zero-filled. S1.6–S1.7 list the HDF5 paths and fields.

== Cohort inventory

The broad read-only cohort inventory enumerated numeric HDF5 paths, finite components in serialized JSON, support/QC counts, and recording identities. A separate v3.4-only audit reads the 1,216 selected combined HDF5 exports, checks coordinate order, core dataset shapes/dtypes, and support masks, summarizes raw computability, final validity, whole-vector completeness, anchor application, and QC by condition, and calculates available-window and matched-complete-nine participant medians. It reports Spearman correlation-matrix eigenvalue spectra, compares the two support policies, and calculates leave-one-coordinate-out effective-rank diagnostics. EO and EC remain separate. Positive affine anchor scaling does not change within-condition ranks. No inferential condition contrast is fitted. Historical output-path observations are described in Supplementary Changelog S6.

The ds004148 repeatability analysis checks the frozen EC anchor identity and implementation metadata in all 360 HDF5 recordings, matches QC to all 26,640 exported windows, and summarizes participant-level coordinate medians separately by session and condition. For each of three session pairs, it estimates two-way absolute-agreement single-measure ICC, bias, median absolute difference, Bland–Altman limits, and Spearman rank correlation for each coordinate and for EO−EC contrasts. Only participants meeting the predefined support and realization-compatibility criteria in both sessions contribute to a row. The analysis reports coordinate-specific eligibility and excluded-subject reasons; it makes no inferential condition tests. Detailed method, tables, input/run manifests, scripts, anchor hashes, and output hashes are preserved in `mndm/docs/ds004148_session_repeatability_fullcohort/`.

Matching between the metadata selection and HDF5 recording records used unique manifest basenames when relative source paths were not preserved. This is not raw-signal content verification. Raw EDF hashes and artifact QC were not produced by the cohort selection.

== Reproduction commands

From an environment with this repository installed and the ds005385 data available at the configured path, feature extraction and summarization use the same version-specific configuration:

#raw("python -m mndm.cli features --config mndm/config/v.3.4/config_ingest_ds005385_baseline_eo_ec.yml --dataset ds005385\npython -m mndm.cli summarize --config mndm/config/v.3.4/config_ingest_ds005385_baseline_eo_ec.yml --dataset ds005385\npython mndm/src/mndm/tools/mnps_v3_4_measurement_quality_audit.py --h5-root project/mnps2_audit/outputs/20261111_mnps_v3_4_ds005385_baseline_cohort/processed/ds005385/neuralmanifolddynamics_ds005385_20261112_004711 --out project/mnps2_audit/outputs/20261116_mnps_v3_4_measurement_quality_audit --figure-out project/articles/NeuralManifoldDynamics/figures/mnps_v3_4_dependency_heatmap.svg\npython mndm/docs/ds004148_session_repeatability_fullcohort/reproduction/scripts/analyze_ds004148_fullcohort_session_repeatability.py", block: true, lang: "powershell")

The ds005385 feature extraction and anchor fitting were not rerun for this manuscript edit. The ds004148 full-cohort assessment reused session-1 HDF5/QC outputs and re-extracted sessions 2–3, then analyzed all sessions with the frozen EC anchor. The analysis command expects the frozen source data, HDF5 files, and QC manifests at the paths defined in the accompanying script; the run package records their hashes and exact configurations. The article reports the frozen analysis; it does not refit references or claim that eligibility implies artifact quality.

= Acknowledgements

Generative AI tools assisted with manuscript structure and language editing under human direction. The author remains responsible for the claims and final text.

= Supporting Information

Supplementary Methods S1 details the EEG contract and separate MEG/LFP engineering pilots. Supplementary Data S2 presents expanded cohort support, dependency, and repeatability results; S3 reports quality states, realization compatibility, and bounded robustness checks; S4 translates the scope and limits for clinical readers. The revised S5 glossary scopes current terms separately from historical routes; S6 documents version history and compatibility. Earlier S2–S4 template sources and their existing PDFs are archived under `project/articles/NeuralManifoldDynamics/old/` as historical snapshots.

#bibliography("NeuralManifoldReferences.bib", style: "apa")

#technical-terms((
  ("measurement contract", [A versioned rule set specifying a measurement and the metadata needed to interpret its output.]),
  ("computability", [Whether a declared estimator can produce a value from its available inputs.]),
  ("comparability", [Compatibility of measurement realizations for a stated comparison.]),
  ("qualification", [Evidence that a measurement is suitable for a specified scientific use.]),
  ("feature anchor", [A frozen reference for centering and scaling transformed source features.]),
  ("coordinate anchor", [A separate frozen reference for scaling assembled coordinate values.]),
  ("window support", [The data and estimator evidence available for a particular time window.]),
  ("transductive reference", [A reference fitted from the same participants who later receive it.]),
  ("channel-set hash", [A digest identifying an ordered or canonicalized set of sensor labels.]),
  ("complete-nine-coordinate mask", [A window-level flag indicating that all nine coordinates are valid.]),
))
