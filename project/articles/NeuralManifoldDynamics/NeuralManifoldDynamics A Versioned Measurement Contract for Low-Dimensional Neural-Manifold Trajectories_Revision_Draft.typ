#import "template_eLife.typ": *
#set page(numbering: "1")

#show: netn-template.with(
  title: "MNPS 3.4: A flat nine-coordinate EEG contract",
  short-title: "Nine-coordinate EEG contract",
  subtitle: "Experimental measurement definition in NeuralManifoldDynamics",
  authors: (
    (name: "Robin Langell", affil: 1),
  ),
  affiliations: (
    "Langell Konsult AB, Vallentuna, Sweden",
  ),
  corresponding-author: "Robin Langell, hello@noeticdiffusion.com",
  keywords: ("EEG", "measurement contract", "signal processing", "robust scaling", "data provenance", "reproducibility"),
  article-type: "Methods",
  abstract: [
    MNPS 3.4 is an experimental, opt-in measurement definition in NeuralManifoldDynamics for producing nine ordered EEG measurements with explicit *window support*, estimator identity, reference scaling, sensor provenance, and validity state. We describe its feature definitions, participant-balanced robust anchors, HDF5 representation, and a reference-cohort run using session-1 pre-task recordings from OpenNeuro ds005385. The cohort comprised 608 participants with eyes-open and eyes-closed recordings. Feature extraction produced 55,183 window rows; condition-specific anchors were fitted and applied to all 608 recordings in each condition. These are *transductive references* because the reference participants also receive the anchors. The output inventory preserves raw values, partial support, QC, and separate qualification flags. The run did not establish artifact quality, raw-file content identity, a shared eyes-open/eyes-closed coordinate frame, or a biological condition effect. MNPS 3.4 is a measurement implementation, not a validated neural state, diagnostic, or predictive model.
  ],
  author-summary: [
    EEG processing can produce many numbers while leaving unclear which channels, filters, time windows, reference values, and missing-data rules created them. This methods paper describes an experimental measurement definition that makes those choices visible for nine window-level EEG measurements. We document how the measures are extracted and scaled, what evidence is written with them, and where they cannot yet support scientific claims. The reference example uses recordings from 608 participants in each of two resting conditions, but the same participants both define and receive the scaling references. The resulting values are therefore technical outputs, not validated population norms or evidence that the conditions differ.
  ],
  data-availability: [
    The source recordings are available as OpenNeuro dataset ds005385, version 1.0.3 @openneuro_ds005385_v103. No new recordings were collected. The v3.4 configuration, anchor manifests, implementation, and derived cohort inventory are contained in the NeuralManifoldDynamics repository under mndm/config/v.3.4, mndm/src/mndm/mnps/v3_4, and project/mnps2_audit/outputs/20261111_mnps_v3_4_ds005385_baseline_cohort. The v3.4 definition is experimental and the repository package metadata reports 3.1.0; this manuscript does not assign a new package version or archival DOI. A release commit and immutable software archive should be identified before external reuse.
  ]
)

= Introduction

Electroencephalography (EEG) is often reduced to spectral and temporal summaries before those values reach a downstream study. A number alone does not identify its signal-processing path. Channel set, reference, filtering, window boundaries, estimator parameters, missingness, and scaling reference can all change what a value represents. An auditable *measurement contract* must therefore describe not only a numeric result but also the realization and support that produced it.

NeuralManifoldDynamics (NMDM) is an ingest and measurement system. Its scientific boundary separates *computability* (whether an estimator has its declared inputs), *comparability* (whether two values share compatible realizations), and *qualification* (whether they meet a study's requirements for a stated use). These are separate decisions. A measurement can be computable while its support is incomplete, its sensor realization differs from another recording, or its scientific qualification remains unresolved.

Here we specify MNPS 3.4, an experimental EEG measurement definition implemented in NeuralManifoldDynamics. The term *flat* means that nine ordered components are emitted side by side: the software does not group them into higher-level families, project them to a lower-dimensional chart, or declare a default distance geometry. Each component retains its own source, unit, estimator, validity, support, and provenance. The purpose of this paper is to state that implementation as a versioned, inspectable contract and report the reference-cohort export used to freeze its current technical anchors.

This article is deliberately narrow. It does not re-present the historical NMDM model or earlier software generations. Their version-specific constructs and changes are summarized in Supplementary Changelog S6. The present paper makes three contributions: it specifies the nine measurements and their extraction order; it documents condition-specific robust references and the output contract; and it states the evidence limits that separate successful serialization from scientific qualification.

= Results

== The contract emits nine ordered window-level measurements

The payload is named mndm.mnps_v3_4.experimental_flat_9d.v1 and is written under #raw("/measurements/v3_4/"). The order is fixed and is stored with the values. Table 1 gives each component's source rule before frozen reference scaling.

#figure(
  clean-table(
  columns: (auto, 1fr, 1fr),
  header: ([*Order*], [*Measurement*], [*Source rule and raw unit*]),
  inset: 4pt,
  [1], [#raw("delta_theta_level")], [0.5 log10(delta power) + 0.5 log10(theta power); log10 microvolt-squared],
  [2], [#raw("alpha_level")], [Log10 alpha power; log10 microvolt-squared],
  [3], [#raw("beta_alpha_ratio")], [Common EEG beta-to-alpha ratio; dimensionless],
  [4], [#raw("gamma_level")], [Log10 gamma power; log10 microvolt-squared],
  [5], [#raw("hjorth_mobility")], [Common EEG Hjorth mobility; dimensionless per sample],
  [6], [#raw("alpha_theta_ratio")], [Common EEG alpha-to-theta ratio; dimensionless],
  [7], [#raw("permutation_entropy")], [Normalized order-5, delay-1 permutation entropy; dimensionless],
  [8], [#raw("alpha_centroid_hz")], [7-13 Hz power-weighted spectral centroid; Hz],
  [9], [#raw("alpha_envelope_cv")], [Median across channels of the alpha-envelope coefficient of variation; dimensionless],

  ),
  kind: table,
  caption: [Nine ordered measurement components and source definitions before frozen feature and coordinate reference scaling.],
)

The first seven components reuse named outputs from the common EEG feature path @gramfort2013_mne. Before window-level feature calculation, this path takes a pointwise median across the available EEG channels. Two candidates are calculated on the native channel realization: a band-limited spectral centroid and a channel-wise alpha-envelope coefficient of variation. A third candidate, spatial log-variance dispersion, is exported as an auxiliary measurement and is not one of the nine coordinates. The source values and transformed values remain distinguishable in the output.

No component is automatically interpreted as a biological axis. In particular, the export does not calculate a composite score, a manifold distance, or local dynamical derivatives. This keeps downstream choices visible instead of embedding an unstated aggregation in the ingest layer.

== Explicit references define the scaled coordinate frame

Scaling uses two distinct stages. First, each source feature is transformed as declared, centered and scaled against a frozen *feature anchor*. The standardized feature values are then assembled into the ordered components. Second, each assembled component is centered and scaled using its own frozen *coordinate anchor*. The anchor manifest records both stages, the measurement definition, units, estimator identity, preprocessing surface, channel realization, reference, and implementation hashes.

The robust center is a participant-balanced median. The robust scale is 1.4826 times the median absolute deviation. Within each participant, finite supported windows share that participant's total weight, so a participant with more usable windows does not receive more total weight. A zero, missing, or otherwise invalid scale leaves the affected anchored values unsupported; the implementation does not substitute an epsilon. Non-positive power inputs to log10 coordinates are unsupported. Coordinates are not clipped or winsorized, and no recording-local refit is performed.

A target run can use a registered external reference or an explicitly configured reference manifest. Strict compatibility checks the declared measurement definition, estimator, units, preprocessing surface, channel set, and reference identity. Known channel-order differences remain QC when the channel set matches. An exploratory transfer requires a reason and records the mismatch; it does not convert units or establish equivalence between modalities.

#figure(
  image("figures/mnps_v3_4_measurement_flow_draft.svg", width: 100%),
  caption: [
    The implemented sequence from window features through frozen feature and coordinate references to the support-aware flat output. The export preserves raw values and provenance and emits no family aggregation, lower-dimensional projection, geometry, or local Jacobian.
  ],
)

== The reference run produced two separate technical frames

The reference cohort was selected from session-1, pre-task recordings in OpenNeuro ds005385 version 1.0.3 @openneuro_ds005385_v103. The metadata-based selection included 608 participants with one eyes-closed and one eyes-open recording each, for 1,216 recordings. Feature extraction produced 55,183 window rows: 27,651 eyes-closed and 27,532 eyes-open. Preprocessing used all 64 EEG channels, average reference, a 50 Hz notch, a 1-45 Hz bandpass, resampling to 250 Hz, and 8 s windows advanced by 4 s. The alpha-envelope estimator additionally required a 10 s segment-edge guard.

One anchor was fitted for each condition using the 608 participants in that condition. Each anchor was then applied to those same 608 recordings, with no anchor identity failures reported. These are transductive fits, not held-out evaluation. The eyes-open and eyes-closed anchors have different centers and scales and therefore define different coordinate frames. Their scaled values cannot be pooled as if they shared one origin and unit. Applying one common anchor to both target conditions is possible, but it is a separate, explicitly selected analysis design.

A descriptive inventory summarized existing numeric HDF5 outputs by full measurement path and realization. It reported mean, population standard deviation, median, median absolute deviation, scaled median absolute deviation, quartiles, interquartile range, minimum, maximum, skewness, excess kurtosis, and support counts. The broader inventory also included other measurement paths in the enclosing files; each path and realization retains its own estimator identity and is not relabeled as a v3.4 measurement. No inferential eyes-open versus eyes-closed test was performed.

== Output groups retain support and provenance

In a combined HDF5 file, the v3.4 payload is separate from other measurement groups. The core arrays include raw features, transformed feature values, raw coordinate inputs, anchored coordinates, ordered names, window start and end times, per-coordinate validity, and a *complete-nine-coordinate mask*. Per-channel candidate values are preserved when supported. Support metadata records the ordered labels and *channel-set hash*, channel count, reference and reference-set identity, source surface, estimator details, and QC. Anchor selection, compatibility mode, hashes, transfer reason, qualification, and failure status are separately recorded.

Finite values do not imply successful QC or scientific qualification. If an anchor cannot be applied, raw measurements and QC remain available while the anchored values are marked invalid with a reason. Partial coordinates are retained. The complete-nine-coordinate mask answers whether all nine components are valid for that window; it does not establish comparability with a second recording.

== Evidence supports implementation, not biological qualification

The ds005385 reference work completed feature extraction across the selected cohort, fitted both registered anchors, and produced a descriptive HDF5 statistics inventory. This establishes that the reference artifacts and their recorded cohort summaries exist. It does not establish raw-file byte identity: the cohort matching used metadata and unique manifest basenames, and raw EDF content hashes were not computed. Artifact-level EEG quality was also not established by this metadata-only reference selection.

The current verification record does not demonstrate the complete fresh-feature plus explicit external-anchor summarize-to-HDF5 route in a single bounded run. The published anchor application evidence covers the registered config-selected cohort workflow, while the specific orchestrator path remains an open engineering check. The article therefore reports no claim of independent implementation validation, test-retest reliability, cross-dataset equivalence, or predictive performance.

= Discussion

== A measurement contract is useful only with its realization

The central result is an explicit accounting surface: a fixed coordinate order is paired with declared feature transforms, two stages of robust references, per-window support, and recording-level sensor and implementation identity. This structure permits a later analysis to select a subset, stratify by realization, or exclude unsupported windows with the relevant evidence in hand. It does not determine which measurements should answer a biological question.

The two reference presets illustrate why a label such as eyes-open or eyes-closed is not enough to make values comparable. Different condition-specific centers and scales create separate frames. The shared participant cohort also means the reference values contain information from target recordings. A study that requires a common frame or held-out normalization must predeclare that reference and apply it consistently.

== Limits and next validation

Artifact QC, raw EDF hashes, independent anchor evaluation, and biological outcomes are not established in the reference run. The output inventory is descriptive and mixes measurement families from enclosing HDF5 files, so v3.4 results must be selected by their full measurement path and realization. The external-anchor orchestration path still requires one bounded end-to-end pilot using fresh features, followed by direct inspection of window masks, sidecar identity, selected anchor hashes, and failure reasons.

Future validation should assess measurement-specific reliability, sensitivity to declared preprocessing choices, sensor/reference compatibility, and missingness under a predeclared use case. Such work should not tune estimators to create a preferred condition contrast. Until that evidence exists, v3.4 should be treated as an experimental measurement definition with transparent technical references, not as a qualified neural state or a clinical or predictive instrument.

= Materials and Methods

== Software and definition identity

The repository package metadata is 3.1.0. MNPS 3.4 is an experimental measurement definition under the package; it is not a package release. Its schema is mndm.mnps_v3_4.experimental_flat_9d.v1 with definition version 3.4. The common EEG and ds005385 configurations, registered anchor JSON files, and source implementation identify the realization. No public release tag or Zenodo record for this new measurement definition is asserted here.

== Dataset selection and preprocessing

The source dataset was OpenNeuro ds005385 version 1.0.3. Selection was restricted to session 1, acquisition pre-task, and the two resting conditions. Selection used source metadata and required sidecars, not signal values. One recording per participant and condition was present for 608 participants. The recorded 64-channel data were average referenced, not interpolated to a common montage, filtered under the versioned EEG configuration, and resampled to 250 Hz. Features were computed in 8 s windows with a 4 s step. Actual exported start and end times remain authoritative; windows do not cross declared source discontinuities.

The common preprocessing used a 50 Hz notch and 1-45 Hz EEG bandpass. The alpha-envelope candidate uses an additional order-4, zero-phase Butterworth bandpass from 8 to 12 Hz on each finite continuous source segment. A 10 s symmetric edge guard is required for that candidate. This candidate-specific restriction does not erase other computable coordinates.

== Feature estimators and coordinate assembly

Common EEG features use a pointwise median across available EEG channels as the global signal before window-level calculations @gramfort2013_mne. The configured power spectral density method is multitaper; the implementation falls back to Welch only if the multitaper routine is unavailable. Permutation entropy uses order 5, delay 1 sample, and normalization @bandtPompe2002; a degraded entropy fallback is retained in the raw feature table but is unsupported as a v3.4 coordinate. The centroid candidate uses Welch power spectra per native channel @welch1967, 4 s segments, 50 percent overlap, a periodic Hann window, and constant detrending. Channel power spectra are reduced by the median at each frequency from 7 to 13 Hz, and the power-weighted mean frequency is emitted. The envelope candidate computes the Hilbert magnitude after the separate alpha filter, calculates population standard deviation divided by mean over time for each channel, and takes the channel median. Spatial log-variance dispersion is 1.4826 times the median absolute deviation of log10 channel variance and remains an auxiliary candidate.

The nine coordinate source rules are listed in Table 1. Power inputs to log coordinates are transformed with log10; ratios and other unitless features use identity transforms. The feature anchor is applied to the transformed source matrix. Coordinates are then assembled with the fixed coefficients in the configuration. Each coordinate is subsequently centered and scaled by its separate coordinate anchor. This order is material: feature and coordinate anchors are not interchangeable.

== Robust reference fitting

For each measurement, only finite supported observations enter its reference. Participant IDs are mandatory. If participant i contributes n_i supported windows, each of their windows receives provisional weight 1/n_i; weights are then normalized over the reference. Thus each participant contributes equal total weight while the distribution of that participant's windows is retained. The weighted median is the center. The scale is 1.4826 times the weighted median absolute deviation around that center. The implementation records participant count, window count, center, scale, and status. Insufficient participants, no finite values, or a degenerate scale produce an unsupported anchor field rather than an imputed value.

The ds005385 manifests were fitted separately for eyes-closed and eyes-open references. Each condition-specific set contained 608 participants and was applied back to the same 608 participants. This overlap is recorded as transductive reference use. The presets are technical centering and scaling references, not normative population parameters. The two references are not one shared coordinate frame.

== Validity, support, and output schema

The v3.4 group stores values and support separately. Per-coordinate validity and the complete-nine-coordinate mask are window-level. The HDF5 group records source and target identity, time bounds, ordered channel labels, channel-set hash, channel count, reference and reference-set hash, QC, support reason, estimator and implementation hashes, anchor ID and hashes, compatibility state, qualification, and failure reason. If an application fails, raw measurements and QC remain; anchored values are invalidated rather than zero-filled or silently normalized. Unsupported windows remain in the output where their source rows exist.

== Cohort inventory

The read-only cohort inventory enumerated numeric HDF5 paths, finite components in serialized JSON, support/QC counts, and recording identities. Array summaries intersected exported validity masks where available. The report calculated mean, population standard deviation, median, median absolute deviation, scaled median absolute deviation, first and third quartiles, interquartile range, minimum, maximum, skewness, excess kurtosis, and support counts. EO and EC remained separate. It did not fit inferential condition contrasts. Its numerical inventory included historical measurement groups present in the source HDF5 files as separate paths and realizations.

Matching between the metadata selection and HDF5 recording records used unique manifest basenames when relative source paths were not preserved. This is not raw-signal content verification. Raw EDF hashes and artifact QC were not produced by the cohort selection.

== Reproduction commands

From an environment with this repository installed and the ds005385 data available at the configured path, feature extraction and summarization use the same version-specific configuration:

#raw("python -m mndm.cli features --config mndm/config/v.3.4/config_ingest_ds005385_baseline_eo_ec.yml --dataset ds005385\npython -m mndm.cli summarize --config mndm/config/v.3.4/config_ingest_ds005385_baseline_eo_ec.yml --dataset ds005385", block: true, lang: "powershell")

These commands describe a reproducibility path; the present manuscript revision did not rerun the cohort. A new validation of explicit external-anchor orchestration should use a small independent target selection and a separate output root.

= Acknowledgements

Generative AI tools assisted with manuscript structure and language editing under human direction. The author remains responsible for the claims and final text.

= Supporting Information

Supplementary Methods S1 details estimator parameters, robust reference fitting, support and HDF5 outputs. Supplementary Changelog S6 documents the historical contract and how it differs from this paper's experimental 3.4 definition.

= References

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
