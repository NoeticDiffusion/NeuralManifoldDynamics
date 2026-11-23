#import "template_eLife.typ": *
#set page(numbering: "1")

#show: netn-template.with(
  title: "Supplementary Changelog: NeuralManifoldDynamics",
  short-title: "Supplementary Changelog S6",
  subtitle: "Historical measurement contracts and the MNPS 3.4 transition",
  authors: (
    (name: "Robin Langell", affil: 1),
  ),
  affiliations: (
    "Langell Konsult AB, Vallentuna, Sweden",
  ),
  corresponding-author: "Robin Langell, hello@noeticdiffusion.com",
  keywords: ("supplementary changelog", "software versioning", "measurement contract", "provenance"),
  article-type: "Supplement"
)

= S6.1 Purpose and version labels

This changelog preserves historical version-specific information that is not part of the new flat nine-coordinate article. The earlier version referenced by this project has already been published on Zenodo, as reported by the project owner. That archived version remains a separate historical record. This local manuscript revision does not overwrite it, assign it a new DOI, or claim a new package release.

Version labels refer to different objects. The Python package metadata in the current checkout is 3.1.0. The v3.4 label identifies an opt-in experimental EEG measurement definition within that package. It is not a package version. The prior article described a broader series of measurement and analysis capabilities accumulated across earlier package generations; these historical details are summarized below for provenance rather than carried into the current article.


= S6.2 Historical coordinate contract

The earlier article described a canonical three-component output named mnps_3d with fixed order [m, d, e]. These labels were operational proxy families associated in the broader NDT literature with metastability, deviation from an integration-segregation balance, and entropy or entropic energy. The exported EEG values were not direct measurements of those theoretical constructs. They were feature-based proxies fixed by the configuration and software realization.

In the earlier EEG configuration, the three components were recomposed from nine named source coordinates. The table records the old feature map and fixed coefficients as historical definitions; it is not a mapping into the v3.4 output.

#figure(
  clean-table(
  columns: (auto, 1fr, 1fr),
  header: ([*Historical coordinate*], [*Historical EEG input*], [*Historical grouping*]),
  inset: 4pt,
  [m_a], [negative half log10 delta power plus negative half log10 theta power], [m = 0.62 m_a + 0.55 m_e + 0.45 m_o],
  [m_e], [negative log10 alpha power], [m family],
  [m_o], [beta-to-alpha ratio], [m family],
  [d_n], [gamma power], [d = 0.50 d_n + 0.82 d_l + 0.28 d_s],
  [d_l], [Hjorth mobility], [d family],
  [d_s], [alpha-to-theta ratio], [d family],
  [e_e], [permutation entropy], [e = 0.85 e_e + 0.62 e_s + 0.03 e_m],
  [e_s], [Hjorth complexity], [e family],
  [e_m], [operational arousal fallback family], [e family],

  ),
  kind: table,
  caption: [Historical EEG coordinate sources and fixed projection weights from the prior contract.],
)

The historical EEG e_m source could be selected from a priority chain of cardiac, vascular, pupil, eye-movement, MEG, and EEG features, with the chosen source recorded in provenance. Thus the e_m label did not always denote the same sensor or physiological quantity. The 9D historical chart and its fixed weighted projection supported recomposition into the three-component output, with runtime normalization and support handling. The exact old projection was a versioned operational rule, not an optimized biological weighting.

The older article also described derivatives, Jacobian outputs, regional measurements, and body-brain auxiliary surfaces. Those objects belonged to the historical software generations and their own estimator contracts. They are not outputs of the experimental flat v3.4 definition.

= S6.3 Summary of software and measurement changes

== MNPS 1.2 and early measurement export

The early implementation provided a compact feature-to-coordinate export with limited provenance and summary layers. It established the motivation for a low-dimensional time series, but did not provide the later separation between source features, reference scaling, sensor realization, and scientific qualification.

== MNPS 2.1 and anchored coordinate layers

The 2.1 generation made feature-level and coordinate-level reference scaling more explicit. Subject-relative and cohort or external-reference coordinates became distinguishable, with anchor provenance retained. The historical chart and estimator semantics remained version-specific.

== MNPS 2.3 and additive physiological surfaces

The 2.3 generation added optional time-aligned physiological and embodied measurements, including cardiac and vascular features, and additional event-locked and block-native derived layers. These were additive output surfaces; they did not make the underlying coordinate family a direct measure of embodiment.

== MNPS 2.4 and 2.5

The 2.4 and 2.5 lines added broader support and provenance fields, BDF and infant-EEG input paths, NWB and Neuropixels feature transport, optional phase-anchor and sleep-related estimators, quality controls, and additional dataset overlays. The associated pilots varied in scope and qualification. Transport, QC, and descriptive output evidence should not be read as cross-modal equivalence or biological validation.

== MNPS 2.6 and 3.0

The 2.6 generation added versioned local-dynamics interpretations on existing Jacobian estimators and stricter claim-status fields. The 3.0 generation formalized an output contract for the historical three- and nine-component charts, local Jacobian outputs, versioned dynamical families, and support/provenance status. Some advanced families remained gated or explicitly unqualified. Their configuration and validation evidence are preserved in the prior article and release history; they are not results of the current v3.4 reference work.

== Package 3.1.0

The current checkout reports package version 3.1.0 and includes additional experimental measurement modules. Package versioning, measurement-definition versioning, YAML schema versions, and anchor IDs must not be conflated.

= S6.4 What changed in the v3.4 article

The new article narrows its subject from a cumulative history of NeuralManifoldDynamics to one implemented experimental measurement definition. It describes nine ordered EEG measurements, per-window support, candidate-specific estimators, participant-balanced robust references, explicit sensor/reference identity, and an HDF5 output contract. It does not carry forward historical model definitions, modality catalogues, regional-network interpretations, Jacobian interpretations, or exploratory cohort findings that are not needed to define this measurement.

The v3.4 output is flat. Its ordered measurements are delta_theta_level, alpha_level, beta_alpha_ratio, gamma_level, hjorth_mobility, alpha_theta_ratio, permutation_entropy, alpha_centroid_hz, and alpha_envelope_cv. It does not aggregate these values into historical families, project them to a three-component output, or emit a v3.4 Jacobian. The order, estimator, and output namespace are versioned. Historical labels and feature relationships do not imply equal units, scale, estimator, or scientific meaning.

Several source features are related by name or lineage to inputs used in earlier configurations. These relationships are recorded here only as migration context. Feature anchors, transforms, and a second coordinate-scaling stage mean that same-named source features do not establish equivalence between anchored outputs. A downstream comparison requires the exact definition, feature order, transforms, channel set, reference, time grid, and anchor realization.

The v3.4 article also elevates three distinctions that were less prominent in the older presentation:

- Computability is an estimator and support question.
- Comparability depends on measurement realization and a declared comparison.
- Qualification is determined by a downstream scientific protocol.

A value can be computed while comparability or qualification is unknown. Unsupported values are not replaced with zero, and one unsupported estimator does not suppress unrelated computable measurements.

= S6.5 Reference-cohort evidence and limits

The v3.4 technical references use session-1 pre-task eyes-open and eyes-closed data from OpenNeuro ds005385. Each condition includes 608 participants, and the same participants both define and receive the condition-specific anchor. The two anchors define separate coordinate frames. They are technical reference artifacts, not validated population norms.

Feature extraction produced 55,183 window rows from 1,216 recordings. Anchor fitting and application completed for all 608 recordings in each condition, with no anchor identity failures reported. The cohort inventory includes descriptive statistics and support/QC summaries by measurement path and realization. Source matching used unique manifest basenames when relative paths were unavailable. Raw EDF hashes and header calibration checks were not computed; the derived-export directory contained no source EDFs. The metadata-only selection did not establish artifact QC.

The frozen-reference cohort run is complemented by a full-cohort ds004148 session-repeatability analysis. A metadata-only selection fixed 60 participants, three sessions, and resting EC/EO pairs before coordinate inspection. One frozen ds005385 EC anchor was applied unchanged to all 360 recordings. Session-1 HDF5/QC outputs were reused and sessions 2–3 were re-extracted; the QC rule and support floor were fixed before coordinate analysis. QC matched 26,640 windows. The support floor passed for 2,858/3,240 coordinate summaries and 306/360 complete-nine participant-session-condition summaries. Across eligible comparisons, descriptive ICC(A,1) ranged 0.041–0.954 for coordinate levels and 0.249–0.864 for EO−EC differences. This establishes technical fixed-anchor application and dataset-specific repeatability estimates, not artifact absence, biological stability, cross-dataset equivalence, or scientific qualification.

= S6.6 Archival and compatibility notes

The earlier Zenodo-published version is a historical release. The exact record DOI, source commit, and manuscript-to-record mapping could not be verified from the repository files inspected for this revision; no DOI or commit is inferred here. Publication metadata should link the archived record and manuscript once those identifiers are confirmed. This article revision is a separate document describing the experimental v3.4 definition.

The v3.4 code writes to its own measurement namespace and may coexist with other outputs in a combined HDF5. The new article does not promise automatic migration, numerical equivalence, or backward compatibility between historical coordinate values and v3.4 coordinates. Read-only checks of selected ds005385 and ds004148 combined exports found root-level historical paths; these checks confirm the examined files, not the layout of every archived historical release. In the ds004148 files, the root manifest identifies `/jacobian` as the primary three-component field and `/jacobian_9D` as the legacy `subcoords_9d_v2_0` field. The separately named subject-anchored layers carry schema `mndm.jacobian_layer.v2.1` and state their coordinate space in attributes. The primary groups themselves lack schema attributes, so path identity must be interpreted with the root manifest rather than matrix shape alone. A bounded summarize smoke verified that disabling both legacy Jacobian switches and the separate v3.4 Jacobian candidate omits all these paths while preserving all nine v3.4 coordinates. The prior full-cohort outputs were retained unchanged; their historical Jacobians were not analyzed.

#figure(
  clean-table(
    columns: (1.3fr, 1.2fr, 2fr),
    header: ([*Historical output or concept*], [*v3.4 equivalent*], [*Compatibility statement*]),
    inset: 4pt,
    [`/mnps_3d` `[m,d,e]`], [None], [Root-level path observed in selected combined exports; flat v3.4 does not aggregate into a three-axis chart.],
    [`/coords_9d`], [None], [Root-level group observed in selected combined exports; historical order, transforms, clipping, and reference semantics are not inherited by v3.4.],
    [Jacobian/MNJ outputs], [None], [v3.4 defines no local derivative or state-transition operator.],
    [Regional MNPS outputs], [None], [v3.4 does not emit regional or anatomical-network coordinates.],
    [Dynamical-family outputs], [None], [Prediction, diffusion, reachability, and resilience families remain separate versioned products.],
    [Embodied or modality-specific projections], [None], [No cardiac, vascular, pupil, MEG, fMRI, or cross-modal projection is part of v3.4.],
  ),
  kind: table,
  caption: [Historical output paths and concepts without a direct equivalent in the MNPS 3.4 flat EEG contract.],
)

Before public release, freeze the code and configuration commit; identify the exact previous Zenodo version; decide whether v3.4 remains experimental or becomes a release; check all manuscript claims against the frozen implementation; and publish the software and manuscript records with explicit version relationships.
