#import "template_eLife.typ": *
#set page(numbering: "1")

#show: netn-template.with(
  title: "S4 Clinical Interpretation and Limits",
  short-title: "S4 Clinical Interpretation",
  subtitle: "A clinician-facing explanation of scope, measurement meaning, and evidence limits",
  authors: (
    (name: "Robin Langell", affil: 1),
  ),
  affiliations: (
    "Langell Konsult AB, Vallentuna, Sweden",
  ),
  corresponding-author: "Robin Langell, hello@noeticdiffusion.com",
  article-type: "Supplement",
)

= S4.1 Scope and intended audience

#netn-box(title: [This is a measurement-method article, not a clinical validation study])[
  The article describes how several neural-signal measurements are defined, recorded in files, and qualified for support. It does not test a diagnosis, treatment, clinical outcome, or patient-level prediction. Eyes-open and eyes-closed human recordings and the separate animal LFP state strata are input contexts for technical analyses; they are not presented as intervention effects. No causal, diagnostic, prognostic, or treatment-response claim is supported.
]

This supplement translates the measurement claims for clinicians and clinically oriented readers without extending them. The human EEG sections report a reference-cohort description and a repeated-session measurement assessment. The MEG and LFP sections are separate, bounded engineering examples. The article does not define a clinical threshold or a validated biomarker, and it does not compare a patient group with a control group.

= S4.2 What the measurement system does

NeuralManifoldDynamics (NMDM) is an ingest and measurement system. It turns recorded signals into named, versioned numerical summaries and retains information about the channels, reference, preprocessing, window timing, missing values, and support behind each result. The purpose is traceability: a downstream analyst can determine how a value was produced and whether it is suitable for a particular comparison.

The article uses the term *coordinate* for one ordered component of a fixed measurement vector. The nine EEG coordinates are not established anatomical regions, independent brain systems, or validated dimensions of an underlying neural manifold. A low-dimensional label in the framework does not by itself demonstrate that neural activity follows a biological manifold.

= S4.3 Data and conditions in plain language

The primary EEG reference dataset, OpenNeuro ds005385 version 1.0.3 @openneuro_ds005385_v103, contributed session-1 pre-task eyes-closed and eyes-open recordings from 608 participants. It yielded 1,216 recordings and 55,183 overlapping 8-second windows. A separate OpenNeuro dataset, ds004148 version 1.0.1 @openneuro_ds004148_v101, contributed resting eyes-closed and eyes-open recordings across three sessions for 60 selected participants. Each repeatability recording used the same frozen reference fitted from ds005385.

The other modalities were not combined with those EEG measurements. A single MEG FacePerception run from ds003645 was used to check a modality-specific export, keeping magnetometers and gradiometers separate. An LFP feature and support pilot used one DANDI 000458 rodent recording with awake, isoflurane, and recovery strata. DANDI 000232 / EF0147 was assessed from metadata only, without reading signal values or calculating features.

= S4.4 What the nine EEG coordinates summarize

These descriptions are measurement definitions, not clinical interpretations. The EEG pipeline scales coordinate values relative to a reference; the values should not be read as raw physical quantities. In the ds005385 reference analysis, eyes-open and eyes-closed anchors were fitted separately. In the ds004148 repeatability analysis, one fixed eyes-closed anchor was applied to both conditions.

#figure(
  clean-table(
    columns: (1.45fr, 2fr, 2.15fr),
    header: ([*Coordinate*], [*Plain-language description*], [*Key limit*]),
    inset: 3pt,
    [#raw("delta_theta_level")], [A combined log-power summary of slow delta and theta activity.], [The two anchored band summaries are averaged; it is not a direct measure of sleepiness or attention.],
    [#raw("alpha_level")], [Log power in the alpha band.], [Power depends on the recording and declared processing; it is not a stand-alone clinical marker.],
    [#raw("beta_alpha_ratio")], [Beta-band power divided by alpha-band power.], [A ratio can change through either its numerator or denominator.],
    [#raw("gamma_level")], [Log power from 30–45 Hz.], [Muscle activity can contribute in this range; clean neural gamma was not established.],
    [#raw("hjorth_mobility")], [A ratio describing sample-to-sample signal variation relative to signal variation.], [The implemented value is not divided by the time step, so it is not a physical frequency in s⁻¹.],
    [#raw("alpha_theta_ratio")], [Alpha-band power divided by theta-band power.], [A small or unsupported denominator invalidates the value; the pipeline does not substitute a floor.],
    [#raw("permutation_entropy")], [A summary of the variety of local sample-order patterns.], [It is an estimator-defined complexity statistic, not a measure of consciousness or cognition.],
    [#raw("alpha_centroid_hz")], [A power-weighted center frequency within 7–13 Hz.], [It is not an alpha-peak detector or a validated individual alpha frequency.],
    [#raw("alpha_envelope_cv")], [Relative variability of the 8–12 Hz signal envelope.], [It is not a burst detector and is unsupported near segment edges under its 10 s guard.],
  ),
  kind: table,
  caption: [Plain-language gloss of the nine ordered EEG components and their immediate interpretation limits.],
)

= S4.5 Main findings and their clinical meaning

*The EEG measurements were computable in the selected reference exports.* The eight coordinates other than alpha-envelope CV had valid outputs in all exported windows. Alpha-envelope CV was supported in 88.1% of EC and 88.0% of EO windows because the estimator requires additional context around a window. Every participant still had at least one window with all nine coordinates. These are support and processing findings; artifact QC was unassessed in the ds005385 audit.

*Repeatability was variable.* In ds004148, descriptive ICC(A,1) estimates for the 54 EC/EO coordinate-level session comparisons ranged from 0.041 to 0.954. The 27 EO−EC contrast comparisons ranged from 0.249 to 0.864. The range depends on coordinate, condition, and which sessions are compared. An ICC is a property of this measurement procedure in the selected data and does not show that a value is clinically stable or meaningful for an individual.

*The modality pilots remained separate and limited.* The one-run MEG check confirmed that declared variants could be exported with sensor families separate. The LFP contact features were computed, but the real-data local-dynamics support requirements were not met and no paired held-out predictions were produced. Neither pilot demonstrates a shared EEG–MEG–LFP scale or a clinical use.

= S4.6 What the article does not establish

The article does not estimate an inferential eyes-open versus eyes-closed group effect. Its separate ds005385 condition anchors define different coordinate frames, so values from those two fits should not be pooled as though they had a common origin and scale. The repeatability study uses a shared frozen anchor but reports session consistency, not a clinical condition effect.

The analyses also do not establish artifact-free signals, a validated latent neural dimension, a biological mechanism, cross-dataset measurement equivalence, equivalence between MEG sensor families, a clinical diagnosis, an individual prognosis, a treatment response, or predictive performance. No null-hypothesis result should be inferred from a technical support failure or from an analysis that was not run.

= S4.7 How to read common terms

*Computable* means that the estimator had its declared inputs and returned a value. It does not mean the value is artifact-free or suitable for a clinical decision.

*Comparable* means that the two values use compatible sensor sets, references, preprocessing, estimator definitions, units, time grids, and support. Matching coordinate names alone is not enough.

*Qualified* means that evidence and acceptance rules for a stated scientific or clinical use have been met. This article reports no such clinical qualification. A numeric value, successful file export, or finite local estimate is not equivalent to qualification.

= S4.8 What clinicians should not conclude

Do not use these measurements to diagnose, screen, stratify, or monitor an individual. Do not infer disease, arousal, cognition, or treatment effect from a coordinate name. Do not interpret the separate EEG condition references as a common clinical scale, or compare EEG, MEG, and LFP values as equivalent. Do not treat the LFP support failure as evidence that awake, anesthetized, or recovery activity lacks dynamics.

= S4.9 Evidence needed before a clinical use

Clinical interpretation would require a prespecified use and decision, a defined intended population, clinically characterized independent cohorts, measurement-specific analytical and test–retest performance, artifact and missingness assessment, and evidence that the measurement adds useful information to existing clinical practice. Diagnostic or prognostic use would additionally need blinded external validation, calibration at the intended prevalence and setting, and evaluation of harms from false results. Monitoring or treatment-response claims would require prospective longitudinal data and a design suited to the proposed causal question. Those studies are outside the scope of the current article.

= S4.10 Summary for clinical readers

This article documents an experimental way to produce and audit neural-signal summaries. It reports strong data-provenance and support detail, alongside uneven repeatability across the nine EEG coordinates and session comparisons. Its MEG and LFP examples are separate technical pilots, with important support and comparability limits. The work may help research teams specify what a neural measurement contains, but the present evidence does not support clinical interpretation, individual prediction, diagnosis, or treatment decisions.

#bibliography("NeuralManifoldReferences.bib", style: "apa")

