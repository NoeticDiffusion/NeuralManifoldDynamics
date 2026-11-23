// S4 Supplementary Clinical Summary — generic template.
// Rename and replace placeholders per article. Keep the cautious tone.

#import "../network_neuroscience_template.typ": *

#show: netn_template.with(
  title: "S4 Supplementary Clinical Summary",
  short_title: "S4 Clinical Summary - NeuralManifoldDynamics",
  subtitle: "A cautious clinician-facing interpretation of the article",
  authors: (
    (name: "Robin Langell", affil: 1),
  ),
  affiliations: (
    "Langell Konsult AB, Vallentuna, Sweden",
  ),
  corresponding_author: "Robin Langell, hello@noeticdiffusion.com",
  article_type: "Supplement",
  keywords: ("clinical caution", "measurement contract", "observational", "QC"),
)

= Scope and intent

_Historical archive: this clinician-facing summary belongs to an earlier
NeuralManifoldDynamics article version and is not part of the current
multimodal methods article._

#netn_box(title: [Human clinical samples are observational; animal transport audits are separate])[
  The human clinical samples summarized here were observational; no clinical
  intervention was tested in those cohorts. Separate ancillary rodent transport
  audits include awake, isoflurane, recovery, and stimulation-related states,
  so the no-manipulation statement does not apply to those recordings. Findings
  do not support individual-level predictions or causal clinical conclusions.
]

This supplement translates the measurement contract for clinically oriented
readers without amplifying it into a diagnostic claim. The main contribution is
software and measurement infrastructure: it makes modality-specific feature
surfaces, low-dimensional proxy trajectories, quality flags, and provenance
available for later research. Reference and validation cohorts are not
interchangeable, and some are healthy laboratory samples rather than patient
cohorts.

= What is the framework, and why does it matter here?

The framework converts a window of neural or physiological data into a
reproducible vector of features, then applies a fixed versioned mapping to a
three-dimensional proxy chart and an optional nine-dimensional refinement.
This can help clinical researchers compare processing choices, identify
unsupported windows, and preserve the exact provenance of a derived result.

The words *mobility*, *deviation*, and *entropy* describe operational proxy
families in this release. They do not prove that a patient occupies a unique
biological state or that the chart is a validated clinical biomarker.

= What was studied

The repository contains reference EEG and fMRI configurations, plus validation
examples using EEG with ECG/PPG/pupil recordings, simultaneous MEG/EEG, sleep
data, verbal working memory, Parkinson-related recordings, DANDI/NWB
neurophysiology, and Figshare BDF infant EEG. These streams primarily test
whether data can be indexed, transformed, quality-controlled, and serialized
under one auditable contract.

The ds003838 export completed for 130 subjects; the corrected stage-contrast
statistics used 62 subjects with complete stage coverage. The ds003645 MEG
result is a five-subject pilot with readiness 0.7879. DANDI LFP and infant EEG
results are transport/QC demonstrations rather than clinical endpoint studies.

= What the coordinates mean in plain language

- `m` is a release-fixed proxy family emphasizing slow-wave, alpha, and
  macrostate-related feature changes.
- `d` is a proxy family emphasizing dispersion, high-frequency or binding
  features, and local dynamical variation.
- `e` is a proxy family emphasizing nonlinear complexity and an additive
  arousal/embodied fallback surface.
- `mnps_3d_dot` describes change along the proxy chart; it is not a clinical
  rate of disease progression.
- A Jacobian describes local deformation of the proxy trajectory. It is useful
  for auditing local geometry but is not a direct physiological mechanism.
- `anchor_state` contains time-aligned body-state proxies such as HRV-derived
  indices. It is parallel to the neural chart, not a fourth neural coordinate.

= The main findings, in order of clinical salience

== 1. The contract improves auditability

The strongest result is infrastructural. The pipeline records coverage,
invalidity, feature transforms, source rows, anchors, and optional Jacobians
in self-describing outputs. This can make future clinical analyses easier to
reproduce and easier to reject when support is inadequate.

== 2. The ds003838 physiology surface changes across task stages

In the corrected ds003838 validation package, `vagal_index`, sympathetic and
HRV measures, trajectory path length, and related coupling measures varied
across rest, listening, and memory-load stages. The listen--mem13
`vagal_index` contrast had Cohen's `d = 1.995` and Wilcoxon
`p = 1.2 × 10⁻¹⁰` on the complete-case sample `N = 62`. This is an internal
task-contrast validation of the export surface, not evidence of a clinical
autonomic treatment effect.

== 3. MEG is usable as a pilot mapping, not as a clinical modality claim

The ds003645 pilot showed task-sensitive face/scrambled separation across
several window sizes. Its readiness score was 0.7879, below the 0.80 expansion
threshold, and subjects showed a bimodal d-family pattern. A later holdout did
not reproduce matched EEG--MEG residual topography. The appropriate clinical
reading is that the engineering path is promising enough for further study,
not that MEG and EEG have been shown to share one validated manifold.

== 4. Neuropixels/LFP ingest is a transport demonstration on animal data, not a human clinical finding

The DANDI 000006 (spike-sorted Units) and 000458 (LFP) examples are rodent
recordings used to show that spike-rate and continuous-electrode data can be
ingested, quality-controlled, and mapped into the same measurement contract as
the human EEG/MEG/fMRI pipelines. No readiness score, no cross-species
comparison, and no clinical or circuit-level claim accompanies these
examples. They should be read as an ingest-format and QC extension only.

= What was NOT found - equally important

The repository does not establish diagnostic accuracy, individual prediction,
causal treatment response, or superiority over HMM, microstate, CAP, LEiDA,
CEBRA, or other latent-state methods. The ds003838 effect has no independent
cohort replication in this evidence pack. Working-memory HRV claims in
ds006848 were withheld because 60 s windows were contaminated by retrieval
events. Pupillometry is implemented but not validated at dataset scale.
DANDI LFP contact/reference comparisons are descriptive one-session checks,
and infant BDF Status labels do not support rest/reaching condition contrasts.

= How to read the technical terminology clinically

“Proxy” means a fixed computational summary, not a validated biomarker.
“Anchor” means a time-aligned physiological surface, not proof that the body
signal causes a neural coordinate. “Geometry validity” means that numerical
support and conditioning passed the stated checks, not that the underlying
biology is correct. “Internal validation” means the repository contains a
reproducible run or audit; it does not mean independent replication.

= What clinicians should not conclude

Readers should not infer diagnosis, prognosis, consciousness level, treatment
response, patient-level risk, or causal physiology from these outputs. They
should not use `m`, `d`, `e`, `vagal_index`, or a readiness score as a clinical
decision threshold. They should not interpret a high-quality HDF5 file as
evidence that the chosen proxy chart is biologically unique.

= What would increase clinical confidence

Confidence would require preregistered replication in independent patient
samples, longitudinal within-person designs, validated clinical outcomes,
explicit out-of-sample evaluation, modality-specific artifact cleaning,
calibration of individual-level uncertainty, and intervention studies. Chart
stability under feature substitution and projection perturbation should also
be tested before clinical interpretation.

= Summary for clinical readers

NeuralManifoldDynamics is best treated as an auditable research measurement
contract. It can organize multimodal data into versioned proxy trajectories,
physiological sidecars, geometry diagnostics, and provenance-rich artifacts.
Some task-linked internal validation results are encouraging, especially the
corrected ds003838 stage package, while MEG, LFP, infant EEG, and phase-anchor
lines remain exploratory or transport-focused. Nothing in the current evidence
supports diagnostic use, individual prediction, or causal clinical claims.

#bibliography("../NeuralManifoldReferences.bib")
