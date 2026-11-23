// S5 Glossary for the current NeuralManifoldDynamics measurement contract.

#import "network_neuroscience_template.typ": *

#show: netn_template.with(
  title: "S5 - Glossary: MNPS 3.4 and Historical Terms",
  short_title: "Glossary - NeuralManifoldDynamics",
  subtitle: "Current experimental terms followed by version-scoped historical terms",
  authors: (
    (name: "Robin Langell", affil: 1),
  ),
  affiliations: (
    "Langell Konsult AB, Vallentuna, Sweden",
  ),
  corresponding_author: "Robin Langell, hello@noeticdiffusion.com",
  keywords: ("glossary", "NDT", "MNDM", "MNPS", "embodied anchoring"),
  article_type: "Supplementary Material",
  abstract: [
    This is the supplementary glossary for the MNPS 3.4 methods article. Current experimental terms appear first; historical and other-version terms are retained afterward only to distinguish related repository outputs.
  ],
)

#set heading(numbering: none)

_This is the companion glossary for the present methods article. “Implemented”
describes a named computation or export; “interpretation” gives its narrow
reading; “not established” marks claims that are not proved by this measurement
definition._

= Experimental MNPS 3.4 terms

The following entries are scoped to the experimental EEG definition
`mndm.mnps_v3_4.experimental_flat_9d.v1` unless the entry says otherwise.

*NDT (Noetic Diffusion Theory)* - *Theory programme.* A framework for
hypotheses about neural state and condition-dependent future dynamics. NDT is
not identical to an MNPS chart or a particular NMD measurement definition.

*NMD (NeuralManifoldDynamics)* - *Measurement layer.* A versioned system for
specified observations, time support, transforms, validity, and provenance.
MNPS 3.4 is one experimental EEG realization within that layer.

*Operational coordinate* - *Output component.* An ordered measurement value
under the named v3.4 contract. The word does not identify a latent biological
axis or establish manifold geometry.

*MNJ (Meta-Noetic Jacobian)* - *Downstream estimator name.* A Jacobian
estimated for a local flow in a declared MNPS chart. MNPS 3.4 emits no MNJ;
the term does not mean temporal acceleration or a direct biophysical
Jacobian.

*Computability, comparability, and qualification* - *Separate decisions.*
Computability asks whether declared inputs and support allow an estimate;
comparability asks whether two realizations suit a stated comparison;
qualification asks whether evidence supports a specified use.

*MNPS 3.4 flat 9D* - *Experimental measurement definition.* An ordered EEG
vector of nine separately anchored coordinates under `/measurements/v3_4/`.
It has no M/D/E family aggregation, 3D projection, or v3.4 Jacobian. It does not
replace the canonical `[m,d,e]` or legacy `coords_9d` contract.

*Feature anchor (MNPS 3.4)* - A frozen, participant-balanced center and
`1.4826 * MAD` scale applied to transformed source features before coordinate
assembly. It is separate from the coordinate anchor.

*Coordinate anchor (MNPS 3.4)* - A frozen, participant-balanced center and
`1.4826 * MAD` scale applied after the fixed 9D coordinate assembly. Separate
EO and EC anchors define different coordinate frames.

*Complete 9D mask* - The per-window mask indicating that all nine experimental
v3.4 anchored coordinates are valid. It is distinct from individual
coordinate validity and does not imply scientific qualification.

*Exploratory transfer (MNPS 3.4)* - Explicit application of a frozen anchor
across a declared measurement-surface mismatch with a recorded reason and
mismatch provenance. It does not perform unit conversion or modality
harmonization.

= Historical and other-version terminology

The terms in this section describe legacy MNPS v2/v3.0 contracts or optional
measurement families from other software versions. Their use in this glossary
does not make them part of the experimental MNPS 3.4 EEG definition. Versioned
exceptions are named in the individual entry.

= A - D

*Attractor* - *Not established in this ingest contract.* In dynamical-systems
language, a stable region to which trajectories return after perturbation.
NeuralManifoldDynamics exports trajectories and local summaries; it does not
identify attractors.

*Anchor state* - *Implemented.* A time-aligned physiological feature surface,
including HRV-, PPG-, or pupil-derived indices. It is additive and does not
replace `mnps_3d`.

*A_bD* - *Operational alignment diagnostic.* The scalar
`b^T a_hat b / ((b^T b) tr(a_hat) + epsilon)` used when an independently
qualified chart velocity `b` is available. It is `not_testable` without that
velocity and is not an Itô drift-to-diffusion identification.

*Anisotropy* - *Implemented.* A summary of directional imbalance in a Jacobian
or block-Jacobian field. Its reliability depends on local support and
conditioning.

*CHART* - *Bounded coordinate audit.* A synthetic audit of coordinate and
projection behavior. Orthogonal and isotropic cases pass the audited identity;
anisotropic cases are expected to change local estimates, projection loses
hidden-coordinate information, and the scalar unregularized diagnostic is not
a production C1 gate.

*Claim status* - *Implemented provenance field.* A machine-readable ceiling on
interpretation, including `no_biological_claim` for computed dynamical-family
objects.

*Coordinate (historical v2/v3.0 usage)* - *Implemented in those contracts.* A
release-fixed numerical axis or subcoordinate. The labels `m`, `d`, and `e`
are operational proxy families, not uniquely identified biological quantities;
they are not the coordinates emitted by the flat MNPS 3.4 definition.

*Coverage* - *Implemented.* The fraction of configured feature or axis support
available after masking and quality control. Minimum coverage is part of
geometry validity.

*Derived layer* - *Implemented.* An event-locked or block-native sidecar that
joins to canonical windows through provenance but does not redefine the HDF5
measurement contract.

= E - L

*Embodied Anchoring Principle (EAP)* - *Implemented as an additive design
principle.* Body-state measurements share the MNPS time grid while remaining
separate from the canonical neural chart.

*Event-locked* - *Implemented derived layer.* A window table aligned to an event
onset or peak, with detector and control-matching provenance.

*Feature anchor (legacy/general usage)* - *Implemented in the named versions.*
Frozen per-feature center/scale metadata used to construct cohort-anchored
coordinates. The MNPS 3.4 feature anchor has its own definition below and is
distinct from `anchor_state`.

*Feature standardization (legacy projection usage)* - *Implemented in the
named versions.* The configured transform applied before projection, commonly
robust-z plus clipping; selected power features may use log10 first. MNPS 3.4
does not clip its anchored coordinates. The transform is recorded for audit.

*Frobenius norm* - *Implemented.* The square root of the summed squared
Jacobian entries; a magnitude summary, not a causal strength.

*Geometry contract* - *Implemented.* The set of support, finite-value,
conditioning, duplicate-coordinate, and time-grid rules used to decide whether
a trajectory or Jacobian is valid.

*Geometry policy `standard_invalidity_v1`* - *Implemented in v2.4.0.* The
versioned policy that records canonical invalidity and `coords_9d` duplicate
tolerance diagnostics.

*HRV v0.1* - *Implemented in v2.4.0.* A centered ECG superwindow surface with
HR mean, IBI mean, SDNN, RMSSD, pNN50, NN count, artifact fraction, coverage
fraction, and quality score. HF and LF/HF are not included.

*Jacobian / MNJ* - *Implemented optional summary.* A local estimate of
`partial f / partial x` in chart coordinates. It is not a direct estimate of a
unique biophysical state equation.

*Analytic conditional-lag witness* - *Model-derived diagnostic.* For a linear
Gaussian system, the Frobenius norm of `C_2 - C V^-1 C` measures one-step lag
memory after conditioning on the current observation, where `C_2 = H A^2 P
H^T`. The conditional next-state covariance is a separate quantity,
`V - C V^-1 C^T`. Neither is an estimator variance, Markov-sufficiency gate,
or biological qualification.

*Cross-segment order corruption* - *Synthetic negative control.* A fixed roll
of held-out predictions changes the source-to-truth mapping while preserving
the truth array. It is a mapping control, not a temporal-shift estimator or a
biological null.

*Discrete conditional velocity* - *Analysis-owned target.* A one-step
conditional expectation divided by the declared sampling interval. It is a
discrete chart target and does not assert an Itô drift.

*Dynamical family* - *Implemented gated surface.* A versioned measurement
family beside the canonical chart, such as diffusion, destination, or
resilience. It carries support, validity, grain, provenance, and claim metadata
and does not add canonical MNPS axes.

= M - R

*MEG shadow mapping* - *Implemented exploratory path.* `meg_*` features route
through the existing 9D projection slots used by the EEG-family contract.
Cross-modal biological equivalence is not established.

*Neuropixels Units-to-rate* - *Implemented in v2.5.0, transport/descriptive
status.* NWB spike times are histogram-binned (default 0.05 s bins, optional
0.05 s Gaussian smoothing) into a population firing-rate matrix, then windowed
into `ephys_*` features that fill `coords_9d` through a dedicated
ephys-specific subcoordinate map. Validated as a one-session smoke run on a
rodent dandiset; not a validated manifold claim.

*Neuropixels LFP-as-EEG routing* - *Implemented in v2.5.0, transport/descriptive
status.* DANDI `ElectricalSeries` LFP channels are routed through the existing
EEG feature and 9D contract with rodent-adapted band edges and optional
depth-balanced contact selection. Validated through smoke and full-session QC
on a rodent dandiset; no readiness score is computed.

*Measurement contract* - *Implemented.* The versioned definition of what is
computed, named, quality-controlled, and serialized for downstream reuse.

*MNPS / `mnps_3d` (historical v2/v3.0 usage)* - *Implemented operational
chart.* The canonical three-axis trajectory `[m, d, e]` exported by those
versioned contracts. It is not the output of the MNPS 3.4 flat 9D definition.

*Regional MNPS (other-version output)* - *Implemented optional output.*
Separate trajectories and Jacobians for fMRI networks or EEG channel groups,
subject to modality-specific support limits. It is not an MNPS 3.4 coordinate
or a default anatomical network.

*Measurement validity* - *Implemented certificate field.* A declared validity
state for a computed object, such as `not_assessed`, `insufficient_support`, or
`translation_qualified` where the named protocol permits it. It is distinct
from `validation_level` and from `claim_status`.

*Grain* - *Implemented schema metadata.* The native row and biological unit of
an output, including recording, subject, and window context. Windows are not
biological replicates and direct between-subject inference is forbidden on
these family exports.

*Support signature* - *Implemented capability metadata.* A machine-readable
summary of coordinate support, modality capability, and fallback provenance;
it does not infer validity or scientific licensing from the presence of a
dataset.

*W_Q* - *Implemented opt-in discrete reachability object.* A chart-space
one-step predictive-spread covariance under the Gate E recording-level
transition-residual contract, written under `/stochastic_reachability/v1`.
It is not `a_hat`, Itô process noise, or a default dynamical-family output.

*Readiness score* - *Implemented pilot gate.* A weighted engineering/validation
summary, not a clinical or biological probability. The ds003645 pilot score
was 0.7879.

*Row provenance (`row_source`)* - *Implemented in v2.4.0.* Per-window source
metadata for simultaneous MEEG rows, avoiding an implicit positional split.

*Robust-z guard* - *Post-v2.4 working-tree safeguard.* A policy for degenerate
feature scales that can emit NaN rather than misleading extreme values. It is
not part of the frozen v2.4.0 contract until versioned.

= S - Z

*Subject-anchored coordinates* - *Implemented.* Coordinates preserving
within-subject or within-session geometry.

*Cohort-anchored coordinates* - *Implemented.* Coordinates based on a frozen
external or fitted feature anchor for between-subject comparison.

*`features_projection_z`* - *Implemented in v2.4.0.* A transform-aware feature
surface used when physical-scale features, especially MEG power, require log10
before robust-z and clipping.

*Block-native* - *Implemented derived layer.* Sustained-block windows with
block labels, relative position, and `source_window_index`; appropriate for
slower variables such as HRV.

*BH-FDR* - *Implemented where declared by a validation package.* Benjamini--
Hochberg correction controls a named multiple-comparison family. The current
methods article does not define one global coordinate FDR family.

*LOOSO* - *Implemented in selected analyses.* Leave-one-subject-out
recalculation used for cohort-specific robustness, not a universal property of
all outputs.

*Bootstrap* - *Implemented in selected follow-ups.* Resampling used to
characterize uncertainty in a named validation package.

*Permutation / label-shuffle null* - *Implemented in selected analyses.*
Labels or pairings are shuffled to form a stated null; the null design and
unit of exchangeability must be reported with the result.

*Phase anchor* - *Implemented optional v2.5.0 surface.* An optional per-epoch
extractor for cardiac phase, respiratory phase, RR intervals, HR, respiration,
and HEP-related features. It was absent from the frozen v2.4.0 contract and
remains dataset-dependent.

*Proxy* - *Interpretation boundary.* A reproducible computational correlate
used under a contract; it is not automatically a latent state, biomarker, or
clinical endpoint.

*Observed conditional covariance* - *Implemented analysis quantity.* The
analytic conditional covariance for an observed one-step target, retained
separately from the raw local increment covariance. It is not a substitute for
the production diffusion object.

*Pointwise MCSE* - *Descriptive uncertainty summary.* The standard error of a
prediction or alignment mean across the twelve independent training fits at
the same held-out query, computed as sample standard deviation divided by
`sqrt(12)`. No standard error of the variance is reported. It is not a
confidence interval for a biological effect.

*Raw increment covariance* - *Implemented diffusion quantity.* The local
covariance of raw increments divided by the declared sampling interval. A
fixed evaluation-neighbour set is a property of the bounded M3 benchmark, not
a universal production rule. This quantity remains distinct from an analytic
conditional covariance and from an unobserved latent drift.
