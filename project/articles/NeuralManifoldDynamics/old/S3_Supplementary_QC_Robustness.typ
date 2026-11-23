// S3 Supplementary QC and Robustness for audited reference runs.

#import "../network_neuroscience_template.typ": *

#show: netn_template.with(
  title: "S3 Supplementary QC and Robustness",
  short_title: "S3 QC/Robustness - NeuralManifoldDynamics",
  subtitle: "Support, artifact, physiology, and modality-specific audits",
  authors: (
    (name: "Robin Langell", affil: 1),
  ),
  affiliations: (
    "Langell Konsult AB, Vallentuna, Sweden",
  ),
  corresponding_author: "Robin Langell, hello@noeticdiffusion.com",
  article_type: "Supplement",
  keywords: ("QC", "robustness", "geometry", "HRV", "MEG"),
)

_Historical archive: this supplement belongs to an earlier NeuralManifoldDynamics article version. It is not part of the current multimodal methods article or evidence for the present measurement definitions._

= S3.0 Sample and epoch coverage overview

Coverage is reported separately for completed exports and inferential subsets.
The numbers below are not interchangeable:

#table(
  columns: (2.1fr, 1.2fr, 1.8fr, 1.8fr),
  table.header([*Run or cohort*], [*Completed units*], [*Coverage evidence*], [*Inferential sample*]),
  [ds004511 reference], [132 HDF5 outputs], [45 subjects, 3 tasks], [not a contrast package],
  [ds003838 full export], [130 HDF5 outputs], [6,543 blocks; 27,670 windows], [N = 62 complete-case stages],
  [ds003645 MEG pilot], [5 subjects], [8/4/2 s window checks], [pilot only],
  [ds006848 verbal WM], [42,593 feature rows], [30 subjects; event-locked QC], [n = 30 for listed contrasts],
  [DANDI 000006 Units], [1 session], [1,752 mnps_3d windows], [transport/descriptive only],
  [DANDI 000458 LFP], [2 probes, 1 session], [3,433 epochs/probe; 315/384 and 200/384 channel pass], [transport/descriptive only],
)

The ds003838 full export and corrected statistics package use different run
directories. The latter reports 26,460 windows and `N = 62`; the former
reports 27,670 windows and 130 completed HDF5 subjects.

= S3.1 Physiological QC pass rates

The ds006848 validation found inverted ECG polarity in 39,477 of 42,593 rows
(92.7%). After polarity correction, the population median HR was 76.2 bpm and
RMSSD was 40.7 ms. The same validation found retrieval-event overlap in
37,352 of 42,593 HRV windows (87.7%). Only approximately 2% of working-memory
windows passed the strict clean HRV gate; rest-phase windows had an
approximately 85% pass rate. Working-memory HRV inference was consequently
withheld.

The ds003838 full run recorded HRV artifact, coverage, and quality columns in
the block-native sidecars and preserved `anchor_quality`. These are contract
and QC outputs; they do not by themselves prove physiology-coordinate
causality.

= S3.2 EEG/EOG artifact balance across tasks

The ds006848 artifact follow-up found a condition-associated frontal
30--45 Hz contrast (`chi^2 = 23.00`, `p = 0.00004`) whose direction was
opposite to the manifold `m`/`d` ordering. Temporal muscle power was not
significant (`p = 0.61`). This narrows, but does not eliminate, high-frequency
artifact as an alternative explanation. ICA/RANSAC cleaning was not run for
that claim package, so the result must not be described as ICA-clean EEG.

Blink and EOG quality are retained as feature-level covariates and QC fields.
A single consolidated blink-rate table is not archived in the current article
evidence pack; no missing values are imputed here.

= S3.3 Robustness and confound audit

== Intersection sample and epoch matching

For ds003838, the full block-native join matched source windows with fraction
1.0. Statistical contrasts were restricted to `N = 62` subjects with complete
stage coverage. For ds006848, event-window purity was evaluated with the 04c
filters; short events and overlapping 8 s windows make some phases unsuitable
for direct interpretation.

== Leave-one-subject-out and bootstrap

The ds006848 encoding `m` and `d` contrasts remained significant in the
cohort-specific leave-one-subject-out analysis for 30/30 omissions. Bootstrap
intervals and directional consistency were reported in the 04e follow-up, with
approximately 67--77% directional consistency depending on the metric. These
checks belong to the ds006848 package and are not a blanket robustness claim
for every modality.

== Within-subject permutation

The ds003645 pilot used label-shuffle nulls for subject-level event-response
cosines. Three of five subjects had observed values above their null means,
and two had shuffle `p < 0.05`; the d-family sign reversal in other subjects
created a bimodal cross-modal pattern. The later holdout found no matched
EEG--MEG residual topographic correspondence beyond its nulls.

== Artifact balance

The ds006848 frontal high-frequency result pointed opposite to the manifold
effect and temporal muscle power was non-significant, but missing ICA remains
an explicit limitation. MEG analyses require `features_projection_z` and
per-row `row_source`; historical positional half-split behavior is not an
acceptable provenance substitute.

== Physiological confound controls

The ECG polarity correction and HRV contamination gates are measurement
validity controls. They do not transform HRV into a phase-pure task measure.
PPG and pupil quality are dataset-dependent. For DANDI 000458, channel pass
counts, stimulation midpoint guards, contact count, and reference choice are
reported as one-session sensitivity information; no cross-probe geometry claim
is made.

= S3.4 Follow-ups

The repository contains several named follow-up tracks rather than the
template's generic Script A--D sequence.

== S3.4a Event purity and physiological follow-up

Scripts `04c_window_overlap_purity_audit.py`, `pr01_eeg_artifact_audit.py`,
`pr03_shortwindow_robustness.py`, and `pr04_artifact_balance.py` evaluate
window overlap, artifact proxies, shorter-window feasibility, and artifact
balance for ds006848. Their outputs are cohort-specific and preserve the
decision to withhold WM-phase HRV.

== S3.4b MEG readiness and holdout

The ds003645 pilot passed the HDF5 contract, feature completeness, window
robustness, and Jacobian-validity gates, yielding readiness `0.7879`. This is
below the `0.80` expansion threshold and was labelled usable pilot mapping.
The holdout's residual topographic correspondence was negative, so the
cross-modal claim remains exploratory.

== S3.4c Conventional-feature baseline

The classical EEG comparator pack reports bandpower, slowing ratios,
complexity, and connectivity summaries beside MNPS features. In ds006848,
the frontal theta ordering was approximately inverse to the MNPS `m`/`d`
ordering, a useful dissociation. This is a descriptive baseline comparison,
not a test of incremental predictive validity.

== S3.4d DANDI, LFP, and infant EEG transport audits

DANDI 000006 Units and 000458 LFP paths complete smoke or one-session QC
checks. Figshare BDF ingest completed a 71-file run with 5,273 retained
epochs and 70 coverage-passing outputs, but behavioural condition labels were
not recoverable from the available Status field. These results document
transport and QC, not clinical or circuit-level manifold inference.

*Units (000006, 000009).* The 000006 smoke session (`sub-anm369962`) produced
1,752 windowed `mnps_3d` rows and a full `(1750, 3, 3)` Jacobian export with
zero pipeline errors, using the dedicated ephys-to-9D subcoordinate map
described in S1.3. This is read as a starter mapping, not validated manifold
evidence: the 000009 inventory run (10 units, 32 electrodes) found unit-quality
labels unresolved (`unknown`) rather than pass/fail.

*LFP (000458).* A channel-count smoke run processed 32 channels into 28
epochs. The full-session two-probe QC pass retained 315/384 channels for
probe B (48 depth-balanced contacts selected) and 200/384 channels for probe
F (32 contacts selected), yielding 3,433 epochs per probe in the full-session
pass. The archived summary also reports 605 awake, 438 isoflurane, and 794
recovery epochs per probe after state/stimulation-aware epoching with a 1 s
stimulation guard. These state counts sum to 1,837, not 3,433; the available
archival record does not account for the remaining 1,596 epochs. Treat these
as separate summaries, not an exhaustive partition of the full-session count.
A contact-count/reference-choice sensitivity check compared an
8-contact native-reference configuration against 4-contact and
median-within-probe-reference variants; the 4-contact/median-reference cell
was not run, and cross-run `mnps_3d` coordinate orientation was not
identified between configurations.

*Readiness gate.* Unlike the MEG pilot (S3.4b), no weighted readiness score
has been computed for either the Units or the LFP ephys path. Both are
scoped as transport-and-descriptive sensitivity checks. Both 000006 and
000458 are rodent recordings; no cross-species claim is made relative to the
human EEG/MEG/fMRI reference paths.

= S3.5 Reviewer-facing robustness summary

#netn_box(title: [Robustness summary])[
  The strongest supported statements are contract-level: valid support is
  recorded, source-window joins can be audited, and modality-specific failure
  modes are surfaced. The ds003838 HRV/block-native export and its complete-case
  stage contrasts survive the stated corrected-ECG pipeline, while ds006848
  working-memory HRV is explicitly gated out by contamination. The MEG path
  remains an exploratory pilot below its readiness threshold, and the later
  residual-topography holdout is negative. Chart invariance, independent
  replication, clinical prediction, and a universal cross-modality robustness
  claim remain unestablished.
]

= S3.6 Synthetic estimator and coordinate robustness

The analysis-owned CHART and M3 extension studies exercise support and
provenance contracts separately from empirical cohort QC. Evaluation targets
are excluded from cross-fit labels; fixed held-out queries, neighbour indices,
and observed raw-increment covariance are shared across comparators. Padding
and inter-segment increments are excluded, and target-only perturbations leave
both candidate predictions unchanged. Undefined statistics retain a validity
reason rather than being replaced by zero.

The fixed mixed orthogonal rotation and isotropic scaling preserve the expected
vector transformations, with maximum discrepancy approximately
`2.9 × 10⁻¹⁴`. Anisotropic scaling changes Euclidean neighbours and estimator
outputs (maximum discrepancies approximately 2.88 for local mean and 4.96 for
local linear); the oracle coordinate identity remains valid. The associated
alignment and effective-dimension changes are metric dependence expected under
anisotropic scaling, not failed coordinate invariance. The unregularized CHART
alignment diagnostic is distinct from the epsilon-regularized production C1
quantity. A 3D-to-2D toy projection has identical projected current states but
different projected expected futures because of a hidden coordinate; this is an
information-loss counterexample, not a failed coordinate transformation.

The M3 extension's six cases all completed with the fixed-query design
reported in S2.8. Its order-corruption controls are labelled as artificial
cross-segment controls, not temporal nulls. The finite synthetic predictions
therefore support implementation and contract checks only. They do not qualify
empirical drift, an Itô drift, Markov sufficiency, a production C1 object, a
biological interpretation, or a universal estimator choice. The separate W_Q
sensitivity diagnostic on the same sub-99 night computed 864/864 planned
comparisons in 3D and 845/864 in 9D, with 19 `post_dtype_psd_failure` cases;
9D `v_norm` remains METHOD-LIMITED. These overlapping same-night perturbation
trials are not 864 independent observations, do not select a production
horizon, and do not close a biological gate. The source full-night 9D status
`reachability_numerical_overflow` remains retained separately; the synthetic
CHART/M3 checks do not override it.

#bibliography("../NeuralManifoldReferences.bib")
