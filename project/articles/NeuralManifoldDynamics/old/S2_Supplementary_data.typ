// S2 Supplementary Data for audited validation runs.

#import "../network_neuroscience_template.typ": *

#show: netn_template.with(
  title: "S2 Supplementary Data",
  short_title: "S2 Data - NeuralManifoldDynamics",
  subtitle: "Audited validation summaries and explicit unavailable result families",
  authors: (
    (name: "Robin Langell", affil: 1),
  ),
  affiliations: (
    "Langell Konsult AB, Vallentuna, Sweden",
  ),
  corresponding_author: "Robin Langell, hello@noeticdiffusion.com",
  article_type: "Supplement",
  keywords: ("validation", "HRV", "block-native", "MEG", "QC"),
)

_Historical archive: this supplement belongs to an earlier NeuralManifoldDynamics article version. It is not part of the current multimodal methods article or evidence for the present measurement definitions._

= S2.0 Global coordinate contrast family

An article-level global `mnps_3d`/`coords_9d` pairwise contrast family with a
single BH-FDR correction is *not available*. The manuscript is a measurement
contract paper, not a pre-registered Task A versus Task B coordinate experiment.
The original template's global-null table is therefore intentionally not
filled. Dataset-specific contrasts below must not be read as a replacement for
that missing article-wide family.

= S2.1 ds003838 primary validation contrasts

The full ds003838 export completed for 130 subjects, 6,543 blocks, and 27,670
block-native windows with a source-window match fraction of 1.0. The corrected
stage-statistics run used 26,460 windows and `N = 62` subjects with complete
stage coverage. The table reports values from
`project/diary/127_20260625_full_stats_package_ds003838.md`; the two window
counts belong to different run directories and are not pooled.

#table(
  columns: (2.5fr, 1.2fr, 1.4fr, 1.4fr),
  table.header([*Metric*], [*$chi^2$*], [*Friedman $p$*], [*Interpretation*]),
  [`traj_path_length`], [226.2], [`8.8 × 10⁻⁴⁸`], [stage-dependent],
  [`traj_efficiency`], [221.8], [`7.8 × 10⁻⁴⁷`], [stage-dependent],
  [`vagal_index`], [88.0], [`3.4 × 10⁻¹⁸`], [stage-dependent],
  [`sympathetic_index`], [63.7], [`4.8 × 10⁻¹³`], [stage-dependent],
  [`ecg_hrv_hr_mean_bpm`], [62.6], [`8.3 × 10⁻¹³`], [stage-dependent],
  [`ecg_hrv_sdnn_ms`], [62.0], [`1.1 × 10⁻¹²`], [stage-dependent],
  [`ecg_hrv_rmssd_ms`], [24.1], [`7.7 × 10⁻⁵`], [stage-dependent],
  [`anchor_index`], [13.6], [`8.6 × 10⁻³`], [stage-dependent],
  [`traj_mean_curvature`], [7.6], [0.11], [not significant],
)

The registered listen--memory contrasts for `vagal_index` were:

#table(
  columns: (2.2fr, 1.4fr, 1.8fr),
  table.header([*Contrast*], [*Cohen's $d$*], [*Wilcoxon $p$*]),
  [listen vs mem5], [+1.718], [`6.9 × 10⁻¹¹`],
  [listen vs mem9], [+1.750], [`1.8 × 10⁻¹⁰`],
  [listen vs mem13], [+1.995], [`1.2 × 10⁻¹⁰`],
)

The later one-line summary in the same diary contains a different `+2.085`
value for a row labelled C3. That contrast definition is not resolved here;
the explicit pairwise table above is the value used for the manuscript.

= S2.2 Regional contrasts

The contract supports regional fMRI network and EEG channel-group trajectories,
regional Jacobians, and optional regional stratified outputs. The cited
ds003838 run enables regional trajectories but not regional block Jacobians.
No complete article-level table of regional coordinate contrasts with a
predeclared multiple-comparison family is archived in the current evidence
pack. Regional capability is therefore reported as implemented, while a full
regional inferential table remains unavailable.

= S2.3 Block-level contrasts

The release-bound ds003838 block-native aggregate is descriptive coverage
evidence:

#table(
  columns: (1.8fr, 1.2fr, 1.2fr),
  table.header([*Stage*], [*Blocks*], [*Windows*]),
  [rest], [65], [3,961],
  [listen], [2,171], [7,944],
  [mem5], [1,273], [2,905],
  [mem9], [1,469], [5,361],
  [mem13], [1,565], [7,499],
  [*Total*], [*6,543*], [*27,670*],
)

These totals establish a successful block/window join for the full export. They
are not the inferential sample for the corrected stage-statistics package.

= S2.4 Secondary coupling

The ds003838 statistics package reports all 12 inter-network `coupl_*` terms
as stage-dependent, with Friedman statistics approximately 65--97 and
uncorrected *p* values from `2.6 × 10⁻¹³` to `5.4 × 10⁻²⁰`. The archived
summary identifies central-to-parietal and temporal-from-central coupling as
among the strongest effects.

These are stage-wise coupling results, not an EAP partial-correlation table.
An article-level table of partial correlations, covariate choices, and FDR
family size is not locked and remains unavailable.

= S2.5 Secondary physiology--coordinate coupling

HRV and anchor indices are exported on the same block-native grid and are
included in the ds003838 stage statistics. This establishes a usable
physiology--coordinate measurement surface and stage dependence under the
specified task contract. It does not establish causality, a general autonomic
biomarker, or a cross-dataset coupling law.

= S2.6 Physiological and temporal confounds

No single combined partial-correlation table exists across all follow-up
scripts. The auditable confound results are instead distributed across QC
artifacts:

- ds006848 ECG polarity was inverted in 92.7% of feature rows before
  correction; the corrected median HR was 76.2 bpm and RMSSD 40.7 ms.
- 87.7% of 60 s ds006848 working-memory HRV windows overlapped retrieval
  events, so working-memory HRV inference was withheld.
- The ds006848 frontal 30--45 Hz artifact contrast differed by condition but
  pointed opposite to the manifold `m`/`d` ordering; temporal muscle power was
  not significant (`p = 0.61`).
- Window overlap and event-purity results are reported in the 04c/04d scripts,
  while LOOSO/bootstrap results are in 04e. These are cohort-specific audits,
  not one combined FDR family.

= S2.7 Descriptive summary by task

The available task descriptors are the ds003838 stage labels
`rest`, `listen`, `mem5`, `mem9`, and `mem13`, together with the block/window
counts in S2.3. The reference ds004511 metadata snapshot contains 132 HDF5
outputs spanning 45 subjects and three tasks. Tonic SCL, EMG-RMS, and other
descriptive columns are not inserted here unless a locked run table supplies
their exact aggregation and source path.

= S2.8 Analysis-owned synthetic validation

Two analysis-owned synthetic studies provide bounded numerical validation of
the existing local-dynamics candidates. They are not additional empirical
datasets and do not alter the ingest measurement contract. The prospective
protocols, frozen numeric specifications, and outcome reports are retained in
`project/astra/m3_chart_20260906/` and `project/astra/m3_extension_20260906/`.

#table(
  columns: (1.5fr, 2.0fr, 2.1fr, 2.6fr),
  table.header([*Study*], [*Fixed design*], [*Computed evidence*], [*Status and scope*]),
  [M3 benchmark], [27 scenario cells; 216 comparator rows], [Known discrete 3D truth; existing local mean and affine candidates plus controls], [PASS for implementation; PARTIAL PASS for M3 qualification],
  [CHART], [96 points; fixed mixed rotation, isotropic and anisotropic maps], [Affine transformed-vector discrepancy at most `2.9 × 10⁻¹⁴`; covariance checked through a transformed Cholesky factor], [PASS for implementation and bounded coordinate checks],
  [M3 extension], [6 cases; 12 training fits; 3 held-out paths × 48 = 144 fixed queries per case], [288 fitted comparator rows with finite predictions; six oracle rows reported separately], [PASS for bounded implementation and frozen synthetic experiment; PARTIAL PASS / METHOD-LIMITED for scientific qualification],
)

The M3 extension compares local-mean and local-affine candidates at the same
held-out queries. The values below are means over the twelve training fits;
normalized RMSE is divided by the case-specific truth RMS, and alignment rank
is Spearman correlation with the same-neighbour oracle alignment.

#table(
  columns: (2.4fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
  table.header([*Synthetic case*], [*Mean nRMSE*], [*Linear nRMSE*], [*Mean rank*], [*Linear rank*]),
  [full linear 3D], [0.338], [0.369], [0.938], [0.946],
  [full nonlinear 3D], [0.330], [0.349], [0.963], [0.961],
  [full linear 9D], [0.536], [0.878], [0.787], [0.772],
  [full nonlinear 9D], [0.554], [0.768], [0.809], [0.770],
  [latent linear 9D → observed 3D], [0.347], [0.391], [0.889], [0.916],
  [linear 3D + observation noise], [0.295], [0.279], [0.949], [0.959],
)

The two cross-segment order-corruption controls had normalized RMSE values of
1.152--1.720 and alignment ranks from −0.096 to +0.296. They use a global
circular source-order roll across held-out segments; they are an artificial
control and not a within-recording temporal null or an independent null
sample. The full linear and nonlinear 9D simulations are known-truth test
systems, not the canonical neural `coords_9d` feature object. The exact target
in the partial and noisy cases is conditional on the current observed state,
not latent velocity or a history-conditioned forecast.

The Gaussian memory witness is approximately `6.58 × 10⁻⁹` in Frobenius norm
for the partial-observation case and `0.0441` for the noisy case. Fully observed
linear 3D and 9D witnesses are approximately `2.0 × 10⁻¹⁷` and
`4.4 × 10⁻¹⁷`, respectively. The small partial-observation value is a resolved
but tiny model diagnostic; the noisy value shows that current-observation
target recovery does not by itself establish observed-state sufficiency.
These values are not effect sizes, biological evidence, or acceptance
thresholds.

The reports retain the source, runtime, protocol, numeric-specification, and
artifact hashes. The compact provenance map is:

#table(
  columns: (1.2fr, 2.4fr, 2.8fr, 2.8fr),
  table.header([*Study*], [*Fixed design*], [*Frozen references*], [*Evidence/status*]),
  [M3 benchmark], [27 scenario cells; 216 comparator rows], [`m3_numeric_spec.json` (SHA256 prefix `0e7cec90da5e`)], [PASS for implementation and bounded affine checks; PARTIAL PASS for M3 qualification],
  [CHART], [96 points; fixed mixed rotation, isotropic and anisotropic maps], [`chart_numeric_spec.json` (SHA256 prefix `36adcfb98985`); `chart_final.json`], [Affine transformed-vector discrepancy at most `2.9 × 10⁻¹⁴`; covariance checked through a transformed Cholesky factor],
  [M3 extension], [6 cases; 12 training fits; 3 held-out paths × 48 = 144 fixed queries per case], [`numeric_spec.json`; `freeze_manifest.json`; primary artifact], [288 fitted comparator rows with finite predictions; six oracle rows reported separately; PARTIAL PASS / METHOD-LIMITED for scientific qualification],
)

Full SHA256 values and runtime details remain in the cited `report.md` files.
Candidate status remains `candidate_not_qualified`; no biological claim or
automatic scientific PASS is emitted.

#bibliography("../NeuralManifoldReferences.bib")
