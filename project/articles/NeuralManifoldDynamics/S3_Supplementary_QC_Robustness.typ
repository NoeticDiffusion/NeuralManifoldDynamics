#import "template_eLife.typ": *
#set page(numbering: "1")

#show: netn-template.with(
  title: "S3 Supplementary QC and Robustness",
  short-title: "S3 QC and Robustness",
  subtitle: "Support, realization compatibility, and bounded robustness checks",
  authors: (
    (name: "Robin Langell", affil: 1),
  ),
  affiliations: (
    "Langell Konsult AB, Vallentuna, Sweden",
  ),
  corresponding-author: "Robin Langell, hello@noeticdiffusion.com",
  article-type: "Supplement",
)

= S3.1 Quality-state distinctions

This article keeps computability, comparability, and qualification separate. A finite estimator output is evidence of computability only. Comparisons require compatible sensor sets, references, preprocessing, estimator realizations, time bases, units, and support. Qualification requires additional evidence for a declared use and is not assigned by this technical audit. QC and support flags are retained as observations rather than used to zero-fill or discard unrelated measurements.

#figure(
  clean-table(
    columns: (1.3fr, 1.5fr, 2.7fr),
    header: ([*Audit*], [*Observed state*], [*Interpretation*]),
    inset: 4pt,
    [ds005385 EEG reference audit], [qc_ok_eeg = -1 in all 55,183 windows; zero qualified recordings.], [Artifact QC was unassessed. Coordinate validity and anchor application do not certify signal cleanliness.],
    [ds004148 EEG repeatability], [QC matched 26,640 windows; participant summaries passed a prespecified support floor when eligible.], [Eligibility supports the reported summary; it does not prove windows are artifact-free or establish biological stability.],
    [ds003645 MEG export smoke], [121 windows; MAG and GRAD reported separately; no anchor applied.], [Successful serialization and matching to cached features do not validate a common metric or cross-sensor comparability.],
    [DANDI 000458 LFP pilot], [Contact features computed; no supported local dynamics estimates or paired held-out predictions.], [The dynamics outcome is a support failure under the frozen rule, not a negative biological result.],
    [DANDI 000232 / EF0147], [Metadata-only review; no signal values or features.], [Deferred for this finite-lag protocol; other measurement uses were not assessed.],
  ),
  kind: table,
  caption: [Quality state and scope across the distinct measurement branches.],
)

= S3.2 EEG support and cohort-level checks

In OpenNeuro ds005385 version 1.0.3 @openneuro_ds005385_v103, all nine coordinates had at least one valid window per participant and condition. The eight coordinates other than alpha-envelope CV were valid in all exported windows. The alpha-envelope CV was valid in 24,361/27,651 EC windows and 24,215/27,532 EO windows; 3,290 and 3,317 windows, respectively, were within its declared symmetric 10 s source-segment edge guard. Every participant had at least one complete-nine window, while only 88.1% of EC and 88.0% of EO windows were complete-nine. These counts are support outcomes, not pass rates for artifact QC.

The OpenNeuro ds004148 version 1.0.1 selection @openneuro_ds004148_v101 fixed 60 participants, all three sessions, and resting EC/EO recordings from metadata before coordinate inspection. Session 1 outputs were reused and sessions 2–3 were re-extracted. Every recording received the same frozen ds005385 EC anchor. QC matched all 26,640 windows. The support floor was at least 30 QC-eligible windows and 120 unique seconds per participant × session × condition × coordinate.

#figure(
  clean-table(
    columns: (1.45fr, 1.2fr, 1.2fr, 1.8fr),
    header: ([*ds004148 support measure*], [*Eligible / total*], [*Below support floor*], [*Unit counted*]),
    inset: 4pt,
    [Coordinate summaries], [2,858 / 3,240], [382], [participant × session × condition × coordinate],
    [Complete-nine summaries], [306 / 360], [54 groups with one or more failing coordinates], [participant × session × condition],
    [Repeatability rows], [81 rows], [Included n = 36–52], [coordinate/condition or EO−EC × session pair],
  ),
  kind: table,
  caption: [Eligibility and support counts for the three-session EEG analysis.],
)

Detailed coordinate-level eligibility and excluded-subject reasons are in #raw("mndm/docs/ds004148_session_repeatability_fullcohort/artifacts/repeatability_eligibility.csv"). The complete-nine support and realization checks are preserved in #raw("complete_9d_support.csv") and #raw("realization_compatibility.csv") in the same artifacts directory. One hundred realization-comparison rows recorded channel-order or case-only label differences. These remained QC observations; a comparison was eligible only where normalized channel set, source surface, reference, preprocessing, estimator identity, time grid, and units were compatible.

= S3.3 Robustness to coordinate-window support policy

For the ds005385 dependency summary, the available-coordinate and matched-complete-nine policies retained the same 608 participants in both conditions. The mean absolute differences between their Spearman matrices were 0.0024 (EC) and 0.0025 (EO); maximum absolute differences were 0.0091 and 0.0101. Effective ranks changed by less than 0.005 in either condition. Participation ratios changed by less than 0.004. This check indicates that the edge-guard missingness made little difference to these particular participant-level dependency summaries. It does not establish robustness to other preprocessing, artifacts, cohorts, references, or estimators.

The leave-one-coordinate-out effective-rank diagnostic varied from 3.63 to 4.67 in EC and 4.01 to 4.99 in EO. These ranges are properties of this correlated coordinate set under the stated aggregation. They are not feature-selection results or estimates of intrinsic dimensionality. The dependency audit and exact row-level values are archived in #raw("project/mnps2_audit/outputs/20261116_mnps_v3_4_measurement_quality_audit/").

= S3.4 MEG support audit

The ds003645 sub-002 FacePerception run produced 121 windows. The three cached coordinate variants matched the exported raw values and support masks to absolute tolerance 1e-12. The 102 MAG channels and 204 GRAD channels were processed and reported separately, with the same window support counts for each family.

#figure(
  clean-table(
    columns: (1.25fr, 0.55fr, 1.5fr, 2.8fr),
    header: ([*Variant*], [*Dimensions*], [*Local Jacobian*], [*Dynamics support*]),
    inset: 3pt,
    [existing_coordinate_set], [9], [Computed at 119/121 centers.], [Velocity: 120/121; pooled rate: 121/121; blocked cross-fit: 112/121; increment-covariance rate: 121/121.],
    [removal_only], [7], [Not testable; current Jacobian requires 9D.], [Drift not testable under the current chart contract; increment-covariance rate: 121/121.],
    [alpha_replacement], [9], [Computed at 114/121 centers.], [Velocity: 115/121; pooled rate: 116/121; blocked cross-fit: 107/121; increment-covariance rate: 116/121.],
  ),
  kind: table,
  caption: [Window support from the one-recording MEG export smoke, with MAG and GRAD kept separate.],
)

The alpha-replacement variant had five unsupported edge windows (epochs 0–2 and 119–120) under its envelope-CV support rule. No coordinates were padded. The raw MEG values were unscaled and had no EEG anchor; anchored dynamics and the anchored one-step operator were therefore not testable. The raw estimates use mixed source units and Euclidean neighborhoods. This smoke supports export integration only, not a qualified metric, local identifiability, repeatability, or equivalence between MAG and GRAD.

= S3.5 LFP support audit

The DANDI 000458 candidate used one continuous Probe F/MOs recording. Contact-level 3–5 Hz centroid, bandpower, and envelope-CV were summarized in two depth groups, with at least 75% of the intended contacts required in each group (41/54 and 24/32 contacts). The finite-lag support thresholds were fixed before the pilot and were not relaxed after insufficient support was observed.

#figure(
  clean-table(
    columns: (1fr, 1.15fr, 1.2fr, 1.5fr),
    header: ([*State*], [*Eligible transitions*], [*Distinct batches (28 required)*], [*Supported estimates / held-out predictions*]),
    inset: 3pt,
    [Awake], [103], [5], [0 / 0],
    [Isoflurane], [75], [3], [0 / 0],
    [Recovery], [65], [4], [0 / 0],
  ),
  kind: table,
  caption: [Observed Probe F/MOs lag-1 transition support. Distinct batches are not assumed to be statistically independent.],
)

The candidate neighborhood required at least 28 distinct non-neighbor source batches per query, at least 30 eligible transition rows per group/state/lag, and at least 20 scorable held-out batches for blocked evaluation. The 28-, 30-, and 20-unit thresholds are operational project rules without formal statistical or synthetic-recovery calibration. Their failure means the pilot yielded no supported local estimates or paired predictions; it does not show that the animal state had no dynamics.

The separate DANDI 000232 / EF0147 review inspected metadata only. Its 244 ElectricalSeries are trial-aligned excerpts; boundary-safe context, reference realization, calibration provenance, and physical contact mapping were unresolved. The largest task-rule stratum had at most 19 eligible lag-1 transition rows and at most 7 eligible non-neighbor source batches per query, below the existing thresholds, and no blocked fold met the 20-held-out-batch rule. The dataset was deferred for this finite-lag protocol; no conclusion is made about other LFP measurements or trial-level uses.

= S3.6 Robustness checks not performed

No EEG artifact-balance-by-condition analysis, EOG/blink audit, EMG-proxy comparison, physiology-coupling test, task contrast, within-subject permutation, leave-one-subject-out effect analysis, or bootstrap effect interval is reported in this article. The earlier generic template's task, regional, physiological, and confound tables did not correspond to this study and are not evidence for its findings. Frozen ds005385 cohort values also predate serialization of the exact PSD runtime backend; the audited files do not allow those historic values to be assigned retrospectively to the multitaper path or Welch fallback.

= S3.7 Summary

#netn-box(title: [QC and robustness summary])[
  The supported evidence is that the declared EEG coordinates and their support can be audited in the selected exports, that a participant-level dependency summary changes little under two specified window-support policies, and that repeatability estimates vary by coordinate, condition, and session pair. The MEG example is a one-run separate-sensor export smoke. The LFP feature computation is demonstrated in one session, while its finite-lag real-data analysis did not meet the frozen support thresholds. None of these checks establishes artifact absence, cross-modal equivalence, biological stability, clinical qualification, or predictive usefulness.
]

#bibliography("NeuralManifoldReferences.bib", style: "apa")

