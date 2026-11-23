#import "template_eLife.typ": *
#set page(numbering: "1")

#show: netn-template.with(
  title: "S2 Supplementary Data",
  short-title: "S2 Data",
  subtitle: "Expanded support, dependence, and repeatability results for the modality-specific measurement branches",
  authors: (
    (name: "Robin Langell", affil: 1),
  ),
  affiliations: (
    "Langell Konsult AB, Vallentuna, Sweden",
  ),
  corresponding-author: "Robin Langell, hello@noeticdiffusion.com",
  article-type: "Supplement",
)

= S2.1 Scope and data inventory

This supplement gives expanded numerical summaries for the current measurement-system article. The EEG results are descriptive measurement and support audits; no eyes-open versus eyes-closed hypothesis test, task contrast, or multiplicity-adjusted contrast is reported. Repeated windows are not treated as independent participants. The separate MEG and LFP examples are engineering pilots and are not pooled with EEG.

#figure(
  clean-table(
    columns: (1fr, 1.25fr, 2.3fr),
    header: ([*Source*], [*Role in this article*], [*Scope*]),
    inset: 4pt,
    [OpenNeuro ds005385 v1.0.3 @openneuro_ds005385_v103], [EEG reference cohort], [608 participants; session-1 pre-task EC and EO; one recording per condition.],
    [OpenNeuro ds004148 v1.0.1 @openneuro_ds004148_v101], [EEG repeatability cohort], [60 selected participants; resting EC/EO across sessions 1–3; fixed external ds005385 EC anchor.],
    [OpenNeuro ds004511 v1.0.2], [EEG dynamics integration smoke], [One Rest recording; raw-layer dynamics export only.],
    [OpenNeuro ds003645], [MEG export smoke], [One participant, one FacePerception run; MAG and GRAD kept separate.],
    [DANDI 000458], [LFP feature and support pilot], [One Probe F/MOs session; contact features and a support-limited dynamics audit.],
    [DANDI 000232 / EF0147], [LFP metadata readiness audit], [Metadata only; no signal values read and no features extracted.],
  ),
  kind: table,
  caption: [Datasets used in the current article. Only the first two contribute to the EEG cohort summaries and repeatability estimates.],
)

= S2.2 EEG coordinate support in the ds005385 reference cohort

The metadata-selected reference cohort contained 1,216 recordings and 55,183 windows. All nine coordinates had supported values from all 608 participants in each condition. The alpha-envelope coefficient of variation (CV) had candidate-specific window loss at the declared 10 s segment-edge guard; the other eight coordinates had no observed support loss in the exported masks. Complete-nine support requires all nine values in the same window.

#figure(
  clean-table(
    columns: (1.5fr, 1.1fr, 1.1fr, 1fr, 2fr),
    header: ([*Coordinate*], [*EC valid / windows*], [*EO valid / windows*], [*Participants with support*], [*Observed support loss*]),
    inset: 3pt,
    [#raw("delta_theta_level")], [27,651 / 27,651], [27,532 / 27,532], [608 in each condition], [None in the exported support mask.],
    [#raw("alpha_level")], [27,651 / 27,651], [27,532 / 27,532], [608 in each condition], [None in the exported support mask.],
    [#raw("beta_alpha_ratio")], [27,651 / 27,651], [27,532 / 27,532], [608 in each condition], [None in the exported support mask.],
    [#raw("gamma_level")], [27,651 / 27,651], [27,532 / 27,532], [608 in each condition], [None in the exported support mask.],
    [#raw("hjorth_mobility")], [27,651 / 27,651], [27,532 / 27,532], [608 in each condition], [None in the exported support mask.],
    [#raw("alpha_theta_ratio")], [27,651 / 27,651], [27,532 / 27,532], [608 in each condition], [None in the exported support mask.],
    [#raw("permutation_entropy")], [27,651 / 27,651], [27,532 / 27,532], [608 in each condition], [None in the exported support mask.],
    [#raw("alpha_centroid_hz")], [27,651 / 27,651], [27,532 / 27,532], [608 in each condition], [None in the exported support mask.],
    [#raw("alpha_envelope_cv")], [24,361 / 27,651], [24,215 / 27,532], [608 in each condition], [3,290 EC and 3,317 EO windows within the edge guard.],
  ),
  kind: table,
  caption: [Final coordinate-valid windows from the frozen HDF5 exports. Window support is not an artifact-free or qualified-signal designation.],
)

= S2.3 Complete-nine support and QC status

The complete-nine mask was true for 24,361/27,651 EC windows (88.1%) and 24,215/27,532 EO windows (88.0%). Every participant had at least one complete-nine window. All 1,216 recordings remained unqualified, and #raw("qc_ok_eeg = -1") for every window because artifact QC was unassessed in this reference-cohort audit. Anchor application was reported as computed for all 608 recordings per condition, with no anchor-identity failures reported. The separate EC and EO reference fits are transductive and define different coordinate frames.

= S2.4 Dependence and support-policy sensitivity

The dependence audit first reduced overlapping windows to one participant median per coordinate and condition. “Available” uses every supported window for each coordinate; “matched” uses only windows where all nine coordinates are supported. Both policies retained the same 608 participants per condition. Entries below are descriptive properties of the Spearman correlation matrices.

#figure(
  clean-table(
    columns: (0.8fr, 1fr, 1fr, 1.4fr, 1.4fr),
    header: ([*Condition*], [*Mean absolute matrix difference*], [*Maximum absolute difference*], [*Effective rank, available → matched*], [*Participation ratio, available → matched*]),
    inset: 3pt,
    [EC], [0.0024], [0.0091], [4.3649 → 4.3655], [3.1687 → 3.1651],
    [EO], [0.0025], [0.0101], [4.7141 → 4.7185], [3.4594 → 3.4575],
  ),
  kind: table,
  caption: [Change in the participant-level Spearman summaries under the two window-support policies.],
)

The maximum absolute leave-one-coordinate-out effective ranks across the nine omissions were 4.67 in EC and 4.99 in EO; corresponding minima were 3.63 and 4.01. These diagnostics describe dependence among this nine-component representation. They are not feature-importance tests, estimates of intrinsic neural-state dimension, or evidence for a latent manifold. Full per-coordinate rows are in #raw("project/mnps2_audit/outputs/20261116_mnps_v3_4_measurement_quality_audit/dependency_leave_one_coordinate_out.csv").

= S2.5 ds004148 session-repeatability estimates

For each session pair, the full-cohort analysis summarized participant-level coordinate medians under a common frozen ds005385 EC anchor. ICC values below are two-way absolute-agreement, single-measure estimates, ICC(A,1), rounded to three decimals. Cell order is session 1–2, session 1–3, then session 2–3. These are descriptive estimates, not inferential tests.

#figure(
  clean-table(
    columns: (1.7fr, 1fr, 1fr, 1fr),
    header: ([*EC coordinate*], [*Session 1–2*], [*Session 1–3*], [*Session 2–3*]),
    inset: 3pt,
    [#raw("delta_theta_level")], [0.895], [0.830], [0.808],
    [#raw("alpha_level")], [0.875], [0.885], [0.809],
    [#raw("beta_alpha_ratio")], [0.835], [0.751], [0.802],
    [#raw("gamma_level")], [0.827], [0.874], [0.752],
    [#raw("hjorth_mobility")], [0.806], [0.752], [0.720],
    [#raw("alpha_theta_ratio")], [0.709], [0.675], [0.608],
    [#raw("permutation_entropy")], [0.837], [0.786], [0.687],
    [#raw("alpha_centroid_hz")], [0.954], [0.792], [0.762],
    [#raw("alpha_envelope_cv")], [0.823], [0.741], [0.727],
  ),
  kind: table,
  caption: [EC coordinate-level ICC(A,1) by session pair.],
)

#figure(
  clean-table(
    columns: (1.7fr, 1fr, 1fr, 1fr),
    header: ([*EO coordinate*], [*Session 1–2*], [*Session 1–3*], [*Session 2–3*]),
    inset: 3pt,
    [#raw("delta_theta_level")], [0.480], [0.120], [0.796],
    [#raw("alpha_level")], [0.846], [0.688], [0.732],
    [#raw("beta_alpha_ratio")], [0.844], [0.676], [0.710],
    [#raw("gamma_level")], [0.686], [0.598], [0.759],
    [#raw("hjorth_mobility")], [0.407], [0.041], [0.668],
    [#raw("alpha_theta_ratio")], [0.891], [0.363], [0.291],
    [#raw("permutation_entropy")], [0.621], [0.285], [0.583],
    [#raw("alpha_centroid_hz")], [0.924], [0.898], [0.924],
    [#raw("alpha_envelope_cv")], [0.674], [0.600], [0.710],
  ),
  kind: table,
  caption: [EO coordinate-level ICC(A,1) by session pair.],
)

#figure(
  clean-table(
    columns: (1.7fr, 1fr, 1fr, 1fr),
    header: ([*EO−EC coordinate*], [*Session 1–2*], [*Session 1–3*], [*Session 2–3*]),
    inset: 3pt,
    [#raw("delta_theta_level")], [0.417], [0.273], [0.683],
    [#raw("alpha_level")], [0.695], [0.655], [0.630],
    [#raw("beta_alpha_ratio")], [0.756], [0.630], [0.672],
    [#raw("gamma_level")], [0.341], [0.249], [0.517],
    [#raw("hjorth_mobility")], [0.508], [0.272], [0.662],
    [#raw("alpha_theta_ratio")], [0.660], [0.666], [0.547],
    [#raw("permutation_entropy")], [0.651], [0.465], [0.588],
    [#raw("alpha_centroid_hz")], [0.814], [0.530], [0.527],
    [#raw("alpha_envelope_cv")], [0.864], [0.793], [0.813],
  ),
  kind: table,
  caption: [Within-participant EO−EC difference ICC(A,1) by session pair.],
)

The number included varied by row. Across the three session pairs, EC/EO level rows included 43–52, 45–47, and 44–50 of the 60 selected participants, respectively. EO−EC rows included 36–40, 36–39, and 37–42. Exact row-specific participant counts, bias, sample SD, 95% Bland–Altman limits, median absolute differences, Spearman rank correlations, and exclusion reasons are preserved in #raw("mndm/docs/ds004148_session_repeatability_fullcohort/artifacts/repeatability_summary.csv"), #raw("mndm/docs/ds004148_session_repeatability_fullcohort/artifacts/repeatability_eligibility.csv"), #raw("mndm/docs/ds004148_session_repeatability_fullcohort/artifacts/participant_session_levels.csv"), and #raw("mndm/docs/ds004148_session_repeatability_fullcohort/artifacts/participant_session_eo_minus_ec.csv"). Spearman correlation reflects rank order and is not a measure of agreement. Anchored coordinates are dimensionless, including #raw("alpha_centroid_hz").

#bibliography("NeuralManifoldReferences.bib", style: "apa")

