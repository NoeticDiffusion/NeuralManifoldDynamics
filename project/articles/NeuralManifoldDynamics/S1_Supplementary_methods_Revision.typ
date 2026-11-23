#import "template_eLife.typ": *
#set page(numbering: "1")

#show: netn-template.with(
  title: "Supplementary Methods: MNPS 3.4 EEG",
  short-title: "Supplementary Methods S1",
  subtitle: "Estimators, anchors, support, and outputs",
  authors: (
    (name: "Robin Langell", affil: 1),
  ),
  affiliations: (
    "Langell Konsult AB, Vallentuna, Sweden",
  ),
  corresponding-author: "Robin Langell, hello@noeticdiffusion.com",
  keywords: ("supplementary methods", "EEG", "robust reference", "HDF5"),
  article-type: "Supplement"
)

= S1.1 Scope and identity

This supplement specifies the implementation details for the experimental flat nine-coordinate measurement definition described in the main article. The schema identifier is mndm.mnps_v3_4.experimental_flat_9d.v1. The repository package version is 3.1.0; 3.4 names a measurement definition, not a software release. The authoritative implementation is under mndm/src/mndm/mnps/v3_4 and its opt-in configuration is under mndm/config/v.3.4.

The nine values are emitted at the window level in fixed order. No subject-level universal summary is created by this definition. A downstream analysis can choose a window, a session summary, or a trajectory according to its own protocol.

= S1.2 Common preprocessing and window support

The ds005385 reference configuration uses session-1 pre-task eyes-open and eyes-closed data. The realized surface is 64 EEG channels with average reference, a configured 50 Hz notch applied before a 1-45 Hz bandpass as separate MNE operations, and resampling to 250 Hz. Source labels and their order are preserved. The reference cohort and target recordings must declare the channel set, reference, frequency, filters, sampling rate, time grid, and estimator realization. The notch and low-pass are separately specified; no equivalence between their effects is assumed.

Windows have an 8 s duration and a 4 s step. The exported start and end times define the effective grid. Windows do not cross source-file boundaries or detected discontinuities. The alpha-envelope candidate is calculated from continuous finite segments and requires 10 s of source support on each side of the window. A window inside the edge guard or spanning a segment boundary is unsupported for that candidate. This candidate-specific restriction does not erase other computable coordinates.

The reference cohort used metadata and required sidecars for selection. It did not use signal values to choose recordings. A missing raw-file hash means that selection-to-HDF5 identity is not a byte-level signal verification. Artifact QC was not established by the selection.

= S1.3 Feature definitions

The common EEG extractor first forms a pointwise median across available EEG channels as the global signal, then computes the window-level common features @gramfort2013_mne. Its inherited band ranges are delta 1-4 Hz, theta 4-8 Hz, alpha 8-12 Hz, beta 13-30 Hz, and gamma 30-45 Hz. The configured PSD method is MNE `psd_array_multitaper` with adaptive tapers, full normalization, null bandwidth, and a 1-45 Hz range. If the multitaper function is unavailable, the code uses SciPy Welch: Hann window, `nperseg=min(window_samples,512)`, half overlap, constant detrending, and density scaling; for the 250 Hz, 8 s configuration this is 512 samples with 256-sample overlap. Band powers use trapezoidal integration over the configured band limits. These are distinct estimator realizations. The current writer records the realized method ID and fallback reason per window; the frozen cohort exports audited here predate that field, so their exact PSD branch is not recoverable from HDF5. The first seven components use the listed source values; transformed values are retained separately.

#figure(
  clean-table(
  columns: (auto, 1fr, 1fr),
  header: ([*Component*], [*Definition before anchor scaling*], [*Unit*]),
  inset: 4pt,
  [#raw("delta_theta_level")], [0.5 log10(P_delta / 1 μV²) + 0.5 log10(P_theta / 1 μV²)], [dimensionless log-power],
  [#raw("alpha_level")], [log10(P_alpha / 1 μV²)], [dimensionless log-power],
  [#raw("beta_alpha_ratio")], [P_beta / P_alpha; NaN if P_alpha is missing or ≤ 0], [dimensionless],
  [#raw("gamma_level")], [log10(P_gamma / 1 μV²)], [dimensionless log-power],
  [#raw("hjorth_mobility")], [sqrt(var(Δx) / var(x)); Δx = diff(x), no Δt division], [sample-difference ratio],
  [#raw("alpha_theta_ratio")], [P_alpha / P_theta; NaN if P_theta is missing or ≤ 0], [dimensionless],
  [#raw("permutation_entropy")], [eeg_permutation_entropy], [dimensionless],
  [#raw("alpha_centroid_hz")], [#raw("sum(f * median_channel_PSD[f]) / sum(median_channel_PSD[f]), f in 7-13 Hz")], [Hz],
  [#raw("alpha_envelope_cv")], [median over channels of population_SD(envelope) / mean(envelope)], [dimensionless],

  ),
  kind: table,
  caption: [Coordinate inputs, source rules, and raw units before frozen reference scaling.],
)

Power features are computed after EEG values are expressed in μV. Log-power is relative to the declared 1 μV² unit. Permutation entropy is normalized, with ordinal order 5 and delay 1 sample @bandtPompe2002. If the inherited extractor signals degraded fallback mode, the raw column remains observable but the v3.4 coordinate is unsupported. Non-positive bandpower is invalid for log10 inputs; the implementation does not replace it with an arbitrary floor.

The inherited Hjorth mobility implementation computes sqrt(var(diff(x))/var(x)) using unscaled first differences. It does not divide by the sampling interval, so it is a sampling-rate-dependent sample-difference ratio rather than a derivative in s⁻¹.

The first seven features are computed after a pointwise median over available average-referenced channels. This exact path is retained as the defined realization; it does not commute with channel-wise feature extraction followed by median aggregation. The summary is reference- and topography-dependent. No comparison against per-channel feature medians or a posterior-channel summary was run in this audit. The frozen derived exports retain per-channel candidates for the centroid and envelope summaries, but not raw channel waveforms or per-channel values for the first seven common features; source EDFs were unavailable. These alternatives therefore cannot be reconstructed from the audited files, and the manuscript does not claim they are interchangeable or that the pointwise median is optimal.

== S1.3.1 Alpha centroid

Power spectral density is calculated separately for every native EEG channel using Welch's averaged periodogram method @welch1967: 4 s segments, 50 percent overlap, a periodic Hann window, and constant detrending. For each frequency bin between 7 and 13 Hz, the channel median PSD is formed. The output is the power-weighted mean frequency within that band. Per-channel centroids and support counts are retained. A non-positive denominator is unsupported. The estimator is a frequency summary, not a marker detector or validated individual alpha frequency.

The centroid is not an alpha-peak prominence estimator. A finite centroid can be produced when the band spectrum has no distinct alpha peak, so the output alone does not establish a reliable individual alpha frequency. Peak-prominence and low-alpha-power sensitivity were not computed from this export and remain open robustness checks.

== S1.3.2 Alpha-envelope coefficient of variation

Each finite continuous source segment is bandpass filtered from 8 to 12 Hz with an order-4 zero-phase Butterworth filter. The analytic signal is obtained with the Hilbert transform and its magnitude is the envelope. For each channel, population standard deviation over time is divided by mean envelope; the across-channel median is emitted for the window. A non-positive mean is unsupported. A symmetric 10 s source-segment edge guard is applied. This metric is an estimator-defined envelope variability summary; the name does not establish physiological burst detection.

Both zero-phase filtering and the Hilbert transform are noncausal offline operations on the continuous segment, not estimators restricted to the nominal window alone. The 10 s edge guard limits boundary effects; it does not make the estimator causal or turn the coefficient of variation into a burst detector.

== S1.3.3 Spatial log-variance dispersion

The auxiliary candidate is 1.4826 times the median absolute deviation of log10 channel variance within the window. It retains the per-channel log variances and channel support. It is not part of the nine-coordinate order and is not included in coordinate scaling.

The 30-45 Hz gamma level is computable under the declared feature contract. Without independent muscle-artifact assessment, it is not necessarily a clean estimate of neural gamma activity.

= S1.4 Anchor fitting

Feature transforms are applied before feature reference scaling. For a given measurement, the fit uses only finite supported windows and a participant identity. Let participant i have n_i supported windows. Each of their windows has provisional weight 1/n_i; the resulting weights are normalized over the reference. This gives each participant equal total influence while preserving their within-participant measurement distribution.

The reference center is the weighted median. The reference scale is 1.4826 times the weighted median absolute deviation about that center. Feature-level centers and scales are fit first. Coordinates are assembled from the scaled features using the fixed input coefficients. A second set of medians and scales is then fit to each assembled coordinate. For a one-input coordinate, when support, observations, and participant weights match, robust median/MAD scaling at the coordinate stage reproduces the feature-scaled values up to floating-point error. The second stage is therefore an explicit uniform interface, not an additional source of information for one-to-one mappings. A focused synthetic identity test checks this implementation property. The multi-input delta/theta coordinate is assembled as 0.5 z_delta + 0.5 z_theta after its two source features have each been anchored; it is not the unanchored average of log powers. The manifest stores centers, scales, participant counts, window counts, and status for both layers.

The implementation does not silently repair an invalid fit. No finite values, fewer than the configured minimum number of participants, or a degenerate scale produce an unsupported reference field. The affected anchored measurement is invalid; it is not zero-filled and is not normalized using the target recording.

= S1.5 Registered ds005385 references

The eyes-closed preset is ds005385_ec_v1 with anchor ID ds005385_ses1_pretask_EyesClosed_20261111_v1. Its JSON SHA-256 is 79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab and its embedded anchor hash is ecbc9b1ffb5cf64fd0c094280f3d7f6c6543690a518d80752a4fb6be1e134fd6.

The eyes-open preset is ds005385_eo_v1 with anchor ID ds005385_ses1_pretask_EyesOpen_20261111_v1. Its JSON SHA-256 is f3bb94310778e91b046fbf77f0540ce860af6a65d4607e1e8593c01f800705b1 and its embedded anchor hash is f571d278f1de9863dc1d71ef974c2728427560e8b4b67363e56fe2c2ad646f1f.

Each reference used the same 608 participants whose recordings received that condition's anchor. This is transductive reference fitting. The two anchors define separate coordinate frames. Either preset can technically be selected for another target condition, but a target-label match does not establish comparability, validity, or a scientific reason to transfer it.

= S1.6 Output paths and interpretation

In a combined HDF5 file, the v3.4 measurements reside under /measurements/v3_4/. Important datasets include raw_features/values, raw_coordinates/values, feature_scaled_values/values, coordinates/values, coordinate names and order, window_start_sec, window_end_sec, coordinate_validity_mask, and complete_9d_mask. Support, QC, per-channel candidates, and provenance are stored in their own groups. Scalar qualified and status fields are siblings of these groups within /measurements/v3_4/.

The export records the ordered channel labels, channel count, channel-set hash, reference and reference-set hash, source and target condition, anchor ID and file hash, compatibility mode, estimator and implementation hashes, failure reasons, and QC state. A sidecar is a valid measurement-only output and does not synthesize arrays from other measurement families.

A finite value, a valid mask, a complete coordinate mask, and qualification are different properties. A value can be computable but fail an application check; a window can contain some valid coordinates but not all nine; and even nine finite coordinates can remain unqualified. Downstream analysis must state its own inclusion and comparison rules.

= S1.7 Core HDF5 schema

The table records the core v3.4 datasets and their shapes, types, units, and mask meanings. `W` is the number of windows in a recording, `F=10` the source-feature count, `D=9` the coordinate count, and `C` the stored channel count. In the selected cohort `C=64`; shapes are recording-specific. The read-only audit checked the listed core paths across all 1,216 combined exports. The two PSD identity paths are now emitted by the writer but absent from these frozen exports, so their values read as `not_recorded` for this cohort.

#figure(
  clean-table(
    columns: (1.65fr, 0.8fr, 2.2fr),
    header: ([*Dataset path*], [*Shape; dtype*], [*Unit and interpretation*]),
    inset: 3pt,
    [`/measurements/v3_4/raw_features/{values,support}`], [`[W,F]`; `float64` / `uint8`], [Mixed source-feature units named in the feature table; support is per-feature computability, not artifact QC.],
    [`/measurements/v3_4/raw_coordinates/{values,support}`], [`[W,D]`; `float64` / `bool`], [Unanchored transformed coordinates; support means all declared raw inputs are computable.],
    [`/measurements/v3_4/feature_scaled_values/values`], [`[W,F]`; `float64`], [Dimensionless feature-reference robust-z values.],
    [`/measurements/v3_4/coordinate_pre_reference_values`], [`[W,D]`; `float64`], [Dimensionless assembled coordinates before coordinate-reference scaling.],
    [`/measurements/v3_4/coordinates/values`], [`[W,D]`; `float64`], [Final coordinate-anchor values; dimensionless.],
    [`/measurements/v3_4/coordinate_validity_mask`], [`[W,D]`; `uint8`], [1 indicates a finite, successfully anchored coordinate; it does not indicate clean EEG.],
    [`/measurements/v3_4/complete_9d_mask`], [`[W]`; `uint8`], [1 only when all nine coordinate-validity entries are 1 for that window.],
    [`/measurements/v3_4/window_start_sec`; `/measurements/v3_4/window_end_sec`], [`[W]`; `float64`], [Window start/end times in seconds on the realized source time base.],
    [`/measurements/v3_4/support/qc/qc_ok_eeg`], [`[W]`; `int8`], [-1 means unassessed; in this cohort it is -1 for every window.],
    [`/measurements/v3_4/support/channel_labels`], [`[C]`; `string`], [Actual ordered channel labels for the recording.],
    [`/measurements/v3_4/per_channel_candidates/{alpha_centroid_hz,alpha_envelope_cv}`], [`[W,C]`; `float64`], [Hz for centroid; dimensionless for envelope CV; unsupported entries remain NaN.],
    [`/measurements/v3_4/support/eeg_psd_method_id`], [`[W]`; `string`], [Realized PSD backend; absent from the frozen exports audited here.],
    [`/measurements/v3_4/support/eeg_psd_fallback_reason`], [`[W]`; `string`], [Configured/fallback selection reason; absent from the frozen exports audited here.],
    [`/measurements/v3_4/qualified`], [`scalar`; `bool`], [Scientific qualification state; false for the selected cohort exports.],
  ),
  kind: table,
  caption: [Core v3.4 HDF5 schema and per-window support semantics.],
)

= S1.8 Cohort summary inventory

Feature extraction covered 1,216 recordings and produced 55,183 windows: 27,651 eyes-closed and 27,532 eyes-open. Each condition's anchor was computed and applied for 608 recordings with no anchor identity failures. The reference run is descriptive and transductive.

The broader HDF5 inventory contains other measurement groups alongside v3.4 outputs. Its report separates rows by full path and realization and enumerates mean, population standard deviation, median, MAD, scaled MAD, first and third quartiles, IQR, minimum, maximum, skewness, excess kurtosis, and support counts. The inventory has 2,433 numeric statistics rows and 2,435 measurement inventory rows; two additional inventory entries indicate that a v3.4 Jacobian is unavailable. The report also contains 600,704 support/QC rows. These counts describe the report's inventory, not independent observations or inferential sample sizes.

A separate v3.4-only audit read 1,216 combined HDF5 recording exports. All eight coordinates other than `alpha_envelope_cv` were valid in every selected window: 27,651 eyes-closed and 27,532 eyes-open windows. `alpha_envelope_cv` was valid in 24,361/27,651 eyes-closed windows and 24,215/27,532 eyes-open windows; the remaining 3,290 and 3,317 windows, respectively, were within the configured segment-edge guard. Every participant had at least one complete-nine-coordinate window. All 1,216 anchor applications reported `computed`; all exports remained `qualified=false`, and `qc_ok_eeg=-1` for all 55,183 windows. The audit does not establish artifact-free EEG.

For dependency summaries, each participant contributed one median per coordinate and condition under available-coordinate windows and matched complete-nine windows. Both policies retained the same 608 participants in both condition strata. The Spearman matrices, eigenvalue tables, and policy comparisons are saved in `project/mnps2_audit/outputs/20261116_mnps_v3_4_measurement_quality_audit/`. Spearman correlation-matrix spectral-entropy effective ranks were 4.3649 (available) and 4.3655 (matched) for eyes closed, and 4.7141 and 4.7185 for eyes open. Mean absolute correlation differences were 0.0024 and 0.0025; maximum absolute differences were 0.0091 and 0.0101. The earlier leave-one-coordinate-out ranges, calculated for available-coordinate medians, were 3.63–4.67 and 4.01–4.99; they are not feature-importance tests. Effective rank is computed from the Spearman correlation matrix, not a covariance matrix, and is not an intrinsic-dimension estimate.

The core schema paths and shape/type contracts were present in all 1,216 files. Runtime PSD backend identity was not recorded in these cached exports, despite recorded implementation hashes. Source matching used unique manifest basenames for all 1,216 recordings where full relative paths were not preserved. Raw EDF content hashes and header calibration checks were not computed; the derived-export directory contained no source EDFs. The dataset README warns that some EDF physical min/max specifications may be invalid @openneuro_ds005385_readme. The cohort-selection report did not establish artifact QC. No inferential condition comparison was performed. Historical output-path checks are summarized in Supplementary Changelog S6.

The separate ds004148 session-repeatability analysis selected 60 participants and all 360 resting EC/EO recordings across sessions 1–3 from BIDS metadata before inspecting features or coordinates @openneuro_ds004148_v101. Each recording used the same frozen ds005385 EC anchor. Session-1 features and QC were reused; sessions 2–3 were re-extracted. The locked eligibility floor was 30 QC-eligible windows and 120 unique seconds per participant, session, condition, and coordinate. QC matched 26,640 windows. This support floor passed for 2,858/3,240 coordinate summaries and for 306/360 complete-nine summaries. Across 54 EC/EO coordinate-level session-comparison rows, ICC(A,1) ranged 0.041–0.954 with 43–52 participants per row. Across 27 EO−EC rows, it ranged 0.249–0.864 with 36–42 participants per row. Alpha-centroid EC/EO level estimates ranged 0.762–0.954; alpha-envelope-CV EO−EC estimates ranged 0.793–0.864. The full table gives each session pair, condition, coordinate, support count, bias, absolute differences, agreement limits, and rank correlation. These estimates are descriptive and specific to the support and compatibility rules; eligibility is not proof of artifact absence.

= S1.9 Reproduction and verification boundary

Example commands from the repository root are:

#raw("$config = \"mndm/config/v.3.4/config_ingest_ds005385_baseline_eo_ec.yml\"\npython -m mndm.cli features --config $config --dataset ds005385\npython -m mndm.cli summarize --config $config --dataset ds005385\npython mndm/src/mndm/tools/mnps_v3_4_measurement_quality_audit.py --h5-root project/mnps2_audit/outputs/20261111_mnps_v3_4_ds005385_baseline_cohort/processed/ds005385/neuralmanifolddynamics_ds005385_20261112_004711 --out project/mnps2_audit/outputs/20261116_mnps_v3_4_measurement_quality_audit --figure-out project/articles/NeuralManifoldDynamics/figures/mnps_v3_4_dependency_heatmap.svg", block: true, lang: "powershell")

The ds005385 feature extraction and reference fitting were not rerun while writing this manuscript; the v3.4-only audit reads frozen outputs. For ds004148, session-1 HDF5/QC outputs were reused and sessions 2–3 were re-extracted. A single frozen ds005385 EC anchor was applied to every target condition without refitting. The full-cohort analysis checks all 360 HDF5 files, matches QC to all 26,640 windows, and calculates participant-level session comparisons only after the predefined support and realization-compatibility checks. The durable package at `mndm/docs/ds004148_session_repeatability_fullcohort/` contains the exact configuration, scripts, manifests, summary tables, anchor identity, file hashes, and suppression smoke verification. Eligibility remains an operational screen and does not establish artifact-free EEG or biological stability.

#bibliography("NeuralManifoldReferences.bib", style: "apa")
