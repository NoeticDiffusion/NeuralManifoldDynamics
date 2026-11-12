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

The ds005385 reference configuration uses session-1 pre-task eyes-open and eyes-closed data. The realized surface is 64 EEG channels with average reference, a 50 Hz notch, a 1-45 Hz bandpass, and resampling to 250 Hz. Source labels and their order are preserved. The reference cohort and target recordings must declare the channel set, reference, frequency, filters, sampling rate, time grid, and estimator realization.

Windows have an 8 s duration and a 4 s step. The exported start and end times define the effective grid. Windows do not cross source-file boundaries or detected discontinuities. The alpha-envelope candidate is calculated from continuous finite segments and requires 10 s of source support on each side of the window. A window inside the edge guard or spanning a segment boundary is unsupported for that candidate. This candidate-specific restriction does not erase other computable coordinates.

The reference cohort used metadata and required sidecars for selection. It did not use signal values to choose recordings. A missing raw-file hash means that selection-to-HDF5 identity is not a byte-level signal verification. Artifact QC was not established by the selection.

= S1.3 Feature definitions

The common EEG extractor first forms a pointwise median across available EEG channels as the global signal, then computes the window-level common features @gramfort2013_mne. Its inherited band ranges are delta 1-4 Hz, theta 4-8 Hz, alpha 8-12 Hz, beta 13-30 Hz, and gamma 30-45 Hz. The configured PSD method is multitaper, with a Welch fallback only if the multitaper routine is unavailable. The first seven components use the listed source values; transformed values are retained separately.

#figure(
  clean-table(
  columns: (auto, 1fr, 1fr),
  header: ([*Component*], [*Definition before anchor scaling*], [*Unit*]),
  inset: 4pt,
  [#raw("delta_theta_level")], [0.5 log10(eeg_delta) + 0.5 log10(eeg_theta)], [log10 microvolt-squared],
  [#raw("alpha_level")], [log10(eeg_alpha)], [log10 microvolt-squared],
  [#raw("beta_alpha_ratio")], [eeg_beta_alpha], [dimensionless],
  [#raw("gamma_level")], [log10(eeg_gamma)], [log10 microvolt-squared],
  [#raw("hjorth_mobility")], [eeg_hjorth_mobility], [dimensionless per sample],
  [#raw("alpha_theta_ratio")], [eeg_alpha_theta], [dimensionless],
  [#raw("permutation_entropy")], [eeg_permutation_entropy], [dimensionless],
  [#raw("alpha_centroid_hz")], [#raw("sum(f * median_channel_PSD[f]) / sum(median_channel_PSD[f]), f in 7-13 Hz")], [Hz],
  [#raw("alpha_envelope_cv")], [median over channels of population_SD(envelope) / mean(envelope)], [dimensionless],

  ),
  kind: table,
  caption: [Coordinate inputs, source rules, and raw units before frozen reference scaling.],
)

Permutation entropy is normalized, with ordinal order 5 and delay 1 sample @bandtPompe2002. If the inherited extractor signals degraded fallback mode, the raw column remains observable but the v3.4 coordinate is unsupported. Non-positive bandpower is invalid for log10 inputs; the implementation does not replace it with an arbitrary floor.

== S1.3.1 Alpha centroid

Power spectral density is calculated separately for every native EEG channel using Welch's averaged periodogram method @welch1967: 4 s segments, 50 percent overlap, a periodic Hann window, and constant detrending. For each frequency bin between 7 and 13 Hz, the channel median PSD is formed. The output is the power-weighted mean frequency within that band. Per-channel centroids and support counts are retained. A non-positive denominator is unsupported. The estimator is a frequency summary, not a marker detector or validated individual alpha frequency.

== S1.3.2 Alpha-envelope coefficient of variation

Each finite continuous source segment is bandpass filtered from 8 to 12 Hz with an order-4 zero-phase Butterworth filter. The analytic signal is obtained with the Hilbert transform and its magnitude is the envelope. For each channel, population standard deviation over time is divided by mean envelope; the across-channel median is emitted for the window. A non-positive mean is unsupported. A symmetric 10 s source-segment edge guard is applied. This metric is an estimator-defined envelope variability summary; the name does not establish physiological burst detection.

== S1.3.3 Spatial log-variance dispersion

The auxiliary candidate is 1.4826 times the median absolute deviation of log10 channel variance within the window. It retains the per-channel log variances and channel support. It is not part of the nine-coordinate order and is not included in coordinate scaling.

= S1.4 Anchor fitting

Feature transforms are applied before feature reference scaling. For a given measurement, the fit uses only finite supported windows and a participant identity. Let participant i have n_i supported windows. Each of their windows has provisional weight 1/n_i; the resulting weights are normalized over the reference. This gives each participant equal total influence while preserving their within-participant measurement distribution.

The reference center is the weighted median. The reference scale is 1.4826 times the weighted median absolute deviation about that center. Feature-level centers and scales are fit first. Coordinates are assembled from the scaled features using the fixed input coefficients. A second independent set of medians and scales is then fit to each assembled coordinate. The manifest stores centers, scales, participant counts, window counts, and status for both layers.

The implementation does not silently repair an invalid fit. No finite values, fewer than the configured minimum number of participants, or a degenerate scale produce an unsupported reference field. The affected anchored measurement is invalid; it is not zero-filled and is not normalized using the target recording.

= S1.5 Registered ds005385 references

The eyes-closed preset is ds005385_ec_v1 with anchor ID ds005385_ses1_pretask_EyesClosed_20261111_v1. Its JSON SHA-256 is 79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab and its embedded anchor hash is ecbc9b1ffb5cf64fd0c094280f3d7f6c6543690a518d80752a4fb6be1e134fd6.

The eyes-open preset is ds005385_eo_v1 with anchor ID ds005385_ses1_pretask_EyesOpen_20261111_v1. Its JSON SHA-256 is f3bb94310778e91b046fbf77f0540ce860af6a65d4607e1e8593c01f800705b1 and its embedded anchor hash is f571d278f1de9863dc1d71ef974c2728427560e8b4b67363e56fe2c2ad646f1f.

Each reference used the same 608 participants whose recordings received that condition's anchor. This is transductive reference fitting. The two anchors define separate coordinate frames. Either preset can technically be selected for another target condition, but a target-label match does not establish comparability, validity, or a scientific reason to transfer it.

= S1.6 Output paths and interpretation

In a combined HDF5 file, the v3.4 measurements reside under /measurements/v3_4/. Important datasets include raw_features/values, raw_coordinates/values, feature_scaled_values/values, coordinates/values, coordinate names and order, window_start_sec, window_end_sec, coordinate_validity_mask, and complete_9d_mask. Support, QC, per-channel candidates, and provenance are stored in their own groups. Scalar qualified and status fields are siblings of these groups within /measurements/v3_4/.

The export records the ordered channel labels, channel count, channel-set hash, reference and reference-set hash, source and target condition, anchor ID and file hash, compatibility mode, estimator and implementation hashes, failure reasons, and QC state. A sidecar is a valid measurement-only output and does not synthesize arrays from other measurement families.

A finite value, a valid mask, a complete coordinate mask, and qualification are different properties. A value can be computable but fail an application check; a window can contain some valid coordinates but not all nine; and even nine finite coordinates can remain unqualified. Downstream analysis must state its own inclusion and comparison rules.

= S1.7 Cohort summary inventory

Feature extraction covered 1,216 recordings and produced 55,183 windows: 27,651 eyes-closed and 27,532 eyes-open. Each condition's anchor was computed and applied for 608 recordings with no anchor identity failures. The reference run is descriptive and transductive.

The broader HDF5 inventory contains other measurement groups alongside v3.4 outputs. Its report separates rows by full path and realization and enumerates mean, population standard deviation, median, MAD, scaled MAD, first and third quartiles, IQR, minimum, maximum, skewness, excess kurtosis, and support counts. The inventory has 2,433 numeric statistics rows and 2,435 measurement inventory rows; two additional inventory entries indicate that a v3.4 Jacobian is unavailable. The report also contains 600,704 support/QC rows. These counts describe the report's inventory, not independent observations or inferential sample sizes.

Source matching used unique manifest basenames for all 1,216 recordings where full relative paths were not preserved. Raw EDF content hashes were not computed. The cohort-selection report did not establish artifact QC. No inferential condition comparison was performed.

= S1.8 Reproduction and verification boundary

Example commands from the repository root are:

#raw("$config = \"mndm/config/v.3.4/config_ingest_ds005385_baseline_eo_ec.yml\"\npython -m mndm.cli features --config $config --dataset ds005385\npython -m mndm.cli summarize --config $config --dataset ds005385", block: true, lang: "powershell")

These commands describe the configured reference workflow; they were not rerun while writing this manuscript. A fresh bounded end-to-end pilot using the explicit external-anchor CLI path remains necessary to verify that orchestrator route and inspect the resulting HDF5 and sidecar provenance. Previously reported focused tests are not repeated here and do not establish scientific validity.

= References

#bibliography("NeuralManifoldReferences.bib", style: "apa")
