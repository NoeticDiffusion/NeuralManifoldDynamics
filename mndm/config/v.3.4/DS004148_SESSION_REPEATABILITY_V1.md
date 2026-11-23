# ds004148 MNPS 3.4 session-repeatability protocol v1

**Frozen:** 2026-11-18, before the session-2/session-3 coordinate summaries.
**Status:** descriptive engineering pilot; not a measurement qualification or
biological reliability claim.

## Question and selection

Estimate whether the same participant's EC and EO MNPS 3.4 coordinate levels,
and the participant-level EO-minus-EC differences, agree across ds004148
sessions 1, 2, and 3 when every recording uses the frozen external
`ds005385_ses1_pretask_EyesClosed_20261111_v1` anchor
(SHA-256 `79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab`).
No ds004148 session is used to fit or update the anchor.

Select sub-01 through sub-60 and the canonical `eyesclosed` and `eyesopen`
recordings in each of the three sessions using BIDS paths and metadata only.
The selection is frozen before inspecting signal, feature, or coordinate
values. If any expected recording is incomplete or ambiguous, stop and report
the metadata gate; do not replace participants.

Reuse the completed session-1 features and QC assessment from
`20261116_ds004148_v3_4_frozen_ec_baseline`. Process sessions 2 and 3 with the
same v3.4 configuration, anchor, 500-to-250 Hz preprocessing, 50 Hz notch,
1–45 Hz bandpass, average reference, native channel realization, and 8-second
windows at 4-second steps. Write all new outputs under the separate
`20261118_ds004148_v3_4_session_repeatability` root.

## Fixed QC and support

Use the frozen v1.1 source-level QC screen unchanged for sessions 2 and 3,
and reuse its completed session-1 results. The screen runs on a separate QC
copy and does not modify feature signals. A window is QC-eligible only if its
source and detector are finite, the detector is valid, and no sample in the
window exceeds the locked muscle score threshold of 4. Do not infer artifact
absence from a passing window.

Apply the same support floor as policy v1.1 separately to each coordinate,
condition, and session: at least 30 eligible windows and at least 120 unique
supported seconds. An EO-minus-EC value also requires both conditions to pass
and the EC/EO measurements within that session to share a compatible
realization. Keep computable values and their support visible when a
repeatability comparison is unavailable.

Before comparing sessions, require the same frozen anchor ID/hash, coordinate
order and source units, channel realization, reference, preprocessing and
estimator identities, and 8-second/4-second time-grid definition. A mismatch
suppresses that comparison with a reason; it does not erase the underlying
measurement.

## Summaries fixed in advance

For each coordinate and condition (EC, EO), calculate one median per
participant/session from eligible windows. For each session, calculate the
participant EO-minus-EC difference from the two condition medians. Report
comparisons for session pairs 1–2, 1–3, and 2–3, separately for condition
levels and EO-minus-EC differences. Use only participants meeting the support
and compatibility requirements for both sessions in that comparison; report
the resulting n for every row.

For absolute agreement, report ICC(A,1), mean session-pair bias, sample SD of
the paired differences, 95% Bland–Altman limits (bias ± 1.96 SD), and median
absolute difference. Also report the Spearman correlation as a rank-order
description, explicitly separate from agreement. Do not calculate p-values or
choose a coordinate based on observed repeatability. Report complete-9D
support separately; coordinate-specific summaries do not require all nine
coordinates to be valid.

All anchored coordinates, including `alpha_centroid_hz`, are dimensionless
reference-relative scores. These summaries assess this fixed measurement
procedure in this dataset; they do not establish biological stability,
cross-session equivalence, or clinical qualification.
