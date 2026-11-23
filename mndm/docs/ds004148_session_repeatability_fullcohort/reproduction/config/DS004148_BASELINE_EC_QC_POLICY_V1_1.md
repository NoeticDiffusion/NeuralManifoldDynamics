# ds004148 MNPS 3.4 baseline EC/EO QC and support policy v1.1

**Frozen:** 2026-11-16, before the full-cohort feature/coordinate analysis.  
**Status:** predeclared exploratory-analysis policy; not a validated EEG
artifact classifier or a qualification rule for clinical or biological use.

Version 1.1 is a technical clarification of the frozen v1 policy after
independent code review. It adds fail-closed handling for uncomputable QC and
checks the v3.4 source-feature unit contract before cohort aggregation. It
does not change the muscle threshold, window support, minimum support, anchor,
or any feature definition. The v1 file is retained as the original record.

## Question and cohort

The analysis applies the already-frozen
`ds005385_ses1_pretask_EyesClosed_20261111_v1` anchor (manifest SHA-256
`79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab`) without
refitting to target recordings.

The baseline set is session 1 resting EO and EC in OpenNeuro ds004148 v1.0.1.
The dataset README describes 60 participants. A metadata-only inventory found
one session-1 `eyesclosed` and one `eyesopen` BrainVision recording for each
`sub-01` through `sub-60`. Include all 60 pairs before inspecting signal,
feature, coordinate, or outcome values. Exclude sessions 2 and 3 and the
mathematic, memory, and music tasks. Preserve a selection manifest containing
the 60 IDs, the two file paths per participant, and the BIDS metadata checks.
The metadata gate requires exactly one canonical EO file and one canonical EC
file with all required BrainVision/BIDS components for each expected
participant, declared `uV` units and 500 Hz samples in all 120 recordings, and
no unexpected participant directories. If that gate fails, stop before feature
extraction and resolve the inventory without replacing participants. The run
command must explicitly select the frozen 60 subject IDs. After feature
extraction/indexing, compare `file_index.csv` paths and the distinct
source-file identities in `features.csv` with the selection manifest; stop
before summarize if either differs.

Use the pilot's fixed measurement surface for all recordings: source 500 Hz,
notch 50 Hz, 1–45 Hz EEG bandpass, average reference, realized 250 Hz, native
ordered EEG channels, and the existing 8-second windows at 4-second steps.
Do not apply artifact correction or the channel-drop heuristic. Keep every
computable measurement in the NMD export; apply this protocol as a separate
analysis mask so unsupported/rejected windows and their reasons remain
auditable.

## Fixed artifact screen

Run one QC-only copy from the original 500 Hz signal, separately for each
recording. Do not alter the feature signal. Reuse the pilot implementation and
lock its parameters:

- average-reference the QC copy and notch 50, 100, and 150 Hz;
- bandpass 110–140 Hz, compute the Hilbert-envelope magnitude, z-score each
  channel's envelope over the recording, sum across EEG channels divided by
  the square root of the channel count, then low-pass the score at 4 Hz;
- mark each sample whose resulting muscle score is strictly greater than 4;
- flag an 8-second feature window if it overlaps any marked sample, and
  exclude that window from the primary participant medians.
- Treat the detector as unassessed for a recording if any source sample or
  muscle-score sample is nonfinite, or if every channel has an exactly flat
  envelope. A detector that is unassessed does not classify windows as clean;
  its recording's windows are ineligible for the primary summaries. Preserve
  this status and reason explicitly. Record
  `muscle_detector_active_channel_count` from the number of channel envelopes
  with nonzero temporal standard deviation. This estimator-derived count is
  distinct from the raw-signal exact-flatline count and is the count used to
  validate detector availability in the summary.

This is the pilot's fixed MNE-inspired muscle-score/sample-overlap screen, not
an exact call to MNE's annotation wrapper. The existing pilot script applies
the documented score recipe directly because its MNE 1.6 wrapper did not work
with the installed `Raw` API; it does not merge short unflagged gaps using
`min_length_good`. MNE describes the resulting intervals as *likely* muscle
artifact and notes that threshold choice is data-dependent. Here 4 is locked
for reproducibility, not tuned to these three participants or to coordinate
values. Any-overlap rejection follows the documented MNE epoch policy for a
window partially overlapping a `BAD` interval. See the [MNE 1.6.1 muscle
annotation documentation](https://mne.tools/1.6/auto_examples/preprocessing/muscle_detection.html)
and [bad-span rejection documentation](https://mne.tools/1.6/auto_tutorials/preprocessing/20_rejecting_bad_data.html).

Record, but do not use as automatic exclusions: finite-sample fraction,
exact-flatline channel count, channel-wise peak-to-peak and absolute-amplitude
summaries, and the pre-notch 49–51 Hz / 1–45 Hz power ratio. No verified ADC
rails are available, so these data do not establish clipping from voltage
magnitude alone. The source has no EOG or ECG channels; eye, cardiac, electrode
pop, and other non-muscle artifacts are not comprehensively classified by this
screen. Report these limits and retain raw QC descriptors.

Do not set HDF5 `qc_ok_eeg=1` from this external QC table. The NMD artifact-QC
state remains unassessed (`-1`) because no artifact-correction/QC method is
applied in preprocessing. Store the independent decision as a named
analysis-level window mask with detector version, thresholds, time support,
and exclusion reasons.

## Fixed support and paired summaries

For each participant, task, and coordinate, the eligible window set is the
intersection of:

1. estimator support/validity in the v3.4 HDF5 (no filling or imputation);
2. finite coordinate values and fully finite source samples for that window;
3. a computable detector for the recording; and
4. no overlap with a sample marked by the fixed muscle screen.

A participant contributes a paired EO−EC difference for a coordinate only
when **both** session-1 conditions have at least **30 eligible windows** for
that coordinate and their interval union covers at least **120 distinct
seconds**. Count actual HDF5 start/end times; overlapping 8-second windows do
not count twice toward unique seconds. The 30-window / 120-second rule is an
operational floor for this descriptive median, not a validated reliability or
power threshold. If either condition fails, keep any supported condition
summary and report the paired difference as unavailable with its reason.

Report coordinate-specific support counts. A complete-9D paired summary also
requires at least 30 eligible windows with all nine coordinates valid and at
least 120 seconds of unique support in each condition. The existing
10-second segment-edge guard for `alpha_envelope_cv` remains in force.

For each eligible participant and coordinate, calculate the EC median and EO
median over that condition's eligible windows, then calculate the participant
difference `EO median − EC median`. Report the cohort median, IQR, and
unscaled MAD (median absolute deviation from the cohort median), plus
minimum/maximum and paired participant count from these participant-level
differences. Do not pool windows across people, treat overlapping windows as
independent participants, or calculate a p-value in this descriptive baseline
step. Preserve individual participant rows, eligible/excluded window counts,
and reasons. Report raw features/coordinates alongside the anchored values;
anchored coordinates are dimensionless reference-relative scores.

Within-participant EO/EC pairing additionally requires the same channel-label
set/order, reference, sampling and preprocessing realization, estimator
manifest, and time-grid definition in the two files. A mismatch leaves
computable measurements available but suppresses that participant's paired
coordinate difference with a stated reason. Do not impose a global Core19
channel-count floor.

For each coordinate, validate the ordered source-feature unit signature
against the MNPS 3.4 contract in every HDF5 file:

| Coordinate | Expected source feature units |
| --- | --- |
| `delta_theta_level` | `eeg_delta=microvolt^2`; `eeg_theta=microvolt^2` |
| `alpha_level` | `eeg_alpha=microvolt^2` |
| `beta_alpha_ratio` | `eeg_beta_alpha=dimensionless_ratio` |
| `gamma_level` | `eeg_gamma=microvolt^2` |
| `hjorth_mobility` | `eeg_hjorth_mobility=dimensionless_per_sample` |
| `alpha_theta_ratio` | `eeg_alpha_theta=dimensionless_ratio` |
| `permutation_entropy` | `eeg_permutation_entropy=dimensionless` |
| `alpha_centroid_hz` | `mnps_v3_4_alpha_centroid_hz=Hz` |
| `alpha_envelope_cv` | `mnps_v3_4_alpha_envelope_cv=dimensionless` |

Keep a participant EO/EC difference only when both recordings use that
coordinate's expected source units. Pool participant differences for a
cohort statistic only when all 120 recordings match the expected signature;
otherwise retain valid participant pairs and report the cohort aggregation as
unavailable for that coordinate.

There is no cohort-level minimum participant count for this descriptive
baseline. Report the actual paired participant count for each coordinate. If
no participant meets the paired support rule for a coordinate, report that
coordinate as unavailable and do not calculate its cohort EC/EO difference.
Other coordinates may still be reported if they have eligible pairs. A
complete-9D summary is unavailable if no participant meets its complete-vector
support rule. Stop the full cohort summary only if the metadata gate fails or
no coordinate has even one eligible paired participant. Do not relax the
muscle threshold, overlap rule, minimum window count, or unique-time support
after examining cohort coordinates. A policy change requires a new version
and rationale before inspecting the affected contrast.
