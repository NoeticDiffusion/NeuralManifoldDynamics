MNPS v3.2: SL-007 targeted protocols
====================================

Status
------

Frozen outcome-blind protocols implementing the SL-007 authorization. These
bounded analyses reuse the twelve selected ANPHY-Sleep sources from
``VECTOR-FEASIBILITY-002``. They do not tune measurement parameters against a
condition difference, fit a predictive model, qualify a measurement, or make a
biological claim. All 180 nominal rows are kept for NIE and OI; unsupported
source windows and metric-local failures remain explicit null rows with
reasons. Summaries use only windows where the particular metric is valid.

The frozen SL-006 result is
``outputs/VECTOR-FEASIBILITY-002/run_20261013T_sl006_vectorfeasibility002_final``.
Its exact selected participant order, source hashes, 250 Hz grid, 20 s windows,
fixed 83-channel average reference, preprocessing and window support are
inherited. Every reloaded OI input must match its recorded SL-006
``input_matrix_sha256_little_endian_float64`` before the new analysis includes
it. The twelve recordings contribute 168 supported windows and 12 unsupported
first windows. Adjacent dependencies overlap; window rows are not independent
observations.

NIE-INCREMENTAL-001
-------------------

Question and frozen estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Does weighted global efficiency vary with edge placement after the exact
multiset of absolute Pearson weights, and therefore mean absolute connection
strength, is held fixed? The frozen primary comparison is original-window
``NIE(W)`` and mean absolute upper-triangle weight ``mean|r|``. The secondary
quantity is ``Delta_path = NIE(W) - mean|r|``. No NIE edge threshold, clipping,
re-referencing, channel removal, or estimator change is permitted.

Graph-position control
~~~~~~~~~~~~~~~~~~~~~~

For every one of the 168 supported SL-006 windows, use its stored symmetric
83-node absolute-Pearson weight matrix, excluding the diagonal. Generate 49
independent deterministic controls. Each control permutes the full upper-
triangle weight vector across the same 3,403 undirected graph positions and
copies each assigned weight to the mirrored position. This preserves the exact
multiset, including zero weights, and hence preserves mean absolute strength.
It does not preserve sensor geometry or represent a realizable EEG correlation
matrix. Recompute NIE with the unchanged inverse-weight global-efficiency
kernel. A control seed is the first 128 bits of SHA-256 over the UTF-8 string
``NIE-INCREMENTAL-001|subject_id|window_id|permutation_index`` and uses NumPy
PCG64. Report the original ``Delta_path`` relative to the control median and
the control 2.5th–97.5th percentile interval. Also report, descriptively, the
fraction of control increments no greater than the observed increment; this is
not called a p-value and is not used as a qualification rule.

Summaries use one median per participant across valid windows, followed by
descriptive cross-participant Pearson correlations (n=12). Report window-level
values and participant-level medians/IQRs. Do not pool windows as independent
observations. No null-based acceptance threshold or claim of predictive,
biological, or population-level incremental value is authorized.

OI-TEMPORAL-SPECIFICITY-001
--------------------------

Question and input
~~~~~~~~~~~~~~~~~~

Does unchanged order-5, delay-1 normalized permutation entropy (OI) show a
pattern not summarized by conventional spectral power, normalized spectral
entropy, or fixed-lag autocorrelation, and how does it compare with
amplitude-adjusted Fourier surrogates? Reconstruct each supported 20 s,
250 Hz, fixed-83 average-referenced ANPHY window using its frozen SL-006
overlay and audited source loader. Exclude no supported windows. Before
computing any comparisons, require each reconstructed float64 matrix hash to
equal the corresponding SL-006 hash. If any mismatch occurs, retain the
window as a failed input audit and do not analyze its metrics.

Conventional comparators
~~~~~~~~~~~~~~~~~~~~~~~~

Use the SL-006 Welch convention: Hann window, 1,000 samples per segment,
500-sample overlap, 1,000-point FFT, constant detrending, density scaling,
250 Hz, and the fixed ``[1,4)``, ``[4,8)``, ``[8,13)``, ``[13,30)`` and
``[30,45)`` Hz bands. Integrate each band's power and report the median over
the 83 channels. Spectral entropy is Shannon entropy over the normalized
Welch density bins in ``[1,45)`` Hz, divided by the natural log of the number
of included bins, then medianed over channels. Autocorrelation uses the full
channel mean and energy denominator, with the prespecified lags 1, 25 and 125
samples (4, 100 and 500 ms at 250 Hz); report the channel median at each lag.
Do not select bands or lags based on observed association with OI.

IAAFT surrogate control
~~~~~~~~~~~~~~~~~~~~~~~

Generate two independent iterative amplitude-adjusted Fourier-transform
(IAAFT) surrogates for each channel and supported window, independently by
channel. Initialize each surrogate by permuting that channel's samples. Run
exactly 50 iterations: replace Fourier magnitudes by those of the original
channel while retaining current phases, inverse transform, then rank-order the
result and assign the original sorted sample values. Use stable mergesort for
rank ties. Use PCG64 seeded from the first 128 bits of SHA-256 over
``OI-TEMPORAL-SPECIFICITY-001|subject_id|window_id|channel_label|surrogate_index``.
The final surrogate has the exact empirical amplitude multiset; its achieved
power-spectrum relative L2 error is recorded because the final rank step means
the spectrum is approximate. This is a univariate control and does not
preserve cross-channel dependence. Apply the unchanged OI kernel and its
median-over-channels aggregation to each surrogate replicate.

Report per-window observed OI, surrogate OI values, their difference, spectral
error, conventional comparators, and audit hashes. Summarize first within
participant, then calculate descriptive Pearson correlations across the
twelve participant medians. No p-values, parameter search, inferential
threshold, or statement of nonlinear-dynamics specificity is authorized.
If OI is close to its IAAFT values, narrow its interpretation to the recorded
conventional statistics; this does not invalidate its computability.

COMA-VECTOR-PREFLIGHT-001
------------------------

Purpose and state
~~~~~~~~~~~~~~~~~

Separate outcome-blind data and measurement feasibility preparation for
PhysioNet I-CARE v2.1. No CPC prediction, CPC group comparison, cohort
qualification, or outcome association is part of this preflight. The current
workspace has I-CARE metadata and ingest configurations but no located raw
WFDB source files. Therefore the protocol is frozen, while its source roster,
source-hash audit, actual dependencies, and metric extraction remain pending.
Do not substitute any existing ComBat-processed MNPS export.

Source and metadata-only selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The permitted source is the raw PhysioNet I-CARE v2.1 WFDB training release:
per-patient/per-segment EEG ``.hea`` and ``.mat`` files plus the release's
non-outcome metadata needed to identify elapsed time from ROSC. Freeze the
release manifest and its hashes before selection. Eligible patients must have
at least one raw EEG segment with a parseable source header and time interval;
selection does not require target-period overlap, a specified number of
channels, outcome, or measurement value. Sort opaque patient IDs by SHA-256 of
``COMA-VECTOR-PREFLIGHT-001|20261013|patient_id`` and take the first 16
eligible IDs (target 12–20); record the full roster, ordering, and reason for
every exclusion. Do not open or join CPC/outcome fields during this step. A
selected patient with no target-period support remains in the preflight roster
with zero supported windows; do not replace that patient based on target
support.

Realizations and source/preprocessing audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep each patient's full native EEG-channel realization as its own output.
When and only when the exact ordered Core19 set
``Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2`` is present,
also emit that separately identified Core19 realization. Do not alias T3/T4
or T5/T6 to T7/T8 or P7/P8, interpolate, or require Core19 for native metrics.
Record each realization's actual ordered labels, label/order/set hashes,
channel count, declared reference/channel metadata, sample rate and units.

Use only raw WFDB signals and the repository's frozen I-CARE v2.1
preprocessing contract in
``mndm/config/sources/other/config_ingest_physionet_i-care_2_1.yaml`` with
``config_ingest_physionet_i-care_2_1_mnps_only.yaml``. Resolve each config's
imports and record each file hash before reading metric values. The frozen
local checksums are ``a77f16dc7109131792d6334a859918759e4009b612b40130a1429f39200d4ff1``
(base config), ``fd94d9f6c08022f71310362687386698c12c2eff0dfba1d5c8e41cdfef1b32943``
(MNPS-only overlay), and ``e1b4d3c1c9e4953166f5fb8269580f2df99a5cad7a3f520d60e9385fdf847f5b``
(``mndm/src/mndm/preprocess.py``). A mismatch requires a separately versioned
preflight protocol before metric extraction. Apply the configured per-file
sample-rate rule (base 100 Hz; integer-ratio candidates
100, 128, 200, 250 Hz, choosing the matching rate closest to 100 Hz, else the
configured 100 Hz fallback), 0.5–40 Hz EEG bandpass, no notch, average
reference over the actual EEG input set, and no artifact removal. Preserve
the configured ``scale_to_microvolts=false`` behavior; record source units and
do not label unknown ``nu`` values as microvolts. The WFDB header conversion
must use its declared gain/baseline as implemented by the pinned
``preprocess_wfdb`` code. Do not apply ComBat, ICA, interpolation, manual
artifact removal, silent channel dropping, or a second rereference. Record
source-header and payload hashes, actual sample rate, filter/resampler
coefficient and code hashes, reference-set hash, and the exact source
dependency/boundary behavior before metric extraction. If support cannot be
audited for a window, preserve its row as unsupported. Windows may not cross a
source-segment or verified gap boundary.

Sampling and per-window audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within each selected patient, inspect only source timebase, channels, signal
availability, support, and machine-readable suppression, discontinuity and
artifact indicators. Sort segments by parsed source start time, breaking ties
by normalized source-relative path. Choose the earliest segment whose verified
time interval overlaps the 12–24 h target period; if none overlaps, retain the
patient with zero target windows. Enumerate every full non-overlapping
20-second interval ``[20k, 20(k+1))`` seconds from ROSC that falls wholly
inside ``[12 h, 24 h)`` and inside that chosen first target-overlapping source
segment. Keep a window row even when one
or more families cannot compute. Before ED/OI/NIE extraction, write a
source-local manifest containing
the actual segment/time IDs, nominal and effective half-open sample
dependencies, gap boundaries, filter margins/warmup, missingness, suppression
and artifact flags, and all source/preprocessing/channel/reference hashes.
This establishes technical support; it is not outcome-based inclusion.

Compute ED, OI and NIE independently on every computable realization/window.
Retain NIE as a comparator. Preserve invalid values as null with family-local
reasons; do not zero-fill or suppress another family's valid result. Mark all
outputs experimental and ``qualified=false``. Only after this outcome-blind
preflight is complete may a new CPC 1 versus CPC 5 protocol be proposed; it
must freeze patient-grouped evaluation and conventional qEEG comparators before
revealing or joining outcome labels.

Outstanding source blocker
~~~~~~~~~~~~~~~~~~~~~~~~~~

This workspace does not currently contain I-CARE v2.1 raw ``.hea/.mat``
files, so no I-CARE patient identifiers, selection draw, exact segment
dependencies, preprocessing coefficient hash, or measurements are claimed.
The execution status is ``PREFLIGHT_PROTOCOL_FROZEN_SOURCE_AUDIT_PENDING``.

AR and PCV implementation boundary
----------------------------------

The v3.2 AR activity-vector kernel and PCV phase-order-parameter kernel are
implemented as low-level, versioned formula APIs. Their input vectors or phase
matrices must come from an explicitly identified upstream construction. This
does not select an EEG preprocessing path, reference, sensor set, filter
support, block partition, or full recording realization. Activity and
connectivity AR remain distinct; component values are never averaged. PCV is
not labeled neural metastability. See :doc:`mnps_v3_2_pcv_oi_ar_contract_003`
for the unresolved end-to-end contracts.

Claim ledger
------------

* ``NIE(W)`` tracks mean absolute Pearson strength closely in SL-006; this is
  an internal descriptive result.
* Whether NIE has a reproducible topology component beyond its fixed edge
  multiset remains unverified until NIE-INCREMENTAL-001 completes.
* OI's information beyond conventional spectral/autocorrelation descriptors
  remains an unverified hypothesis until OI-TEMPORAL-SPECIFICITY-001 completes.
* OI sensitivity to temporal permutation does not establish nonlinear
  dynamical specificity.
* ED recovery-linked information in coma and any v3.2 improvement over v3.1
  remain unverified.
* No I-CARE metrics have been computed under COMA-VECTOR-PREFLIGHT-001.

Implementation artifacts and separate output roots
---------------------------------------------------

* Runner: ``mndm/src/mndm/mnps/v3_2/experiments/sl007_candidate_value.py``.
* Focused tests: ``mndm/tests/test_mnps_v3_2_sl007_candidate_value.py``.
* NIE output root: ``outputs/NIE-INCREMENTAL-001/``.
* OI output root: ``outputs/OI-TEMPORAL-SPECIFICITY-001/``.
* I-CARE protocol/preflight record: this page; no output data until raw source
  hashes and dependency manifests can be produced.

