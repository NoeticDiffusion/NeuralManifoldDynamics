MNPS v3.2: VECTOR-FEASIBILITY-002 protocol
==========================================

Status and scope
----------------

**Frozen bounded, outcome-blind feasibility and redundancy protocol**, authorized
by science-lead review SL-006. The study asks whether the existing experimental
ED, OI, and NIE estimators can be reproduced across twelve independently
selected ANPHY-Sleep participants on one common, audited 83-channel realization
and whether the descriptive redundancy seen in EPCTL01 recurs at participant
level. It is not a v3.2 release, vector qualification, predictive evaluation,
biological claim, or eight-dimensional dynamical model.

The source roster and random draw were frozen before cohort window signals or
candidate metric values were inspected. The frozen selection is in
``mndm/config/mnps_v3_2_vector_feasibility_002_selection_20261013.json``.
EPCTL01 remains the earlier development recording and is not in the sample.
There are twelve selected participants, no replacement rule, and no
outcome-driven selection or exclusion. All 15 nominal windows in the declared
first 300-second interval are retained per selected source, including any
unsupported window or family-local failure.

Corrected source identity and remaining provenance limits
---------------------------------------------------------

The source is ANPHY-Sleep, OSF DOI ``10.17605/OSF.IO/R26FH``; the dataset paper
is DOI ``10.1038/s41597-024-03722-1``. The source describes overnight sleep PSG
recordings and one subject folder per participant. The local inventory has 29
demographic IDs but EDFs for only 27 folders. The local source does not provide
an archive revision or archive-level checksum. The publication describes
EDF+D; the local EDF headers report EDF+C. The cohort is therefore identified
as the frozen local source snapshot under the public dataset accession, not
asserted to be byte-identical to a particular OSF archive revision. The
selection manifest records each selected EDF path, size, modification time,
header hash, recording date/time, and acquisition header fields. The loader
records and verifies each complete EDF SHA-256 before and after extraction.

The source supplies no explicit numeric session or run labels. Here
``session_id`` means the single overnight recording in each subject folder and
``run_id`` means that folder's sole local EDF; the absence of source-provided
numeric identifiers remains in the provenance. No sleep-stage or spindle
labels determine eligibility or selection.

Frozen sample and eligibility
-----------------------------

The metadata-only candidate pool is the 21 local EDFs whose 83 EEG labels match
the frozen position-file set case-insensitively, with no aliasing, suffix
stripping, interpolation, or channel substitution. EPCTL01 was removed from
that pool as the prior development anchor. Five other local EDFs are not
eligible for this fixed realization because their channel labels differ from
the fixed set (EPCTL05, EPCTL06, EPCTL07, EPCTL21, EPCTL25). This is a
realization-specific rule; it is not a global channel-count gate for NMD.

Selection used the sorted candidate IDs and
``numpy.random.Generator(PCG64(20261013)).choice(..., size=12, replace=False)``.
The frozen draw order is:

``EPCTL08, EPCTL12, EPCTL15, EPCTL17, EPCTL20, EPCTL19, EPCTL23, EPCTL22,
EPCTL13, EPCTL24, EPCTL03, EPCTL09``.

The fixed-header audit confirmed EDF+C, 83 selected EEG labels, 1000 samples per
one-second data record, and ``uV`` units for these selected files. The loader
still independently audits source format, time support, gaps, window bounds,
and effective filter dependencies. If a selected source or window is
unsupported, it remains in the output; no replacement is drawn.

Frozen realization and estimators
---------------------------------

* Input: one local ANPHY-Sleep EDF per selected participant.
* Fixed sensor surface: the 83 channel labels and order from the frozen
  co-registered positions file; the realization records the actual ordered
  labels, set/order hashes, channel count, and reference set.
* Preprocessing: reuse the existing E1-G2 audited loader and unchanged
  coefficients: continuous 1000 Hz EDF, causal polyphase 1/4 resampling to
  250 Hz, minimum-phase 1-45 Hz FIR bandpass, constant padding, instantaneous
  average reference over those same 83 channels; no notch, CSD, delay
  compensation, interpolation, or additional channel selection.
* Grid: the first 300 seconds, split into half-open 20-second windows
  ``[0,20)``, ``[20,40)``, ..., ``[280,300)``; each full output matrix is
  float64 ``(83, 5000)``.
* ED: unchanged raw sensor-covariance participation ratio using centered sample
  covariance (``ddof=1``), without shrinkage or channel-count normalization.
* OI: unchanged order-5, delay-1 normalized permutation entropy, stable
  mergesort tie handling, median across all supplied channels.
* NIE: unchanged weighted global efficiency from absolute zero-lag Pearson
  weights, no threshold, no clipping, no shrinkage, and inverse-weight edge
  lengths. Edge construction and graph-kernel failures remain NIE-local.

Each supported input window is passed unchanged to all three primary kernels.
The same per-window matrix hash, source dependency, channel realization,
reference set, preprocessing coefficients, config, code, protocol, and runtime
identities are carried in the result. Computability does not imply
comparability or qualification; every measurement has ``qualified=false``.

Covariance-connectivity audit
-----------------------------

On each supported common window, calculate the four SL-006 diagnostics without
redefining ED, OI, or NIE:

* ``ED(R) = tr(R)^2 / tr(R^2)`` from the Pearson correlation matrix;
* mean absolute off-diagonal correlation;
* mean squared off-diagonal correlation;
* ``NIE - mean_absolute_correlation`` as the graph-path increment.

Independently reconstruct raw covariance ED using the recorded channel
variances and pairwise correlations. Store the formula value and absolute
residual from the primary ED kernel. A failed diagnostic must not suppress a
valid primary measurement. These are structural diagnostics, not independent
biological variables.

Paired time-index permutation control
-------------------------------------

For each supported participant/window matrix, generate a deterministic
PCG64 permutation of its 5000 time indices from SHA-256 of the frozen control
ID, participant ID, and window ID. Apply that exact same index permutation to
all 83 channels. Do not independently shuffle channels, change sample values,
or refit preprocessing. Recompute ED, OI, and NIE on the permuted matrix.

The tolerance for covariance and Pearson-correlation preservation, and for ED
and NIE parity, is ``rtol=1e-10, atol=1e-12``. Record the maximum absolute
matrix difference and each paired metric delta. OI is allowed to change. A
failure in this control is reported explicitly and does not turn a computable
original measurement into zero or erase other results. This control tests
temporal-order sensitivity only; it does not establish biological utility.

Reference, amplitude, and spectral comparators
-----------------------------------------------

Keep the same average-reference operation and set for every primary output.
Report its exact identifier and label-set hash, plus the maximum absolute
across-channel mean after referencing as an implementation residual. Do not
alter the primary realization to create alternate references.

For each supported window, summarize channel standard deviation (median,
minimum, maximum) and global RMS in microvolts. As conventional comparators,
calculate one Welch PSD per channel using a Hann window, 1000 samples per
segment, 500-sample overlap, constant detrending, density scaling, and the
250 Hz output grid. Integrate bins in the frozen bands ``[1,4)``, ``[4,8)``,
``[8,13)``, ``[13,30)``, and ``[30,45)`` Hz; report per-band median and mean
channel power in ``uV^2``. These summaries describe the same filtered,
average-referenced realization and are not independent reference-sensitivity
experiments or preferred MNPS features.

Participant-level summaries and inference boundary
---------------------------------------------------

For each participant and measure, summarize valid supported windows with their
count, median, and interquartile range. Also report descriptive ED/OI/NIE and
diagnostic correlations within each participant with exact matched window IDs;
effective source dependencies of adjacent windows overlap, so these are not
independent-window evidence. The primary cross-participant association matrix
uses one median per participant and pairwise-valid participant counts. Do not
pool windows across participants as independent observations. Do not report
p-values or confidence claims. With twelve participants these correlations
remain descriptive feasibility evidence.

Predeclared downgrade-review triggers
-------------------------------------

The study cannot automatically remove or promote a coordinate. It flags a
candidate for scientific review only under these frozen conditions:

* Redundancy review: an absolute participant-median correlation of at least
  ``0.90`` for a pair of primary metrics with at least 10 pairwise-valid
  participants. This flags follow-up; it does not justify dropping either
  coordinate.
* ED variance-component review: in at least 10 participants, the participant
  median absolute difference between raw ED and ``ED(R)`` is no greater than
  ``1e-10``.
* NIE path-contribution review: in at least 10 participants, the participant
  median absolute ``NIE - mean_absolute_correlation`` is no greater than
  ``1e-10``.
* OI temporal-order review: in at least 10 participants, the participant
  median absolute original-versus-permuted OI change is no greater than
  ``1e-12``.

These are transparent review flags tied to SL-006's specified comparators and
controls, not qualification thresholds or evidence of biological redundancy.
Other values, including a large or small inter-metric correlation, remain
descriptive. No coordinate is automatically removed. Engineering PASS and
scientific HOLD are reported separately.

Engineering completion conditions
----------------------------------

Engineering PASS requires all twelve frozen source IDs to be accounted for,
each source hash stable across extraction, all 15 nominal windows represented,
all local unsupported states and family failures retained, and every computed
measurement tied to its exact input matrix, support, realization, config, code,
protocol, and source hashes. The covariance identity and permutation checks
must be reported with their frozen tolerances. It is acceptable for a selected
participant, window, or family to be unsupported if its reason and surviving
measurements remain explicit; unsupported results are never replaced or
imputed.

Scientific status remains HOLD for claims of complementary neural properties,
prediction improvement, biological interpretation, or an MNPS state-space
model. This pilot only establishes bounded measurement feasibility and
descriptive redundancy evidence.

Tracked implementation
----------------------

* Selection manifest: ``mndm/config/mnps_v3_2_vector_feasibility_002_selection_20261013.json``.
* Runner: ``mndm/src/mndm/mnps/v3_2/experiments/vector_feasibility_002.py``.
* Focused tests: ``mndm/tests/test_mnps_v3_2_vector_feasibility_002.py``.
* Results page: :doc:`mnps_v3_2_vector_feasibility_002`.
* Output root: ``outputs/VECTOR-FEASIBILITY-002/`` (ignored run artifacts;
  raw EDFs are not copied).
