MNPS v3.2: VECTOR-PILOT-001 protocol
===================================

Status and scope
----------------

**Frozen bounded engineering protocol**, authorized by science-lead review
SL-005. The experiment asks whether ED, OI, and NIE can be assembled from the
same audited ANPHY EPCTL01 recording and 20-second windows. It does not qualify
the v3.2 vector, claim cross-modal equivalence, select a biological mechanism,
estimate a Jacobian, or integrate these estimators into pipeline/export.

The experiment processes exactly one recording and preserves all 15 nominal
windows in the audit output. The first window remains explicitly unsupported;
the comparison candidate intersection is the 14 supported windows at indices
1 through 14. No subject/session aggregate is defined.

Frozen realization IDs
----------------------

* Experiment: ``nmdm.mnps_v3_2.vector_pilot_001.v1``.
* ED: ``ED.anphy_epctl01_E1_G2_fixed83_avgref.v1``; kernel
  ``ED.sensor_covariance_participation_ratio_raw_matrix.v1``.
* OI: ``OI.anphy_epctl01_G2_fixed83_PE5_delay1_median.v1``; kernel
  ``OI.matrix_input_channel_median_normalized_PE5_delay1.v1``.
* NIE: ``NIE.anphy_epctl01_E1_G2_fixed83_avgref_abs_pearson_no_threshold.v1``;
  kernel ``D1.sensor_weighted_global_efficiency_invweight.v0``.
* NIE edge estimator: ``absolute_zero_lag_Pearson_r``; edge rule:
  ``no_threshold``.

The NIE realization ID is specific to this ANPHY source branch. It does not
reuse the ds004148-bound realization in
:doc:`mnps_v3_2_nie_edge_contract_001` and does not alter the NIE matrix
kernel.

Input and source identity
------------------------

Use the existing frozen E1-G2 audited EEG grid and loader once. Expected source
identity:

* Recording: ANPHY / EPCTL01; dataset release/session/task/run identifiers are
  unresolved and remain a context limitation.
* EDF SHA-256:
  ``a16ddc068edcc4ce2ff8c499cca51b9a890b76b7ddb569f9a96978013f8b5e8c``.
* Positions SHA-256:
  ``b5bbc85ef1e840e0ca703c62c22a1000dd3252f32210e251b8d0e81123fa1bd2``.
* E1-G2 overlay SHA-256:
  ``3f1d3135d45ad43648ef1f1c24b0732970ad1492209a414804c35c6eed240824``.
* Ordered 83-channel set: reuse the position-file labels exactly and in order.
  v3.2 JSON-label order hash:
  ``9559b64ead401c7399f6218720bc47fe9f7b7c8b785c27c48526542412e2e1e7``.
  Sorted set and average-reference-set hash:
  ``9819377836ecdb54673e8eb5e84a9dc4c19ee0a42e0ff665c9f56794e914fcce``.
* Preprocessing: continuous 1000 Hz EDF; causal polyphase 1/4 resampling
  (81 taps, coefficient SHA-256
  ``33a200a29b1076712d73b299c490abad2e3d6096e6f75a67fb8b4476af66ce73``);
  minimum-phase 1–45 Hz FIR bandpass (825 taps, constant padding, coefficient
  SHA-256
  ``fb33689327b6b81f630bf36cdf04b794ba66a022d3d1eb40b1b8a9dc2d081537``);
  instantaneous average reference over the exact fixed 83-channel surface.
  No notch, CSD, channel selection, interpolation, or delay compensation.
* Output grid: 250 Hz; half-open 20-second windows; exactly 5,000 samples per
  full window.

For every ready window, all three kernels consume the same float64
``(83, 5000)`` array and exact ordered labels. Record its content hash once and
repeat it in each family record. Retain the loader's source-dependency bounds,
support flags, preprocessing and coefficient hashes, source/config hashes,
and runtime/code/protocol hashes. Do not re-reference or otherwise transform
the EEG matrix between the three estimators.

Window 0 is ``[0,20)`` with effective raw-source dependency
``[-3376,19997)``. Its dependency reaches 3.376 seconds before the recording,
so no signal matrix is supported. Preserve all three values as null with status
``unsupported`` and reason
``insufficient_effective_filter_edge_support``. Never shift the window, pad,
truncate, impute, or use it in an association. Windows 1–14 are expected to be
supported, including ``[20,40)`` with dependency ``[16624,39997)`` and
``[280,300)`` with dependency ``[276624,299997)``. Effective dependencies of
adjacent windows overlap; the 14 values are not independent observations.

Estimator handoff
-----------------

* ED uses the unchanged raw sensor-covariance participation-ratio kernel:
  centered sample covariance (``ddof=1``), no shrinkage, and no normalization
  by channel count. Preserve its raw matrix-kernel result and add source/time
  support only in the outer audited record.
* OI uses the unchanged order-5, delay-1 normalized permutation entropy kernel,
  stable mergesort tie handling, and median over all supplied rows. Preserve
  the raw matrix-kernel result and audited wrapper record.
* NIE computes ordinary Pearson correlations from centered float64 samples.
  Use the centered Gram matrix, read each upper-triangle pair once, and assign
  that same pair value to both undirected entries. Apply absolute value to
  off-diagonal correlations and set the diagonal exactly to zero. There is no
  threshold, clipping, shrinkage, rank transform, or matrix repair. Constant
  channels, non-finite values, out-of-range weights, or any other invalid
  kernel input yield an NIE-local null with its reason; ED and OI remain
  available. Save each successfully constructed edge matrix and its SHA-256.

Each output family record contains its versioned estimator and realization,
value or null, computed/valid/qualified state, status/reason/QC, actual ordered
labels and order/set hashes, count, average-reference identity/set hash,
window support and source dependencies, and reproducibility metadata. Retain
the raw matrix-kernel results separately from the audited records. Every result
has ``qualified=false``.

Descriptive analyses and controls
---------------------------------

Calculate Pearson pairwise associations among ED, OI, and NIE on the
pair-specific windows where both values are valid. Report ``n``, the exact
window IDs, and the coefficient only; do not report inferential p-values or
confidence claims. These are exploratory associations from one recording,
with overlapping effective dependencies, and cannot establish independence,
orthogonality, predictive value, or population behavior.

Run one fixed synthetic negative-control batch, separate from empirical EEG:
64 paired replicates of 83 independent stationary Gaussian AR(1) channels,
``rho=0.95``, 5,000 samples at 250 Hz. Generate each replicate with NumPy
PCG64 using a seed derived from the frozen experiment/version and zero-based
replicate index. For each replicate, compute ED/OI/NIE once before and once
after instantaneous average reference over the same 83 synthetic channels.
This directly checks finite-window absolute-correlation behavior and the
reference transformation under a fixed independent colored-source model. It
does not reproduce the EDF resampler/filter branch, model EEG, or supply an
empirical null distribution. Report descriptive distributions and paired
reference-minus-unreferenced changes without significance claims or baseline
subtraction from EEG measurements.

HOLD and success interpretation
--------------------------------

Hold the matched pilot if the source, exact fixed-83 labels/reference,
preprocessing coefficients, 20-second grid, or per-window dependency audit
cannot be reproduced, or if NIE requires an unspecified edge transform. A
family-local numerical failure remains null for that family and window; it
does not erase other computable metrics. Engineering success means every
nominal row and local failure is accounted for, the measurement kernels and
shared inputs are reproducible, and the records preserve support and
realization identity. Even a successful run is only a one-recording
engineering pilot. SL-006 is not authorized by this protocol.

Tracked implementation and outputs
-----------------------------------

* Runner: ``mndm/src/mndm/mnps/v3_2/experiments/vector_pilot_001.py``.
* Focused controls: ``mndm/tests/test_mnps_v3_2_vector_pilot_001.py``.
* Byte-identical, tracked E1-G2 overlay:
  ``mndm/config/mnps_v3_2_anphy_epctl01_e1_g2_grid_20260929.yaml``. Its SHA-256
  is unchanged from the previously audited ignored copy.
* Results page: :doc:`mnps_v3_2_vector_pilot_001`.
* Default run root: ``outputs/VECTOR-PILOT-001/`` (ignored run data; no raw
  EDF is copied).

The protocol and executable runner live outside the ignored ``project/``
directory so repository history can recover the procedure.
