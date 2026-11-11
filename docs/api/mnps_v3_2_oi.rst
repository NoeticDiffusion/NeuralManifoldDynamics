MNPS v3.2 matrix-level OI
========================

Status: implemented as a standalone experimental matrix kernel. It is not
integrated into a preprocessing pipeline, candidate grid, export path, or
scientifically qualified measurement.

Estimator identity
------------------

* Family: ``OI``
* Estimator: ``OI.matrix_input_channel_median_normalized_PE5_delay1.v1``
* Result schema: ``nmdm.mnps_v3_2.oi_matrix_result.v1``
* Units: normalized entropy (dimensionless)

The identifier is new. It does not rename or replace the fixed-set v3.1 E1
estimator ``mndm.mnps_v3_1.e1_channel_median_pe.v0`` or the separate v3.1
native-window pipeline measurement
``OI.pipeline_native_window_permutation_entropy_order5_delay1.v1``.

Frozen matrix-level definition
------------------------------

The input must be a finite, real ``(channels, samples)`` array. For each row,
the estimator constructs every consecutive order-5 pattern at delay 1 sample.
For ``T`` samples this gives ``T - 4`` patterns. Pattern positions are sorted
with stable mergesort, so exact ties retain their original within-pattern
sample order. The Shannon entropy uses natural logarithms and is divided by
``ln(5!) = ln(120)``. The reported scalar is the median across all supplied
matrix rows, in their input order. No fixed sensor set is required by this
matrix kernel; it does not drop channels, impute data, or remove non-finite
samples. Non-finite data and ``T < 5`` return an unsupported null value.

Constant and tie-heavy rows remain computable: the declared stable tie rule
assigns their ordinal patterns. No extra variance threshold or minimum number
of distinct observed patterns is imposed.

Audit boundary
--------------

The kernel knows the matrix dimensions and can preserve optional ordered row
labels with separate order-sensitive and set hashes. Labels are audit metadata;
missing, malformed, or count-mismatched labels do not suppress an otherwise
computable value. A label issue is retained in QC, and invalid labels do not
receive hashes.

The matrix alone does not establish source identity, sample rate, segment/gap
support, preprocessing, reference, or a named native/fixed/study-common sensor
realization. The common record therefore reports temporal support as unknown,
the actual realization as unresolved/incomplete, reproducibility as not
supplied, and qualification as false. The matrix column count is recorded as
an array fact; it is not evidence of a time window or supported recording
interval. QC names the unaudited preprocessing, reference, realization, and
temporal/source support.

This is not an end-to-end 4/8/12/20-second EEG realization. In particular, no
sampling rate, filter, resampling method, reference, source dependency, gap
policy, fixed channel list, or effective window support is selected here. A
wrapper may only populate those fields after independently auditing them.

Reference implementation distinction
--------------------------------------

The v3.1 helper ``features/eeg_entropy.py`` also uses stable mergesort for ties,
but it filters non-finite samples and has its own minimum-length behavior. The
fixed-set E1 path additionally requires 83 declared channels, 250 Hz and a
20-second support contract. This kernel rejects a matrix containing any
non-finite sample, accepts the mathematically minimal five samples, aggregates
over every supplied row, and lacks E1's preprocessing/support contract. It is
therefore a separately versioned candidate, not a parity adapter.

Validation
----------

Ten deterministic tests in ``mndm/tests/test_mnps_v3_2_oi.py`` cover a known
tie-heavy pattern distribution, median aggregation, channel-row order and hash
behavior, the common measurement envelope, malformed optional labels, and
unsupported short, non-finite, complex, and zero-channel inputs; they also
check parity with the v3.1 helper on finite supported rows. The focused test
command passed 10 tests. The combined ED/OI/NIE and legacy-parity command
passed 73 tests on 2026-10-12. No EDF/pipeline pilot, cross-window support
audit, reliability analysis, or scientific qualification was performed.

Source: ``mndm/src/mndm/mnps/v3_2/oi.py``. The broader PCV/OI/AR decision
record remains at ``mnps_v3_2_vector_contract_002.rst``; PCV, AR, and the full
OI EEG realization remain unresolved.
