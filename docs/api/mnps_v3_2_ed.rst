MNPS v3.2: raw effective dimensionality
========================================

Status
------

``ED.sensor_covariance_participation_ratio_raw_matrix.v1`` is an experimental
matrix-level estimator. It is technically validated by deterministic synthetic
controls and a bounded numerical parity check. It is not registered in the
feature pipeline or export schema, and ``qualified`` is always false.

The raw kernel is available as
``mndm.mnps.v3_2.ed.compute_ed_raw_sensor_covariance_participation_ratio``.
It implements

.. math::

   C = \frac{XX^\top}{T-1}, \qquad
   ED = \frac{\operatorname{tr}(C)^2}{\operatorname{tr}(C^2)},

where each sensor row is temporally centered. The input is float64, shaped
``(channels, samples)``. Support requires ``N >= 2``, ``T > N``, finite input,
and a finite positive covariance spectrum. Shrinkage, ``ED/N``, filtering,
window extraction, resampling, rereferencing, and effective-sample correction
are outside this estimator.

The result records temporal/source support as ``unknown`` because this module
receives only an array. Numeric validity and audit completeness remain separate;
missing labels or support do not fabricate provenance. Invalid labels preserve
a computable ED value but yield null label hashes with QC. Complex input returns
``value=null`` and ``complex_signal_not_supported`` before float conversion.

The v1 PSD rule rejects eigenvalues below
``-1e-10 * max(max(abs(eigenvalues)), 1.0)`` and clips only smaller negative
roundoff. This tolerance was retained from the inspected v3.1 E3 code for
legacy numeric parity. It is scale dependent below unit covariance magnitude;
the near-singular scale sweep below is a sensitivity check, not a proof that
the tolerance is optimal. A changed tolerance would require a new numerical
policy and estimator/schema version.

Validation record
-----------------

On 2026-10-12, this command passed with 24 tests:

.. code-block:: console

   python -m pytest mndm/tests/test_mnps_v3_2_ed.py mndm/tests/test_mnps_v3_1_eeg_metrics.py -q

The v3.2 tests cover rank-one and equal-eigenvalue known cases, unsupported
inputs, unknown-support/null serialization, channel permutation and label QC,
complex arrays including object arrays, and a colored AR(1) signal with
independent-sample support explicitly left ``not_assessed``. A fixed-seed
accepted-support comparison matched the v3.1 E3 value at ``rel=1e-13`` and
``abs=1e-13``. A near-singular covariance with eigenvalues
``(1, 0.5, 0.25, 1e-10)`` remained ED-invariant within ``rel=1e-8`` under
uniform signal gains ``1e-8``, ``1``, and ``1e8``.

These checks establish numerical behavior for the specified inputs. They do
not establish temporal reliability, effective independent sample count,
sensor-set equivalence, or biological qualification.

ED challengers and ED-RELIABILITY-001
-------------------------------------

The channel-count diagnostic and isotropic covariance shrinkage are separate
experimental estimator IDs documented in
:doc:`mnps_v3_2_ed_challengers`. They do not alter raw ED v1 or correct
cross-realization comparability.

The bounded, single-recording within-recording stability protocol is documented
in :doc:`mnps_v3_2_ed_reliability_001`. It uses the first participant in the
outcome-blind MNPS-BENCH-001 selection and retains all supported window-level
values. The frozen run completed on 2026-10-12 as an engineering execution:
3,936 real-data measurements across all 82 windows and 19,200 AR(1) simulation
measurements across 48 cells. Every row was computed and valid; qualification
remains false. ED/N matched raw ED divided by channel count, while the lambda=1
control stayed within ``5.69e-14`` of the channel count on real windows and
``4.27e-14`` in simulation. These are implementation checks, not reliability
criteria.

The source protocol and review materials also remain in the git-ignored
``project/new_mnps_v.3.2`` working folder. This tracked documentation page is
the authoritative copy of the estimator definition and validation record.
