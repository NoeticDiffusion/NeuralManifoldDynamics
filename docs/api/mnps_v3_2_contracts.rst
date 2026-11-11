MNPS v3.2: shared measurement record
====================================

The experimental v3.2 raw ED, OI and NIE kernels expose a shared
``MeasurementRecord`` envelope through ``to_measurement_record()`` and under
the ``measurement_record`` key in ``to_dict()``. Its versioned identifier is
``nmdm.mnps_v3_2.measurement_record.v1``.

The record carries family and estimator identity, schema, value and units,
computation/validity/qualification flags, reason and QC, plus nested temporal
support, realization and reproducibility records. Scalar values remain
scalars; ordered vectors require an ordered component name per value. JSON
serialization uses null for unavailable metadata and lists for ordered labels,
QC codes and vector values.

The raw kernels explicitly leave recording/time support unknown because their
matrix inputs do not establish source identity, sample grid, segment, gaps,
filter support or reference. ED and OI may report the input sample count, and
the kernels report supplied labels and their validated order hashes. A numeric
result does not fill missing audit fields or change ``qualified=false``.

The record is an interface, not a claim that v3.2 pipeline/export integration
exists. A future pipeline wrapper must verify and populate source, time support,
preprocessing and requested/actual realization metadata. It must preserve
candidate-local errors and all supported windows. This interface does not
define comparability rules, qualification thresholds, a universal subject
summary, or a canonical MNPS chart.

Validation record
-----------------

The shared envelope is exercised together with ED, OI, NIE and legacy ED parity by:

.. code-block:: console

   python -m pytest mndm/tests/test_mnps_v3_2_contracts.py mndm/tests/test_mnps_v3_2_ed.py mndm/tests/test_mnps_v3_2_ed_challengers.py mndm/tests/test_mnps_v3_2_oi.py mndm/tests/test_mnps_v3_2_nie.py mndm/tests/test_mnps_v3_2_nie_topology_001.py mndm/tests/test_mnps_v3_1_eeg_metrics.py -q

Run on 2026-10-12: **73 passed**. The suite also checks OI tie handling and
matrix support, the ED/N diagnostic, isotropic-shrinkage challengers, and NIE
topology controls.
