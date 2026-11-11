MNPS v3.2: raw network integration efficiency
==============================================

Status
------

``D1.sensor_weighted_global_efficiency_invweight.v0`` is a standalone,
experimental matrix-level estimator. It computes from an already constructed
coupling matrix and does not define how EEG connectivity is estimated. It is
not registered in the feature pipeline or export schema. ``qualified`` is
always false and temporal/source support remains unknown.

The kernel is available as
``mndm.mnps.v3_2.nie.compute_nie_raw_weighted_global_efficiency``. For a
symmetric matrix ``W`` with zero diagonal and off-diagonal weights in ``[0,1]``,
positive weights become inverse-weight edge lengths and zero weights are
absent edges. The output is the mean reciprocal shortest-path distance over
all ordered pairs of distinct nodes; disconnected pairs contribute zero.

The caller must supply unique, nonempty node labels in matrix order, their
verified order hash, and nonempty ``edge_estimator_id`` and ``edge_rule_id``.
Invalid matrix or provenance inputs produce a null value with a reason. The
exact symmetry and zero-diagonal checks follow the legacy contract; no numeric
tolerance or matrix repair is introduced.

Validation record
-----------------

On 2026-10-12, this command passed with 22 tests:

.. code-block:: console

   python -m pytest mndm/tests/test_mnps_v3_2_nie.py -q

The tests cover a known weighted chain, complete and disconnected graphs,
unsupported matrix values and shapes, complex weights, and ordered-label and
edge-provenance failures. These checks establish the matrix formula for the
declared input contract. They do not validate any edge estimator, compare
topology against overall strength, establish temporal reliability, or support
a biological interpretation.
