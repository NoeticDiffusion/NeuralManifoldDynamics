MNPS v3.2: ED diagnostic and isotropic-shrinkage challengers
============================================================

These are separate experimental measurement identities alongside raw ED v1.
They are not registered in the pipeline or export schema, and their
``qualified`` flag is always false.

Channel-count diagnostic
------------------------

``ED.sensor_covariance_participation_ratio_normalized_by_channel_count.v1``
is derived only from a successful raw ED result:

.. math::

   ED_N = ED_{raw} / N.

This is a descriptive channel-count diagnostic. It does not correct
differences in montage, reference, covariance rank, temporal support, or
cross-realization comparability. Report it beside absolute ED and preserve the
source raw estimator identity.

The API is ``mndm.mnps.v3_2.ed_normalized.compute_ed_normalized_by_channel_count``.
Unsupported raw ED remains null with its reason; this function does not
recompute or repair the source measurement.

Isotropic covariance shrinkage
------------------------------

``ED.sensor_covariance_participation_ratio_isotropic_shrinkage.v1`` computes
the participation ratio after applying an explicit isotropic covariance
mixture:

.. math::

   C_\lambda = (1-\lambda)C + \lambda\frac{\operatorname{tr}(C)}{N}I,
   \qquad 0 \leq \lambda \leq 1.

The API is ``mndm.mnps.v3_2.ed_shrinkage.compute_ed_isotropic_shrinkage``.
The parameter is part of the result and estimator realization; it must not be
selected from recording outcomes. For the frozen reliability experiment,
``lambda=0.5`` is a midpoint sensitivity arm and ``lambda=1`` is an algebraic
control. With positive trace, the latter gives ``ED=N`` (absolute error at
most ``1e-12`` in the protocol) and is not evidence for neural dimension or a
promotion candidate. ``lambda=0`` is tested for parity with raw ED and is not
an experiment arm.

Both functions accept arrays/results with unknown source and time support,
record that limitation in the shared measurement contract, and make no
biological claim. The definitions do not establish that raw ED or these
challengers are reliable or scientifically useful.

Validation record
-----------------

Deterministic synthetic controls verify ED/N provenance and unsupported-value
propagation, the ``lambda=0`` parity check, the ``lambda=0.5`` formula, the
``lambda=1`` channel-count control, and rejection of invalid shrinkage
parameters. The combined focused suite passed on 2026-10-12:

.. code-block:: console

   python -m pytest mndm/tests/test_mnps_v3_2_contracts.py mndm/tests/test_mnps_v3_2_ed.py mndm/tests/test_mnps_v3_2_ed_challengers.py mndm/tests/test_mnps_v3_2_nie.py mndm/tests/test_mnps_v3_2_nie_topology_001.py mndm/tests/test_mnps_v3_1_eeg_metrics.py -q

Result: **63 passed**. This is engineering evidence for the declared
matrix-level contracts, not an empirical reliability or qualification result.
