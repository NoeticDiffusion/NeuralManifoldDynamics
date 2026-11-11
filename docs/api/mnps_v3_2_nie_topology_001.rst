MNPS v3.2: NIE-TOPOLOGY-001 synthetic topology control
=======================================================

Decision
--------

The existing matrix NIE distinguishes the three frozen graph exemplars even
though they have the same node count, the same complete upper-triangle edge
weight multiset (including zeros), and the same mean edge strength. This is a
successful synthetic discrimination control for sensitivity to edge placement
and shortest-path structure beyond mean upper-triangle strength. It is not
evidence of biological value, reliability, neural specificity, or scientific
qualification. The NIE estimator was not changed.

Frozen protocol
---------------

The protocol is ``NIE-TOPOLOGY-001`` (experiment date 2026-10-12), frozen
before the completed run. It uses five labeled nodes ``A``–``E`` and all ten
undirected upper-triangle slots in lexicographic order. Each graph contains six
zero/nonedges and positive weights ``0.2, 0.4, 0.6, 0.8``, each once. The
mean over all ten edge slots, calculated with ``math.fsum`` to avoid
order-dependent rounding, is ``0.2``; the mean among positive edges is ``0.5``.

Three fixed placements were compared:

* Star: ``AB=.2, AC=.4, AD=.6, AE=.8``.
* Path: ``AB=.2, BC=.4, CD=.6, DE=.8``.
* Triangle plus a disjoint edge: ``AB=.2, AC=.4, BC=.6, DE=.8``.

The exact reference enumerates every labeled assignment by choosing four
positive positions among ten and assigning the four distinct positive weights
in ascending permutation order. It covers ``C(10,4) * 4! = 5,040`` unique
assignments, with no random seed and no isomorphic-graph collapsing. The exact
distribution is descriptive, not a p-value or biological null. Relabeling
controls permute both axes and node labels by ``(4, 2, 0, 3, 1)``; the complete
multiset and ``math.fsum`` mean must remain exact, while NIE may differ by at
most ``1e-15`` from floating-point summation order.

The kernel IDs are
``D1.sensor_weighted_global_efficiency_invweight.v0`` and
``nmdm.mnps_v3_2.nie_matrix_result.v1``. Synthetic edge metadata passed to the
kernel are ``synthetic.fixed_weight_multiset.v1`` and
``synthetic.undirected_weight_placement_no_threshold.v1``. All outputs remain
exploratory, ``qualified=false``, with no biological claim.

Results
-------

All three exemplars had ``N=5``, four positive edges, the same six zeros and
four positive weights, upper-triangle mean ``0.2``, and positive-edge mean
``0.5``. Their NIE values differed:

.. list-table:: NIE under the frozen weight-preserving topology controls
   :header-rows: 1

   * - Placement
     - NIE
     - Exact null midrank percentile
   * - Triangle plus disjoint edge
     - 0.20400000000000001
     - 1.7659
   * - Path
     - 0.310589676989677
     - 72.2123
   * - Star
     - 0.3292857142857143
     - 98.6012

Across the 5,040 assignments, NIE ranged from ``0.19999999999999998`` to
``0.32928571428571435``; the mean was ``0.28314615685066813`` and population
standard deviation was ``0.033805011428534926``. Linear percentiles were 5th
``0.22833333333333333``, 25th ``0.2582857142857143``, median
``0.2918601503759398``, 75th ``0.31377582417582417``, and 95th
``0.32349305080884033``. Exemplar percentiles are inclusive midranks against
this exact labeled-assignment distribution, not inferential statistics.

Reproduction and provenance
---------------------------

Run the exact enumeration and write its machine-readable output with:

.. code-block:: console

   python mndm/tests/test_mnps_v3_2_nie_topology_001.py --output-dir outputs/NIE-TOPOLOGY-001/2026-10-12_frozen_multiset_n5_v1

Run the protocol/permutation tests together with the raw NIE contract tests
with:

.. code-block:: console

   python -m pytest mndm/tests/test_mnps_v3_2_nie_topology_001.py mndm/tests/test_mnps_v3_2_nie.py -q

Result: ``25 passed in 1.41s``. The separate exact enumeration command
completed successfully and wrote ``summary.json`` plus all 5,040 rows in
``exact_null_distribution.csv`` under
``outputs/NIE-TOPOLOGY-001/2026-10-12_frozen_multiset_n5_v1/``.

SHA-256 values for the frozen run are:

* Protocol ``project/new_mnps_v.3.2/protocols/NIE-TOPOLOGY-001.md``:
  ``88211fa71ed28cd095994104a9fbd3c1819a4b91fdfb3b6a8a040254e22bf62f``.
* NIE source ``mndm/src/mndm/mnps/v3_2/nie.py``:
  ``5f58f64e741302f3a58ec99d622ea902a0b630fed003afd813bcf081de62decf``.
* Executable runner/tests ``mndm/tests/test_mnps_v3_2_nie_topology_001.py``:
  ``c30ba9291895619e743c2559f61e8dec6c1df4a0b210160dec1f7f8e90b89457``.
* Output ``summary.json``:
  ``10b93a176ac415d9a5784df70e679911bcfedc87d05a609c0ec94e401e805ef0``.
* Output ``exact_null_distribution.csv``:
  ``0ac06cd4d2ebdaaac8e7447a5ef38bcc9bf290f2a48ab21ea320613421510b835``.

The protocol copy under ``project/`` is git-ignored in this repository. This
page and the source test/runner are the trackable record of the protocol result
and reproducible procedure.

Limits
------

This experiment demonstrates only that this weighted shortest-path efficiency
kernel responds to edge placement for a small, hand-specified synthetic graph
family after holding the complete weight distribution fixed. It does not
validate how empirical edges should be estimated, choose a threshold or
connectivity definition, test robustness at larger graph sizes, establish
test–retest reliability, compare modalities or sensor realizations, or
establish incremental value for a neural or clinical question. The exact
enumeration has no sampling uncertainty, but its scope is limited to this
particular ``N=5`` multiset and labeled assignment space.
