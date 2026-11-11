MNPS v3.2: NIE-REFERENCE-NULL-001
================================

Status: completed bounded synthetic engineering diagnostic
-----------------------------------------------------------

This is the SL-004 priority 1 reference-null experiment. The frozen
``NIE-EDGE-CONTRACT-001`` and ``D1.sensor_weighted_global_efficiency_invweight.v0``
kernel are unchanged. This experiment processed no participant or empirical
EEG data, ran no empirical graph comparison, changed no pipeline/export path,
and leaves all results ``qualified=false``.

Protocol and run
----------------

- Frozen protocol: ``project/new_mnps_v.3.2/protocols/NIE-REFERENCE-NULL-001.md``
- Protocol ID: ``nmdm.mnps_v3_2.nie_reference_null.v1``
- Run: ``outputs/NIE-REFERENCE-NULL-001/run_20261012T142730Z_sl004``
- Run manifest SHA-256: ``1c00fce276cdfbb97309ca083368fec72db50b1bce46c1f2c6a4e5bb48e9b042``
- Protocol SHA-256: ``7b0249004a1a2e2642fb805322a07273ffc091ed654b52404de492ac5f377536``
- Kernel SHA-256: ``5c570126fb2f91e1122d147f95f17416e4231f2f86a4e6013efa81b80bd6528f``
- Runner SHA-256: ``0310553d6df3d03ef8f065aaa4eaffcf5c5d7b56e6938dabcc6fd611b686d03b``
- Environment: Python 3.12.0, NumPy 1.26.4, SciPy 1.14.1, MNE 1.6.1.

The frozen source model used 64 independent stationary Gaussian AR(1) channels
(``rho=0.95``), 60 seconds at 500 Hz, initialized with ``x[0]=epsilon[0]`` and
innovation scale ``sqrt(1-rho^2)``. Seeds use SHA-256-derived unsigned 64-bit
values and NumPy PCG64. The same generated and preprocessed data were compared
as N=10, 19, and 64 channel prefixes, with and without average reference.
Each replicate contributes two half-open 20-second windows at 20–40 and
40–60 seconds.

Preprocessing used the repository's frozen branch: causal polyphase 500→250 Hz
(``up=1``, ``down=2``, 41 taps), followed by the MNE minimum-phase 1–45 Hz FIR
(``firwin``, Hamming, automatic length, constant padding; 825 taps). The branch
was not rereferenced. To avoid its zero-extension startup region, both windows
were selected after the beginning of the generated recording. Their exact
source dependency bounds were 8312–19998 and 18312–29998, inclusive; every
window passed the finite dependency and finiteness checks.

Analytic population result
--------------------------

For independent equal-variance sources, instantaneous average reference
produces population off-diagonal correlation ``-1/(N-1)``. The frozen absolute
edge rule therefore produces a complete graph with equal edge weight
``1/(N-1)`` and NIE ``1/(N-1)``. The unreferenced population graph has zero
off-diagonal correlations, no edges, and NIE zero.

================== ========================== ======================
Channels N         Population reference NIE   Population unref NIE
================== ========================== ======================
10                 0.111111                   0
19                 0.055556                   0
64                 0.015873                   0
================== ========================== ======================

The runner passed 9/9 controls against the unchanged kernel: empty graphs,
the analytic equal-weight complete graph, and unit-weight path graphs at each
channel count. Maximum analytic-control absolute error was below
``1.4e-16``.

Finite-window simulation results
--------------------------------

Each summary below describes 128 windows from 64 deterministic replicates.
The 2.5%–97.5% ranges are empirical batch quantiles, not inferential
confidence intervals. Windows within a replicate are dependent.

================= =================== ================== ================== ====================
Channels N        Analytic ref NIE    Finite unref mean  Finite ref mean    Paired ref−unref mean
================= =================== ================== ================== ====================
10                0.111111            0.04397            0.11730            0.07333
19                0.055556            0.04652            0.07355            0.02703
64                0.015873            0.05108            0.05415            0.00307
================= =================== ================== ================== ====================

The finite-window reference NIE ranges were 0.11291–0.12214 (N=10),
0.06991–0.07720 (N=19), and 0.05276–0.05581 (N=64). Paired reference-minus-
unreferenced differences ranged 0.06828–0.07874, 0.02473–0.02915, and
0.00254–0.00349, respectively.

The mean signed correlations after reference were -0.11100, -0.05550, and
-0.01586, close to the analytic population values. Mean absolute edge weights
after reference were 0.11118, 0.06025, and 0.03842. Without reference, their
means were 0.03583, 0.03611, and 0.03623; finite sample estimation plus
absolute value makes those weights positive even though the unreferenced
population correlations are zero. All finite graph rows had edge density 1.0
under the predeclared no-threshold rule.

The finite-window results do not equal the analytic complete, equal-weight
population graph: estimated edge weights vary, and shortest paths can use
multi-edge routes. The result demonstrates two effects in this specified
simulation: an absolute-correlation/shortest-path baseline exists even without
reference, and average reference adds a paired NIE increase whose size varies
with channel count. The analytical value is not subtracted from finite
measurements and is not a general correction for heterogeneous data.

QC and limits
-------------

- 768/768 simulated window-condition rows were computed, valid, finite,
  support-complete, and ``qualified=false``.
- 9/9 analytic graph controls passed. The constant-channel failure test
  returned the candidate-local null reason
  ``constant_channel_pearson_undefined``.
- All outputs, the full seed list, per-replicate source and filtered-array
  hashes, realized preprocessing tap hashes, code hashes, and dependency
  versions are recorded in the run manifest. CSV artifact hashes are in that
  manifest; its companion ``run_manifest.sha256`` records the final manifest
  hash.
- Pearson correlations are formed from the centered cross-product (Gram)
  matrix, which is the ordinary Pearson definition and preserves exact
  symmetry for the frozen kernel contract. Edges are not repaired or
  symmetrized after estimation.
- The AR(1) model is not a generative model of EEG. The finite-batch quantiles
  describe only this deterministic simulation. The experiment does not
  validate neural connectivity, exclude other reference effects, compare
  empirical graph topology, or qualify NIE.

Validation command: ``python -m pytest
project/new_mnps_v.3.2/experiments/test_nie_reference_null_001.py -q`` — 5
passed. The matrix kernel, edge definition, estimator IDs, and ``qualified``
status were not changed.
