# V3-4 `global_gain_kernel_12d` contract-only preregistration (019)

**Status: IMMUTABLE PREREGISTRATION DRAFT — CONTRACT-ONLY SCOPE.**  
**Date:** 2026-09-27  
**Location:** scratch only; no source, fixture, test, result, or experiment file
was changed by this document.

This preregistration freezes a new construct named
`global_gain_kernel_12d`.  It is not E022, is not an alias for
`soma_global_gain_only`, and does not promote any E022 output.  It authorizes,
after the gates in §9, only an analytic contract check and a bounded
deterministic contract run over the 20 active rows and four parity controls in
§3.  It does **not** authorize endpoint fitting, an Ftrue operator/matrix,
endpoint matching, observer/J-sign scoring, Monte Carlo, or a biological/G4
claim.  Protocol-011 historical outputs and C2 historical outputs remain
unchanged.

The accepted scientific decisions frozen here are:

1. the interaction matrix is `Q = I + K`, with the identity term explicit;
2. the gain is instantaneous and multiplies only the structural `M` write;
   it does not multiply passive decay and is not a persistent state.

These choices make the state exactly the existing ordered 12-D
`(E,P,M)` state.  No `G` coordinate, zero padding, or unregistered projection
is permitted.  The choices are new scientific semantics and must not be
described as inherited E022 behavior.

## 1. Frozen equation and semantic contract

Branches and all matrices use the exact order `[b0,b1,b2,b3]`.  Let
`E_before,b` be the eligibility read before the inherited `P` update, let
`P_tilde,b` be the inherited post-replay/post-sleep/post-clamp and
post-budget-normalization translation state, and let
`d = max(0, window.modulatory_drive)`.  Define

```text
u_b = E_before,b * P_tilde,b * d
Q   = I + K
S_Q = sum_{b,c} Q[b,c]
q_K = sum_{b,c} Q[b,c] * u_b * u_c / S_Q
g_K = 1 + eta_G * q_K

write_b = structural_lr * E_before,b * P_tilde,b * d
          * (1 - M_old,b / M_max,b)
decay_b = branch_b.structural.decay_rate * M_old,b
M_new,b = clamp(M_old,b + g_K * write_b - decay_b, 0, M_max,b)
```

The gain is the same scalar for all branches in a transition.  It multiplies
`write_b` only.  `decay_b` is not gain-scaled.  E and P retain the inherited
transition equations and order.  There is no stored `G` and no persistent
kernel state.  `eta_G=0` is a parity control, not an active candidate.

The candidate is valid only when all inputs and intermediate quantities are
finite, `S_Q > 0`, `0 <= g_K <= 2`, and the declared matrix and state domains
hold.  A guard failure or transition-side saturation is recorded as invalid;
it is never repaired by clipping, recentering, or replacing a value with zero.

## 2. Exact fixed matrices

Set `q = exp(-2)`, `q2 = q^2 = exp(-4)`, and
`m = (q + q2)/2`.  The four fixture IDs and matrices are frozen as follows.

```text
K_ZERO =
[[0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0]]

K_EXP_ALIGNED =
[[0,  q,  q2, q2],
 [q,  0,  q,  q ],
 [q2, q,  0,  q2],
 [q2, q,  q2, 0 ]]

K_EXP_PERMUTED_S91101 =
[[0,  q2, q,  q2],
 [q2, 0,  q,  q2],
 [q,  q,  0,  q ],
 [q2, q2, q,  0 ]]

K_MEANFIELD_MASS_MATCHED =
[[0, m, m, m],
 [m, 0, m, m],
 [m, m, 0, m],
 [m, m, m, 0]]
```

For the aligned star, `S_K = 6q + 6q2`; the mean-field matrix has the same
off-diagonal mass.  Every row records `K_kind`, `ell` or `mass_override`,
tree identity, assignment, permutation provenance, matrix, diagonal, row
sums, `S_K`, `S_Q`, and a fixture fingerprint.  `Q=I+K` is constructed in
the same order and is serialized; no silent relabeling, resizing, or
mean-field substitution is allowed.

The permutation is source-generated, fixed before any output, and is
`P_91101` with current-branch-to-aligned-node map `[0,2,1,3]`:

```text
P_91101 =
[[1,0,0,0],
 [0,0,1,0],
 [0,1,0,0],
 [0,0,0,1]]
```

`K_EXP_PERMUTED_S91101` is `P_91101 K_EXP_ALIGNED P_91101^T`, expressed in
canonical current-branch order.  It is not a relabeled output row.

## 3. Exact row universe and fixed scope

The active grid is eta-major, followed by the fixture order
`K_ZERO`, `K_EXP_ALIGNED`, `K_EXP_PERMUTED_S91101`,
`K_MEANFIELD_MASS_MATCHED`:

| eta | four active row IDs, in fixed order |
|---:|---|
| 0.01 | `ACTIVE_ETA_0p01_K_ZERO`; `ACTIVE_ETA_0p01_K_EXP_ALIGNED`; `ACTIVE_ETA_0p01_K_EXP_PERMUTED_S91101`; `ACTIVE_ETA_0p01_K_MEANFIELD_MASS_MATCHED` |
| 0.02 | `ACTIVE_ETA_0p02_K_ZERO`; `ACTIVE_ETA_0p02_K_EXP_ALIGNED`; `ACTIVE_ETA_0p02_K_EXP_PERMUTED_S91101`; `ACTIVE_ETA_0p02_K_MEANFIELD_MASS_MATCHED` |
| 0.05 | `ACTIVE_ETA_0p05_K_ZERO`; `ACTIVE_ETA_0p05_K_EXP_ALIGNED`; `ACTIVE_ETA_0p05_K_EXP_PERMUTED_S91101`; `ACTIVE_ETA_0p05_K_MEANFIELD_MASS_MATCHED` |
| 0.10 | `ACTIVE_ETA_0p10_K_ZERO`; `ACTIVE_ETA_0p10_K_EXP_ALIGNED`; `ACTIVE_ETA_0p10_K_EXP_PERMUTED_S91101`; `ACTIVE_ETA_0p10_K_MEANFIELD_MASS_MATCHED` |
| 0.20 | `ACTIVE_ETA_0p20_K_ZERO`; `ACTIVE_ETA_0p20_K_EXP_ALIGNED`; `ACTIVE_ETA_0p20_K_EXP_PERMUTED_S91101`; `ACTIVE_ETA_0p20_K_MEANFIELD_MASS_MATCHED` |

The four parity controls are exactly:

* `PARITY_ETA_0_K_ZERO`;
* `PARITY_ETA_0_K_EXP_ALIGNED`;
* `PARITY_ETA_0_K_EXP_PERMUTED_S91101`;
* `PARITY_ETA_0_K_MEANFIELD_MASS_MATCHED`.

They all have `eta_G=0` and must be exact parity with the inherited aligned
K0 transition, including state, reports, counters, replay counts, step
indices, clamps, and caches.  They are not active candidates and are not
eligible for endpoint fitting or candidate selection.  The 24 rows above are
the complete contract-only scope; no row may be added, dropped, pooled, or
selected using an observed result.

The five existing protocol-011 core cells remain historical references only:
`K0_ALIGNED`, `C1_ALIGNED`, `C2_ALIGNED`, `C1C2_ALIGNED`, and
`C2_MEAN_FIELD`.  `HEBBIAN_KERNEL` remains explicitly
`BLOCKED_MISSING_COMPARATOR`.  Neither the core rows nor E022 values are
retuned by this preregistration.

## 4. Per-eta geometry annotation

For each positive eta independently, retain the six pre-transition values
from the aligned, permuted, and mass-matched mean-field rows:

```text
Delta_perm(eta) = max_t abs(q_EXP_ALIGNED(eta,t)
                            - q_EXP_PERMUTED_S91101(eta,t))
Delta_mf(eta)   = max_t abs(q_EXP_ALIGNED(eta,t)
                            - q_MEANFIELD_MASS_MATCHED(eta,t))
tau_q = 1e-10
```

The primary annotation is `GEOMETRY_SEPARATING` iff all required values are
finite and `Delta_perm(eta) > tau_q`; otherwise it is
`NOT_GEOMETRY_SEPARATING` (or `GEOMETRY_ANNOTATION_INVALID` if the required
control run is invalid/nonfinite).  `Delta_mf` is retained as the
mass-matched interaction-null contrast and is not folded into the primary
decision.  Values are not pooled across eta and no eta is chosen after
observing them.  A separating scalar annotation is not by itself evidence
of a spatial-kernel causal claim.

## 5. Resolved inputs, state setup, and exact order

Every row uses the same source-resolved construction except for the frozen
eta/K fixture:

* canonical motif, `n=4`, `graph_seed=0`, branch order `[b0,b1,b2,b3]`;
* aligned exponential tree, source-generated permutation seed `91101`, or
  mass-matched mean-field as specified in §2;
* `mechanism_seed=0`, a deterministic provenance label and not an independent
  replicate;
* replay condition `all_traces`, epoch definition `consolidation_only`, and
  exactly six scheduled windows/transitions;
* the existing V3 closure-builder encoding path with `cue_scale=0.05` and
  `n_repeats=1`;
* `saturation_avoiding` resolved parameters with
  `replay_gain=0.1`, `sleep_gain=0.0`, `translation_decay=0.05`,
  `structural_lr=0.18`, `eligibility_decay=0.12`,
  `structural_decay=0.005`, `structural_gain=6.0`,
  `readout_threshold=0.3`, `structural_noise=0.0`,
  `translation_budget=0.0`, and `structural_max=1.0`;
* the source base `INITIAL_STATE=(0.0,0.05,0.45)` is not substituted for the
  epoch state.  The runner must capture and manifest the actual ordered
  12-vector immediately after encoding and before transition 0, as returned by
  the source path.

Within each window the frozen order is: (1) read all `E_before` values and
replay IDs; (2) update/clamp P with inherited replay/sleep equations; (3)
apply inherited translation-budget normalization to obtain `P_tilde`; (4)
compute `d` and `u`; (5) validate K/Q and compute `S_Q`, `q_K`, and `g_K`; (6)
compute the structural write and passive decay and update/clamp M; (7)
recompute inherited slow/effective-access caches and decay E in the inherited
post-write position; (8) update replay counts and return inherited report
fields/counters.  The same trace allocations, IDs, contexts, drives, cache
initialization, and counters are used for all rows.

For parity rows, dispatch the inherited K0 path directly, or provide a
reviewed proof of field-, report-, counter-, replay-, and cache-equivalence.
An eta-zero wrapper that merely happens to produce the same final vector is
not sufficient.

## 6. Analytic checks versus the numerical contract run

The following are **analytic/unit contract checks**, not a numerical result:

* exact construction, symmetry, zero diagonal, nonnegative entries, branch
  order, masses, row sums, permutation, `Q`, and `S_Q` for all fixtures;
* known-answer checks for `q_K`, `g_K`, the displayed M update, and analytic
  local derivatives in the unclamped interior;
* exact eta-zero parity fingerprints, including reports, counts, step indices,
  and caches;
* fail-closed checks for nonfinite/negative/out-of-grid eta, malformed,
  asymmetric, negative, wrong-shape, or wrong-order K; nonfinite/nonpositive
  `S_Q`; invalid domains; nonfinite parameters; gain guard failure; and
  transition-side clamps;
* deterministic duplicate handling: repeating seed 0 is a fingerprint/error
  check, not replication or a confidence interval;
* geometry annotation fixtures and the fixed `tau_q=1e-10` rule.

After an independent implementation/code+test review, the **numerical
contract-only run** may execute the fixed 24 rows for six transitions.  It
retains resolved states, E/P/M values, `E_before`, `P_tilde`, `u`, K/Q
metadata, `q_K`, `g_K`, write/decay terms, clamps, counters, replay IDs,
cache fingerprints, and per-eta geometry annotations.  It does not calculate
or retain FD Jacobian matrices, composed Ftrue operators, endpoint values,
endpoint gaps, linked-pair scores, observer scores, J-signs, or Monte Carlo
summaries.  This distinction is contractual: passing analytic checks or a
finite trajectory is not an Ftrue qualification.

## 7. Failure, invalidity, retention, and null policy

Rows fail closed before transition if eta is nonfinite, negative, outside the
declared grid (except eta-zero parity), inconsistent across windows, or if a
source parameter, state, K/Q, `M_max`, `S_Q`, `q_K`, or `g_K` is nonfinite;
`S_Q <= 0`, nonpositive `M_max`, wrong branch order, nonzero diagonal,
asymmetry, forbidden negative K entry, or pre-transition E/P/M domain failure
also invalidates the row.  `g_K` is not clipped to satisfy `[0,2]`.

Any transition-side saturation, hidden input recentering, wrong update order,
wrong cache/counter behavior, or failed parity fingerprint invalidates the
affected row/control.  A missing/invalid aligned, permuted, or mean-field
control makes the corresponding geometry annotation invalid; it is not
treated as zero separation.

Every preregistered row remains in the output, in the fixed order.  A failed
row retains `row_id`, resolved inputs, failure code(s), provenance, and values
through the last valid checkpoint; unavailable numeric fields after failure
are JSON `null`.  The output must never use numeric zero as a substitute for
unsupported, invalid, or unrun quantities.  Per-eta geometry fields are
`null` with an explicit status when a required control is unavailable.

Because endpoint/Ftrue work is out of scope, each row records
`endpoint_status: OUT_OF_SCOPE_CONTRACT_ONLY`,
`endpoint_value: null`, `ftrue_status: OUT_OF_SCOPE_CONTRACT_ONLY`, and
`ftrue_operator: null`; no endpoint or matrix is computed under this
preregistration.  These nulls are scope markers, not negative biological
evidence.

## 8. Manifest, provenance, and write-once retention

The future contract archive, if authorized, uses the write-once root
`results/v3_4_global_gain_kernel_contract/` and refuses to run if that root or
its `.inprogress` sibling already exists.  It writes only to `.inprogress`
and finalizes once.  Required retained files are a protocol snapshot, a
condition/row manifest, per-row contract records, geometry annotations, a
claim ledger, and `summary/source_manifest.json`.  The manifest records:

* checkout identity, commit, dirty-state flag, Python/NumPy versions, command,
  timestamp, and environment;
* this preregistration hash; hashes of 015, 016, 017, 018, the independent
  017 review, protocol 011, and the source files below;
* resolved parameter values, actual post-encoding 12-vector, branch/field
  names, replay IDs, six-window drives, counters, cache initialization,
  fixture matrices and fingerprints, eta/K row IDs, and all failure statuses;
* explicit statement that no endpoint, Ftrue, observer, J-sign, MC, or G4
  output was requested or generated.

Read-only source references and hashes at drafting time are:

| Source | SHA-256 |
|---|---|
| `project/slow_branch_v3/015_v3_4_kernel_comparator_source_design.md` | `cbdd80ddb0b9e8472ec320a399169aa6bec4109bbfd9c63c945b40ad4c4af0e2` |
| `project/slow_branch_v3/016_v3_4_global_gain_kernel_implementation_readiness.md` | `0bf91b624da15936c3a5e1baa3563271fb29cb14b5f40dc445176c866a30a9ba` |
| `project/slow_branch_v3/017_v3_4_global_gain_kernel_readiness_revision.md` | `467fd4ab511aa3171f2ec9f6dd9dde49fdd4a7dfc08353418fe6b2b2446d031e` |
| `project/slow_branch_v3/018_v3_4_global_gain_kernel_scope_count_clarification.md` | `7b4f8180acbaeec2c7895d9844a25aa3e4b1d0a1d6258f91f7cbfa5847a865bc` |
| `project/slow_branch_v3/review/017_v3_4_global_gain_kernel_readiness_review.md` | `cf7d2810cf645414a8281d6f4b89ea22ca7d964a4ce93daf02dbfb58e15a9736` |
| `project/slow_branch_v3/011_v3_4_f_true_feasibility_protocol.md` | `1b2ff769c3da7b694e97e48d4f7267998e4a84113e31b8380e5e54810c22ccf4` |
| `src/cytodend_accessmodel/tree_coupling.py` | `13bdb26843fd9837dbc646d9698b2698a358a35b9582344b114d9cda03cf6172` |
| `src/cytodend_accessmodel/simulator.py` | `b63b965b707e4072acf72d201bb5bb3e3048d5f473e23d71dcaeead0835caffd` |
| `src/cytodend_accessmodel/contracts.py` | `3af57d9a3b3936e82802269e421b68f02a700ef693979a06d5f8aa97a72bc144` |
| `src/m3-bridge-sim/src/m3_bridge_sim/evaluation/protocol_parameters.py` | `8f166e381dd6b0c1293834b76432df4ae93e156488139aa3c160ba6194500826` |
| `src/m3-bridge-sim/src/m3_bridge_sim/evaluation/gate_q_protocol.py` | `4bc301f8114e559abe0de84619dff4e4531e96512a2b92c588401ee5d7da08ed` |
| `src/m3-bridge-sim/src/m3_bridge_sim/evaluation/v3_3_closure_geometry.py` | `8632090510fa5c7f403be4151725d9992342d1ff393ed775d2d11de4a09cf802` |
| `src/m3-bridge-sim/src/m3_bridge_sim/evaluation/matrix_metrics.py` | `f70a4946e10b3726ed9de63c00433b42dcc120116ce6b1334b8a22f4f5740a55` |
| `src/m3-bridge-sim/src/m3_bridge_sim/micro/tree_coupled_adapter.py` | `9a6bdc0af157bdd49160c737a32c26c058575ad070bfc607de0b41c2c7243095` |

The self-hash of this file is reported outside the byte stream after writing;
it is not inserted into the file, so the artifact remains hash-stable.

## 9. Required staged gates

The stages are separate and cannot be collapsed:

1. **Independent preregistration review:** a read-only reviewer must check the
   frozen equations, matrices, all 24 row IDs, geometry rule, resolved input
   path, null policy, and scope before any implementation.
2. **Independent implementation/code+test review:** after an additive
   implementation exists, a different reviewer must inspect the code, tests,
   parity fingerprints, fail-closed behavior, fixture provenance, and
   historical-output preservation. This review does not authorize a run.
3. **Separate contract-run decision:** the coordinator must explicitly decide
   whether the bounded 24-row contract run may execute. No result may be
   generated merely because implementation tests pass.
4. **Later Ftrue preregistration/execution gate:** endpoint/Ftrue work requires
   a new preregistration and separate implementation/run reviews. This
   contract-only artifact cannot be amended by looking at its outputs.

No code edit, test execution, source qualification, or numerical contract run
has been performed for this artifact.

## 10. Precommitted future Ftrue candidate (not part of this run)

If and only if a later Ftrue preregistration is accepted, the first and only
precommitted active scientific cell is
`GLOBAL_GAIN_KERNEL(eta_G=0.05, K_EXP_ALIGNED)`.  This choice is frozen now
and must not be selected, retuned, or rejected using contract-only q values,
parity outcomes, geometry annotations, endpoint values, ranks, or any other
later output.  It is a candidate-only reservation; no Ftrue status is
claimed.

That later Ftrue preregistration must require same-eta auxiliary deterministic
runs for:

1. `eta_G=0.05`, `K_EXP_PERMUTED_S91101`, with full permutation provenance;
2. `eta_G=0.05`, `K_MEANFIELD_MASS_MATCHED`, with mass provenance; and
3. the `eta_G=0` aligned parity control.

By default these are q-only geometry/parity auxiliaries, not endpoint-matched
cells and not entries in the operator-distance pair matrix.  The Ftrue scope
therefore preserves the seven conceptual cells—five existing protocol-011
core cells, the one selected global-gain cell, and blocked
`HEBBIAN_KERNEL`—and exactly 21 conceptual pair slots.  If a later protocol
promotes either auxiliary to a scientific endpoint/operator cell, it must
preregister the expanded cell and pair counts; no implicit reuse of 7/21 is
allowed.  The later protocol must also preserve protocol-011 endpoint rules,
FD grids, adjacent perturbation validity, prefix ranks, resolution envelope,
explicit nulls, and `NO_MM_SUPPORT` separation from Ftrue validity.

## 11. Unresolved blockers and choices

The material equation choices and contract-only row scope are resolved in
this draft.  The following remain gates/blockers rather than hidden
assumptions:

* independent acceptance of this preregistration before implementation;
* independent review of any additive implementation and its tests;
* coordinator authorization of the bounded numerical contract run;
* implementation-time source capture of the actual post-encoding 12-vector,
  exact six-window IDs/drives, cache fingerprints, and resolved environment;
  the source base triple must not be substituted;
* later independent preregistration and review for any endpoint/Ftrue use;
* any missing, changed, or hash-mismatched source at execution is a provenance
  blocker and invalidates the affected row rather than being silently
  repaired.

Until these gates are satisfied, the only safe verdict is
**DESIGN/PREREGISTRATION READY; NO IMPLEMENTATION OR RUN AUTHORIZED**.  No
scientific separation, geometry effect, endpoint claim, Ftrue result, or G4
claim follows from this document.
