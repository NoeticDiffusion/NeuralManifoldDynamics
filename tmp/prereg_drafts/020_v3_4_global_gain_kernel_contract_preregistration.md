# V3-4 `global_gain_kernel_12d` contract-only preregistration revision (020)

**Status: IMMUTABLE REVISION OF 019 — CONTRACT-ONLY SCOPE.**  
**Date:** 2026-09-27  
**Predecessor:** `019_v3_4_global_gain_kernel_contract_preregistration.md`,
SHA-256
`7ec6c2e9004f052bac73f5058fc83f1b117ccfd7b67a6fd6ff3894d0a942e5f0`.

This revision does not edit 019 or any source, implementation, test, fixture,
result, or experiment file.  The effective preregistration is 019 with only
the two amendments in §§1–2 below.  Every equation, matrix, row ID, contract
scope, geometry annotation, transition order, resolved input, failure/null
rule, historical-output boundary, and review gate in 019 is preserved unless
explicitly amended here.

## 1. Amendment: fixed future Ftrue candidate and fail-closed outcome

The future Ftrue candidate remains fixed in advance as the single cell
`GLOBAL_GAIN_KERNEL(eta_G=0.05, K_EXP_ALIGNED)`.  This is a precommitment, not
a selection rule.  No eta, K fixture, or replacement candidate may be chosen,
retuned, or substituted after observing the contract-only run, q values,
geometry annotations, parity outcomes, endpoint values, ranks, or any other
output.

The later Ftrue gate must first assess this exact fixed candidate under the
ordinary contract validity rules already frozen in 019.  If it is
`CONTRACT_INVALID`, method-limited, endpoint-unmatched, nonfinite, or lacks
any required same-eta control/parity evidence, its status is retained as
reported and Ftrue qualification is withheld/stopped.  No other eta/K row may
replace it, and the seven-cell/21-slot Ftrue scope must not be silently
reopened.  This stop rule does not disable ordinary fail-closed checks: source
and state validation, finite/domain/guard checks, transition-clamp detection,
parity fingerprinting, and required control-run checks still execute and
retain their negative evidence.

Only if the fixed candidate passes the separately reviewed contract and
control prerequisites may a later, separately preregistered Ftrue execution
be considered.  That execution still requires same-eta
`K_EXP_PERMUTED_S91101`, same-eta `K_MEANFIELD_MASS_MATCHED`, and eta-zero
aligned parity auxiliaries.  By default they remain q-only auxiliaries; the
selected aligned candidate remains the only added scientific cell.  Promoting
an auxiliary to a pair cell requires a new preregistration declaring expanded
cell and pair counts.  The contract-only 24-row run never produces Ftrue
qualification.

## 2. Amendment: complete implementation and provenance manifest binding

The future manifest requirement in 019 is expanded as follows.  Before the
bounded contract run, the independent implementation/code+test review must
bind exact SHA-256 hashes and reviewed checkout paths for:

* every new `global_gain_kernel_12d` implementation module;
* the exact contract-runner module/script;
* every contract/unit-test module and fixture used by the reviewed tests;
* every provenance-relevant imported source used by those modules, including
  the adapter, simulator, resolved-parameter path, tree/kernel construction,
  and any scorer or contract collector that is imported or otherwise used.

At minimum, the source manifest must explicitly account for the following
known provenance surfaces (with the exact implementation-time path and hash):

* `src/m3-bridge-sim/src/m3_bridge_sim/micro/tree_coupled_adapter.py`;
* `src/cytodend_accessmodel/simulator.py`;
* `src/m3-bridge-sim/src/m3_bridge_sim/evaluation/protocol_parameters.py`;
* `src/cytodend_accessmodel/tree_coupling.py`;
* `src/m3-bridge-sim/src/m3_bridge_sim/evaluation/gate_q_protocol.py`;
* `src/m3-bridge-sim/src/m3_bridge_sim/evaluation/v3_3_closure_geometry.py`;
* `src/m3-bridge-sim/src/m3_bridge_sim/evaluation/matrix_metrics.py` when
  imported/relevant; and
* the exact scorer/contract-collector module(s), such as the protocol-011
  collector or later Ftrue scorer, when imported/relevant.  If such a scorer
  or collector is not used by the contract implementation, the manifest must
  record an explicit `NOT_IMPORTED`/`NOT_RELEVANT` disposition rather than
  silently omitting it.

The manifest must also hash 019 and this revision, 015–018, the independent
017 review, protocol 011, all candidate modules, runner, tests, fixtures, and
all transitively imported provenance sources required to reproduce the
resolved state and six-window order.  The implementation review must compare
the exact hashes against the reviewed source snapshot and bind them before
the separate contract-run decision.  A missing, changed, unresolved, or
hash-mismatched implementation/test/source dependency is
`PROVENANCE_BLOCKED`; the run must not start, and no numeric zero or cached
substitute may be emitted.  The failed manifest retains the missing/mismatch
identity and explicit nulls for unavailable outputs.

This amendment does not authorize implementation, tests, experiments,
endpoint fitting, Ftrue matrices, Monte Carlo, or any run.  The only safe
current verdict remains **PREREGISTRATION REVISION READY; IMPLEMENTATION AND
RUN STILL GATED**.
