"""Simulator bridge: connects the cytodend_accessmodel simulator to data observables.

The simulator is used as a **signature generator** — it produces model-side
linking indices and context margins that can be compared directionally with
empirical co-reactivation scores from DANDI 000718.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from cytodend_accessmodel.contracts import (
    ConsolidationWindow,
    DynamicsParameters,
    EngramTrace,
    TraceAllocation,
)
from cytodend_accessmodel.simulator import CytodendAccessModelSimulator


# ---------------------------------------------------------------------------
# Default parameter sets (mirrors exp015_comparator_baselines.py)
# ---------------------------------------------------------------------------

BASE_PARAMS = DynamicsParameters(
    fast_gain=2.0,
    structural_gain=2.0,
    structural_lr=0.15,
    structural_decay=0.01,
    eligibility_decay=0.1,
    translation_decay=0.05,
    context_mismatch_penalty=0.25,
    replay_gain=1.0,
    sleep_gain=1.0,
)

FAST_CONTEXT_PARAMS = DynamicsParameters(
    fast_gain=2.0,
    structural_gain=0.0,
    structural_lr=0.0,
    structural_decay=0.01,
    eligibility_decay=0.1,
    translation_decay=0.05,
    context_mismatch_penalty=0.25,
    replay_gain=0.0,
    sleep_gain=0.0,
)

REPLAY_NO_STRUCT_PARAMS = DynamicsParameters(
    fast_gain=2.0,
    structural_gain=0.0,
    structural_lr=0.0,
    structural_decay=0.0,
    eligibility_decay=0.1,
    translation_decay=0.05,
    context_mismatch_penalty=0.25,
    replay_gain=1.0,
    sleep_gain=1.0,
)

FIXED_ALLOC_PARAMS = DynamicsParameters(
    fast_gain=2.0,
    structural_gain=2.0,
    structural_lr=0.0,
    structural_decay=0.0,
    eligibility_decay=0.1,
    translation_decay=0.05,
    context_mismatch_penalty=0.25,
    replay_gain=0.0,
    sleep_gain=0.0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace(
    trace_id: str,
    branches: list[str],
    weights: dict[str, float],
    context: str | None = None,
) -> EngramTrace:
    """Helper to create an EngramTrace from a set of branch weights.

    Args:
        trace_id: Unique identifier for the trace.
        branches: List of all branch IDs in the simulation.
        weights: Mapping of branch IDs to their allocation weights.
        context: Optional context associated with this trace.

    Returns:
        An initialized EngramTrace object.
    """
    allocation = TraceAllocation(trace_id=trace_id, branch_weights=dict(weights))
    return EngramTrace(trace_id=trace_id, allocation=allocation, context=context)


def _run_encoding(
    sim: CytodendAccessModelSimulator,
    trace: EngramTrace,
    *,
    n_steps: int = 5,
) -> None:
    """Drive the cue corresponding to *trace* for *n_steps* encoding steps.

    Args:
        sim: The simulator instance.
        trace: The engram trace to encode.
        n_steps: Number of simulation steps to apply the cue.
    """
    cue = {b: w * 0.9 for b, w in trace.allocation.branch_weights.items()}
    for _ in range(n_steps):
        sim.apply_cue(cue, context=trace.context)


def _run_consolidation(
    sim: CytodendAccessModelSimulator,
    replay_trace_ids: list[str],
    *,
    n_passes: int = 8,
) -> None:
    """Run offline consolidation passes for the given traces.

    Args:
        sim: The simulator instance.
        replay_trace_ids: List of trace IDs to prioritize during replay.
        n_passes: Number of consolidation windows to run.
    """
    window = ConsolidationWindow(
        window_id="offline",
        modulatory_drive=1.0,
        sleep_drive=1.0,
        replay_trace_ids=tuple(replay_trace_ids),
    )
    for _ in range(n_passes):
        sim.run_consolidation(window)


def compute_model_linking_index(
    sim: CytodendAccessModelSimulator,
    trace_id_a: str,
    trace_id_b: str,
) -> float:
    """Compute the model linking index L(mu1, mu2) = sum_b a_mu1b * a_mu2b * M_b.

    The linking index represents the structural overlap between two engrams,
    weighted by the accessibility of the shared branches.

    Args:
        sim: The simulator instance.
        trace_id_a: ID of the first trace.
        trace_id_b: ID of the second trace.

    Returns:
        The computed linking index, or 0.0 if either trace is unknown.
    """
    trace_a = sim.traces.get(trace_id_a)
    trace_b = sim.traces.get(trace_id_b)
    if trace_a is None or trace_b is None:
        return 0.0

    total = 0.0
    for branch_id, branch in sim.branches.items():
        w_a = trace_a.allocation.branch_weights.get(branch_id, 0.0)
        w_b = trace_b.allocation.branch_weights.get(branch_id, 0.0)
        M_b = branch.structural.accessibility
        total += w_a * w_b * M_b
    return float(total)


# ---------------------------------------------------------------------------
# Core scenario runners
# ---------------------------------------------------------------------------

def run_linking_scenario(
    n_branches: int = 10,
    overlap_weight: float = 0.6,
    n_consolidation_passes: int = 8,
    *,
    params: DynamicsParameters | None = None,
    random_drift: bool = False,
) -> dict[str, float]:
    """Run the canonical two-trace overlap scenario.

    Two traces (mu0, mu1) share a set of branches weighted at *overlap_weight*.
    After encoding and consolidation, the linking index and context margin are
    returned.

    Args:
        n_branches: Number of dendritic branches (typically 10).
        overlap_weight: Allocation weight for the shared (overlap) branches.
        n_consolidation_passes: Number of offline replay passes.
        params: DynamicsParameters to use; defaults to BASE_PARAMS.
        random_drift: If True, disable trace-specific replay (random drift baseline).

    Returns:
        Dictionary with keys:
            - linking_index_model: The structural linking index.
            - context_margin_model: Match vs mismatch recall difference.
            - recall_mu0: Recall strength for trace mu0 in matching context.
            - recall_mu1: Recall strength for trace mu1 in matching context.
    """
    effective_params = params or BASE_PARAMS
    branches = [f"b{i:02d}" for i in range(n_branches)]
    sim = CytodendAccessModelSimulator.from_branch_ids(branches, parameters=effective_params)

    overlap_branches = branches[:2]
    exclusive_mu0 = branches[2:5]
    exclusive_mu1 = branches[5:8]

    def _weights(exclusive: list[str]) -> dict[str, float]:
        w: dict[str, float] = {b: 0.02 for b in branches}
        for b in exclusive:
            w[b] = 0.80
        for b in overlap_branches:
            w[b] = overlap_weight
        return w

    trace_mu0 = _make_trace("mu0", branches, _weights(exclusive_mu0), context="ctx_A")
    trace_mu1 = _make_trace("mu1", branches, _weights(exclusive_mu1), context="ctx_A")

    sim.add_trace(trace_mu0)
    sim.add_trace(trace_mu1)

    _run_encoding(sim, trace_mu0)
    _run_encoding(sim, trace_mu1)

    if random_drift:
        # Simulate random drift: consolidation ignores specific traces
        from copy import deepcopy
        import random
        for branch in sim.branches.values():
            branch.structural.accessibility += random.gauss(0, 0.05)
            branch.structural.accessibility = max(0.0, min(1.0, branch.structural.accessibility))
    else:
        _run_consolidation(sim, ["mu0", "mu1"], n_passes=n_consolidation_passes)

    linking_index = compute_model_linking_index(sim, "mu0", "mu1")

    # Context margin: recall of mu0 in matching vs mismatching context
    cue = {b: trace_mu0.allocation.branch_weights[b] * 0.5 for b in branches}
    supports_match = sim.apply_cue(cue, context="ctx_A")
    recall_match = next((r.expressed_strength for r in supports_match if r.trace_id == "mu0"), 0.0)

    supports_mismatch = sim.apply_cue(cue, context="ctx_B")
    recall_mismatch = next((r.expressed_strength for r in supports_mismatch if r.trace_id == "mu0"), 0.0)

    context_margin = float(recall_match) - float(recall_mismatch)

    return {
        "linking_index_model": linking_index,
        "context_margin_model": context_margin,
        "recall_mu0": float(recall_match),
        "recall_mu1": next(
            (float(r.expressed_strength) for r in supports_match if r.trace_id == "mu1"), 0.0
        ),
    }


def run_baseline_scenarios(
    n_branches: int = 10,
    overlap_weight: float = 0.6,
    n_consolidation_passes: int = 8,
) -> dict[str, dict[str, float]]:
    """Run all five model configurations (full model + 4 comparators).

    Args:
        n_branches: Number of branches per simulator instance.
        overlap_weight: Shared branch allocation weight.
        n_consolidation_passes: Number of consolidation iterations.

    Returns:
        Dictionary keyed by baseline name, each value being the output of
        ``run_linking_scenario`` for that configuration.
    """
    baselines: list[tuple[str, DynamicsParameters, bool]] = [
        ("full_model",            BASE_PARAMS,             False),
        ("fast_context_only",     FAST_CONTEXT_PARAMS,     False),
        ("replay_no_structure",   REPLAY_NO_STRUCT_PARAMS, False),
        ("random_slow_drift",     BASE_PARAMS,             True),
        ("fixed_allocation_only", FIXED_ALLOC_PARAMS,      False),
    ]

    results: dict[str, dict[str, float]] = {}
    for name, params, random_drift in baselines:
        results[name] = run_linking_scenario(
            n_branches=n_branches,
            overlap_weight=overlap_weight,
            n_consolidation_passes=n_consolidation_passes,
            params=params,
            random_drift=random_drift,
        )
    return results


# ---------------------------------------------------------------------------
# Bootstrap uncertainty
# ---------------------------------------------------------------------------

def run_bootstrap_scenarios(
    n_repeats: int = 50,
    n_branches: int = 10,
    overlap_weight: float = 0.6,
    n_consolidation_passes: int = 8,
    *,
    overlap_weight_jitter: float = 0.05,
    consolidation_jitter_frac: float = 0.10,
    ci_alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Run the canonical two-trace linking scenario N times with parameter jitter.

    Each repeat independently perturbs:
    - *overlap_weight* by ± *overlap_weight_jitter*,
    - *n_consolidation_passes* by ± *consolidation_jitter_frac* of its value.

    Args:
        n_repeats: Number of bootstrap iterations.
        n_branches: Number of branches.
        overlap_weight: Base overlap weight.
        n_consolidation_passes: Base consolidation pass count.
        overlap_weight_jitter: Maximum uniform jitter for overlap weight.
        consolidation_jitter_frac: Fraction of passes to use as jitter range.
        ci_alpha: Significance level for confidence intervals (default 0.05 for 95% CI).

    Returns:
        Dictionary keyed by scenario name. Each value contains statistical summaries
        (mean, std, ci_lo, ci_hi) for each metric.
    """
    import numpy as np

    baselines: list[tuple[str, DynamicsParameters, bool]] = [
        ("full_model",            BASE_PARAMS,             False),
        ("fast_context_only",     FAST_CONTEXT_PARAMS,     False),
        ("replay_no_structure",   REPLAY_NO_STRUCT_PARAMS, False),
        ("random_slow_drift",     BASE_PARAMS,             True),
        ("fixed_allocation_only", FIXED_ALLOC_PARAMS,      False),
    ]

    rng = np.random.default_rng(42)
    # Pre-draw jitter vectors
    ow_jitters = rng.uniform(-overlap_weight_jitter, overlap_weight_jitter, size=n_repeats)
    cp_jitters = rng.integers(
        -max(1, int(n_consolidation_passes * consolidation_jitter_frac)),
        max(1, int(n_consolidation_passes * consolidation_jitter_frac)) + 1,
        size=n_repeats,
    )

    all_results: dict[str, dict[str, list[float]]] = {
        name: {"linking_index_model": [], "context_margin_model": [], "recall_mu0": [], "recall_mu1": []}
        for name, _, _ in baselines
    }

    for rep in range(n_repeats):
        ow = float(np.clip(overlap_weight + ow_jitters[rep], 0.1, 0.95))
        cp = max(1, n_consolidation_passes + int(cp_jitters[rep]))
        for name, params, random_drift in baselines:
            result = run_linking_scenario(
                n_branches=n_branches,
                overlap_weight=ow,
                n_consolidation_passes=cp,
                params=params,
                random_drift=random_drift,
            )
            for key, val in result.items():
                all_results[name][key].append(val)

    lo_q = ci_alpha / 2.0
    hi_q = 1.0 - ci_alpha / 2.0

    summary: dict[str, dict[str, float]] = {}
    for name, metric_lists in all_results.items():
        summary[name] = {}
        for metric, vals in metric_lists.items():
            arr = np.array(vals)
            summary[name][f"{metric}_mean"] = float(arr.mean())
            summary[name][f"{metric}_std"] = float(arr.std())
            summary[name][f"{metric}_ci_lo"] = float(np.quantile(arr, lo_q))
            summary[name][f"{metric}_ci_hi"] = float(np.quantile(arr, hi_q))
        summary[name]["n_repeats"] = n_repeats

    return summary
