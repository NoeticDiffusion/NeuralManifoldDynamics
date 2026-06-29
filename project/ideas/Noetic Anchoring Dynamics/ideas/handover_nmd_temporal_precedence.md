# Handover to NeuralManifoldDynamics team
# Request: Temporal precedence analysis + coupling columns
**From:** Robin Langell / NoeticDiffusion  
**Re:** ds003838 — two analytical requests following the embodied anchoring paper

---

## Context

We have completed an empirical paper ("Embodied Anchoring in Noetic Diffusion") using
the ds003838 export from MNDM 2.3/2.4. The paper establishes that HRV-linked anchor
indices (vagal_index, vascular_index, anchor_index) co-vary strongly with task phase
and that manifold trajectory efficiency tracks the same non-monotone load gradient.

A core unresolved question in the paper — and in EAP theory generally — is whether
anchor state changes **precede** or **follow** MNPS manifold changes at task transitions.
This is the difference between EAP as a boundary condition (anchor primes the manifold)
versus EAP as a downstream readout (manifold reorganization drives autonomic response).

We have one suggestive hint: `vagal_index_dot` is significantly negative at listen onset
(d = −0.49, q = 0.001), meaning vagal tone was already decelerating in the 8 s before
the listen segment — the autonomic shift preceded the official onset marker. But 8-second
bins are too coarse for a rigorous precedence test.

---

## Request 1: Cross-lag correlation export

We would like a per-subject cross-lag correlation analysis between anchor indices and
MNPS coordinates across block-native windows.

**What we need:**

For each subject × stage combination, the signed Spearman correlation between:

```
r(anchor_col[t], mnps_col[t + lag])
```

across all block-native windows within that stage, for lags:

```
lag ∈ {−3, −2, −1, 0, +1, +2, +3}   (in units of windows, ~2–6 s each)
```

Anchor columns: `vagal_index`, `sympathetic_index`, `anchor_index`  
MNPS columns: `m`, `d`, `e`

**Output format:**

A CSV with columns:
```
subject_id, stage, anchor_col, mnps_col, lag, spearman_r, n_pairs
```

If this is straightforward to add as a pipeline output (e.g., a `cross_lag_correlation`
sidecar per subject-task), that would be ideal. If it is easier to produce a pooled
group-level summary (median r per lag per anchor×MNPS cell), that is also useful.

**Why this matters:**

If the cross-lag curve peaks at lag > 0 (anchor at time t predicts MNPS at t+1, t+2),
it supports the EAP boundary-condition interpretation (anchor precedes manifold).
If it peaks at lag = 0 (simultaneous), both are co-modulated by task phase.
If it peaks at lag < 0 (MNPS at t predicts anchor at t+1), the autonomic surface follows
neural reorganization.

This is not causal proof, but it is temporal precedence evidence — a meaningful step
beyond stage-level correlations.

---

## Request 2: `coupl_*` columns — what enables them?

In the MNDM 2.4 export for ds003838 (`20260610`), the block-native parquet files contain
twelve inter-network Jacobian coupling columns:

```
coupl_cntr_from_frnt, coupl_cntr_from_par, coupl_cntr_from_temp,
coupl_frnt_from_cntr, coupl_frnt_from_par, coupl_frnt_from_temp,
coupl_par_from_cntr,  coupl_par_from_frnt,  coupl_par_from_temp,
coupl_temp_from_cntr, coupl_temp_from_frnt, coupl_temp_from_par
```

All 12 columns are all-NaN for every subject in our export. The schema is present but
no values are populated.

**Questions:**
1. Is there a configuration flag in `config_ingest_ds003838.yaml` that must be set to
   enable inter-network coupling computation?
2. Is there a minimum number of channels or windows per network that ds003838 fails to
   meet for some subjects?
3. If this requires a separate pipeline run, what would the config change be?

These columns are directly relevant to our article's wishlist item #7 (inter-network
Jacobian off-diagonal coupling) and would allow us to report cross-network anchor
coupling effects beyond the 3D Frobenius summary.

---

## Summary of requests

| # | Request | Format | Priority |
|---|---------|--------|---------|
| 1 | Cross-lag correlation anchor → MNPS per window | CSV per subject or pooled | High |
| 2 | How to enable `coupl_*` columns | Config change or re-run | Medium |

We are happy to run scripts ourselves if you can share the relevant pipeline helper
functions (e.g., how `coupl_*` values are computed and what gating they apply).

Thank you!
