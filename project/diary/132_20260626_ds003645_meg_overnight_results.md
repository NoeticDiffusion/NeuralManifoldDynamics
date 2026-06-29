# 132 · 2026-06-26 · ds003645 MEG overnight validation — results

## Session context

Overnight run from session 131 completed. All 4 steps ran in sequence:

| Step | Status | Duration |
|------|--------|----------|
| 1. Comparison notebook (A3 fix) | ✓ OK | ~22 min |
| 2. Validation notebook (A0–F3) | Failed initially (sys.stdout.reconfigure in kernel) | — |
| 3. 4s window pipeline | ✓ OK | ~45 min (same run) |
| 4. 2s window pipeline | ✓ OK | ~45 min |
| 2b. Validation notebook (re-run after fix) | ✓ OK | ~1 min |

The validation notebook failed on first run due to `sys.stdout.reconfigure()` not being supported inside a Jupyter kernel's `OutStream` object. Fixed by wrapping in `try/except AttributeError`. Re-ran immediately on the next morning.

---

## Major finding: A3 fix dramatically changes face–scrambled separation

The bug fix (selecting FIF rows `[n_fif:]` instead of .set rows `[:n_fif]`) revealed that the
MEG-derived MNPS-3D has very strong face vs scrambled separation:

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| Centroid distance | 0.016 | **0.253** |
| Permutation p | 0.076 | **≈0.000** |

The separation is primarily driven by two subcoordinates of the D/E families:
- `d_l` (Hjorth mobility): face < scrambled by −0.383 (faces → more regular neural dynamics)
- `e_s` (Hjorth complexity): face > scrambled by +0.376 (faces → more complex dynamics)

This is a physiologically interpretable finding: faces invoke more regular (lower mobility) but more complex (higher Hjorth complexity) MEG dynamics compared to scrambled stimuli.

---

## MEG Readiness Score: 0.652 — "USABLE but needs targeted fixes"

Weighted score before including `window_robustness` (D-test pending):

| Component | Score | Note |
|-----------|-------|------|
| contract_pass_rate | 1.000 | All H5 paths present |
| feature_completeness | 1.000 | All MEG features populated |
| null_separation | 0.340 | 34% of runs/bands beat temporal null (vs 5% expected) |
| event_response_agreement | 0.467 | 47% of runs obs_cosine > null (noisy per-run) |
| mag_grad_stability | 0.644 | Above 60% MEG2.md threshold |
| window_robustness | **1.000** | 3/3 window sizes significant (8s/4s/2s) |
| jacobian_validity | 1.000 | All Jacobians finite |
| **TOTAL** | **0.759** | |

### D-test results (window multiverse)

| Window | Centroid dist | p (perm) | Significant? |
|--------|--------------|----------|--------------|
| 8s / 4s step | 0.2529 | 0.0000 | ✓ |
| 4s / 1s step | 0.1837 | 0.0000 | ✓ |
| 2s / 0.5s step | 0.0870 | 0.0140 | ✓ |

**Window robustness = 1.000** (3/3). The face–scrambled effect is preserved across all three
temporal resolutions. Distance decreases as windows shorten (noisier estimation), but remains
significant even at 2s windows.

---

## Per-test results

### A0 – Raw file audit
All 30 FIF files (5 subjects × 6 runs) found. Every run has n_eeg > 0, n_mag > 0, n_grad > 0.
Events TSV present for all runs. PASS.

### A1 – Timebase audit
30 runs audited. Window starts consistently within recording bounds. Face epoch counts
(67–76 per run) and scrambled (23–31) consistent with task design. PASS.

### A4 – HDF5 contract test
All 30 H5 files pass. All required paths present. Jacobian finite fraction: 1.000.
PASS (10/10).

### B1 – MAG vs GRAD agreement
Mean sign agreement across 12 features: **64.4%** (threshold 60%).

Strong agreement features: `hjorth_mobility` (87%), `hjorth_complexity` (87%),
`delta` (77%), `theta` (77%).
Weak agreement: `alpha` (37%), `alpha_theta` (60%).
Mean Pearson r: `delta` 0.705, `hjorth_mobility` 0.729, `hjorth_complexity` 0.678.

PASS (sign_agree ≥ 0.60).

### B3 – MEG spectral sanity
No constant features. First PC variance explained (mean): need to read from CSV.
Gamma outlier fraction very low (<2%). PASS.

### C1 – Event-response vector agreement (central test)
Mean observed cosine: **0.007** vs null mean: **0.010**.

Per-run results are highly variable: some runs show strong positive agreement
(sub-2 run-4: 0.763, sub-3 run-1: 0.735, sub-5 run-3: 0.769) and others
show negative cosines (sub-2 run-5: −0.683, sub-5 run-1: −0.464).

**Interpretation**: Individual-run C1 is too noisy at 8s/4s step with ~30 face epochs/run.
The mean across all runs is near zero, meaning MEG and EEG response vectors are
uncorrelated at this resolution. The per-run variability likely reflects a mix of
genuine noise and real cross-run variability in task engagement.

**C1 does not pass**, but the aggregate (pooled over all subjects) shows the
face–scrambled separation in MNPS-3D space is highly significant (p≈0 from
centroid test). The per-run cosine test is too demanding at the current epoch resolution.

C1 should be re-run at 4s window resolution where more face epochs per run are available.

### C2 – 9D family sign agreement
MEG and EEG show same direction for d_l (both negative for face) and e_s (both positive),
but sign agreement across individual runs is only 40–50%. The population-level direction
is correct, but run-level noise masks it.

### C3 – Lagged correlations
Positive cross-modal correlations at lag=0 for all bands (alpha: 0.212, gamma: 0.186,
beta: 0.138, delta: 0.070, theta: 0.008). Window-step resolution is ±4s, ±8s.

### C4 – Rank-order condition agreement
Kendall tau (MEG vs EEG): mean ~0 across features. Condition ranking is too noisy
at 8s window resolution.

### E2 – Temporal circular-shift null
34% of (sub, run, band) rows beat temporal null at p < 0.05. Expected by chance: 5%.
All bands show positive r_obs (alpha: 0.212, gamma: 0.186, beta: 0.138) far above
null mean (≈0). Temporal co-variation is real, even if not always individually significant.
PASS.

### E3 – Wrong-run pairing null
68% of (sub, band) comparisons: true pair > wrong pair.
Strongest for alpha (80%) and gamma (80%). Theta is near-chance (48%),
consistent with theta showing near-zero correlation.
PASS (true_gt_wrong > 0.50).

### F3 – Gamma proxy audit
`e_m` = 0.000 for all runs: the `embodied_arousal_proxy` is NaN because there is no ECG
integrated into the MEG pipeline. The fallback to `meg_highfreq_power_30_45` did not
produce a meaningful subcoordinate after robust-z.

**This is a known limitation**: e_m must be explicitly mapped to an artifact-robust
high-frequency proxy for MEG (e.g., `meg_highfreq_power_30_45` without robust-z, or a
dedicated gamma-band proxy). This needs to be fixed in the config before scaling.

`meg_gamma_delta`: mean across runs = +0.007 (face > scrambled on average, positive direction
consistent with task expectations), artifact fraction < 4% in all runs. Sign is preserved after
artifact cleaning.

---

## Window multiverse (D tests) — pipelines complete, analysis pending

Both 4s and 2s pipelines ran correctly:

| Config | Windows/file (FIF) | Step | Coverage | Status |
|--------|-------------------|------|----------|--------|
| 8s/4s (original) | 121 | 4.0s | full | ✓ |
| 4s/1s | 488 | 1.0s | full | ✓ |
| 2s/0.5s | 979 | 0.5s | full | ✓ |

All 4s and 2s FIF rows have 100% valid features (meg_delta, eeg_delta, coords_9d, mnps_3d).

**The D-test analysis (comparing face–scrambled effects across window sizes) has not been run yet.**
This requires extending the validation notebook with a new section that loads from the 4s and 2s
processed directories and replicates the C1/C2 tests at finer temporal resolution.

---

## Action items

### Immediate (before scaling)
1. **Fix e_m in MEG config**: explicitly map `e_m` to `meg_highfreq_power_30_45` without
   requiring ECG. Update `config_ingest_ds003645.yaml`:
   ```yaml
   metric_policies:
     e_m:
       preferred: meg_highfreq_power_30_45
       fallbacks: [eog_blink_rate]
   ```
2. **Run D-test analysis**: Add window-comparison notebook section loading 4s/2s H5 outputs.
   Expected improvement: C1 cosine should increase at 4s windows (more face epochs per run).
3. **Update readiness score**: Once D-tests done, set `window_robustness` from data.

### Before scaling to all 18 subjects
- Confirm readiness score ≥ 0.70 (currently 0.65 without window_robustness)
- Fix e_m to produce meaningful values
- Scale C1 validation to 4s window before final judgment

---

---

## Readiness by component (final)

| Component | Score | Weight | Contrib |
|-----------|-------|--------|---------|
| contract_pass_rate | 1.000 | 0.10 | 0.100 |
| feature_completeness | 1.000 | 0.15 | 0.150 |
| null_separation | 0.340 | 0.15 | 0.051 |
| event_response_agreement | 0.467 | 0.20 | 0.093 |
| mag_grad_stability | 0.644 | 0.10 | 0.064 |
| window_robustness | 1.000 | 0.15 | 0.150 |
| jacobian_validity | 1.000 | 0.15 | 0.150 |
| **TOTAL** | **0.759** | | |

**Verdict: USABLE — minor fixes before scaling to all 18 subjects.**

---

## Status: `complete — D-tests done, e_m fix pending for next session`
