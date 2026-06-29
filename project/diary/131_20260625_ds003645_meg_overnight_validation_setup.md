# 131 · 2026-06-25 · ds003645 MEG overnight validation setup

## Session goal

Set up and schedule the overnight MEG–EEG NMD validation run for ds003645
following the specification in `project/ideas/MEG_incorporation/MEG2.md`.

---

## Root cause of A3 bug (missing MEG band contrast)

Identified the exact cause of the missing MEG rows in `face_scrambled_band_contrast.csv`
from session 130.

The H5 files for ds003645 contain **two stacked halves**:
- Rows `0 : n_half` — `.set`-derived epochs (EEG-only; `meg_*` features = NaN)
- Rows `n_half : 2*n_half` — FIF-derived epochs (MEG+EEG; `meg_*` features populated)

The original Cell 26 used `d['feat_raw'][:n_fif]` (where `n_fif = shape[0] // 2`),
selecting the **first** half — the `.set` rows with `meg_*` = NaN.
As a result, every MEG band feature in `mnps_df` was NaN, and the `.dropna()` filter
in Section 7d left zero MEG rows in `face_scrambled_band_contrast.csv`.

**Fix:** Changed `[:n_fif]` → `[n_fif:]` in Cell 26 of
`project/analysis/ds003645_meg_eeg_comparison.ipynb`.
The window_start values are identical in both halves (confirmed from H5 inspection),
so condition labels from Cell 25 are valid for the FIF rows without recomputation.

---

## Files created

### Validation notebook
`project/analysis/ds003645_meg_validation_package.ipynb` — 30 cells

Implements the prioritized test suite from MEG2.md:

| Section | Test | Output CSV |
|---------|------|------------|
| A0 | Raw file audit (n_eeg, n_mag, n_grad per FIF) | `a0_raw_audit.csv` |
| A1 | Timebase audit (window counts, condition distribution) | `a1_timebase_audit.csv` |
| A4 | HDF5 contract test (all required paths present) | `a4_h5_contract.csv` |
| B1 | MAG vs GRAD feature agreement (corr, sign agreement) | `b1_mag_grad_agreement.csv` |
| B3 | MEG spectral sanity (distributions, PCA, gamma outliers) | `b3_spectral_sanity.csv` |
| C1 | Event-response vector agreement (central test, cosine similarity) | `c1_event_response_agreement.csv` |
| C2 | 9D subcoordinate family sign agreement | `c2_family_sign_agreement.csv` |
| C3 | Temporal co-variation with window-shift tolerance (±1, ±2 steps) | `c3_lagged_correlations.csv` |
| C4 | Rank-order condition agreement (Kendall tau) | `c4_rank_agreement.csv` |
| E2 | Temporal circular-shift null | `e2_temporal_shift_null.csv` |
| E3 | Wrong-run pairing null | `e3_wrongrun_null.csv` |
| F3 | Gamma proxy audit (e_m, gamma artifact cleaning) | `f3_gamma_proxy_audit.csv` |
| Dashboard | MEG readiness score (weighted, from MEG2.md formula) | `meg_readiness_score.json` |

The readiness score formula:
```
score = 0.20×contract + 0.20×feature_completeness + 0.20×null_separation
      + 0.15×event_response + 0.10×mag_grad + 0.10×window_robustness + 0.05×jacobian
```
`window_robustness` is set to 0 as a placeholder; updated after D-tests complete.

### Window-multiverse configs (D tests)
- `mndm/config/config_ingest_ds003645_4s.yaml` — 4s/1s step, overlap=0.75, output `processed_4s`
- `mndm/config/config_ingest_ds003645_2s.yaml` — 2s/0.5s step, overlap=0.75, output `processed_2s`

Both verified to load correctly via the chained import system (resolution confirmed via
`core.config_loader`). Each writes to its own `processed_dir` to avoid intermediate JSON
cache collisions with the existing 8s run.

### Orchestration script
`project/analysis/run_overnight.ps1`

Sequential execution plan:
1. Re-run comparison notebook (writes corrected `labeled_manifold_epochs.csv` + `face_scrambled_band_contrast.csv`) — est. 45–75 min
2. Run validation notebook (A0–F3 + dashboard) — est. 30–60 min
3. Run 4s pipeline — est. 2–3 h
4. Run 2s pipeline — est. 2–3 h

Total: ~6–8 hours. Logs written to `project/analysis/logs/`.

---

## Key technical finding

The H5 files have separate `meg_mag_*` and `meg_grad_*` columns (confirmed from
`features_raw/names` inspection — 49 features total including 12 MAG, 12 GRAD,
12 combined MEG, and 13 EEG + auxiliary). This enables the B1 (MAG vs GRAD)
agreement test without any additional feature extraction step.

---

## Pending after overnight run

- Update `window_robustness` in `meg_readiness_score.json` using 4s and 2s pipeline outputs
- If readiness score ≥ 0.80: scale pipeline to all 18 subjects (sub-002 to sub-019)
- If 0.60–0.79: target the weakest subscores for fixes before scaling

## Status

`pending_overnight_run` — all files created, configs verified, script ready to launch.
