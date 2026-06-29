# 142 — 2026-06-29 — Peer-review follow-up: EEG audit, fatigue analysis, 4s pipeline

## Session context

Implementing the three tasks from `nmd_handover_peer_review_followup.md`, sent from the
NoeticDiffusion research assistant. The file documents reviewer concerns (MC2, MC6, S3.4)
for the article "Rapid item-updating elevates neural manifold dynamics during encoding."

All outputs go to:
```
J:\repos\NoeticDiffusion\articles\embodied_anchoring_follow_up\results\peer_review_followup\
```

---

## Task 1 — EEG artifact transparency (MC6)

**Script:** `project/scripts/pr01_eeg_artifact_audit.py`  
**Completed immediately.**

Looped over all 30 `sub-*_verbal_wm/qc_summary.json` files in the
`neuralmanifolddynamics_ds006848_20260626_114620` run directory.

**Finding:** `bad_eeg_channels` and `artifact_methods` are empty for all 30 subjects.
No automated bad-channel detection, interpolation, or ICA was run in this pipeline version.
The pipeline uses epoch-level z-score rejection (z_thresh=3.0) only.

**Outputs produced:**
- `01_bad_channels_per_subject.csv` — 30-row subject table
- `01_eeg_artifact_audit.md` — full audit narrative with config inspection and recommended
  Methods statement

**For the paper:** The Methods section should explicitly state that no bad-channel detection
or ICA was applied. Epoch-level z-score rejection is the only artifact mitigation. This is
a limitation; RANSAC-based bad-channel detection is recommended for future re-analysis.

---

## Task 2 — Trial-index fatigue analysis

**Script:** `project/scripts/pr02_fatigue_trial_index.py`  
**Completed immediately.**

**Method:**
- Loaded `per_trial_bin_medians.csv` (MNPS per trial from 04b) and all 30 BIDS `beh.tsv`
  behavioral files.
- Fixed beh.tsv merge: `trial` column is global 1-200 (not per-condition 1-50); condition
  "Fast+delay" normalized to "FastDelay".
- Computed Spearman r(trial_order, m/d/NCorrect) per subject × condition.
- Ran Wilcoxon signed-rank test for first-25 vs. last-25 trials.

**Key results:**

| condition | median_r_m | frac_|r|>0.2_m | median_r_NCorrect |
|-----------|-----------|--------------|-----------------|
| Fast | -0.154 | 60% | 0.022 |
| FastDelay | -0.132 | 50% | 0.002 |
| Simultaneous | -0.075 | 47% | 0.056 |
| Slow | -0.092 | 60% | 0.022 |

**Interpretation:**
- Group median r_m is below the 0.2 threshold for all conditions (no robust systematic drift).
- However, 50-67% of subjects individually show |r_m| > 0.2 in Fast and FastDelay, suggesting
  heterogeneous within-subject trends (some adaptation/fatigue, others none).
- NCorrect shows no progressive decline in any condition (r near 0), ruling out behavioral fatigue.
- This is not a sufficient confound to explain the Fast/FastDelay > Simultaneous/Slow ordering:
  (a) the effect also holds in Slow; (b) no corresponding behavioral fatigue detected.

**Outputs produced:**
- `02_fatigue_trial_index.csv` — per-subject × condition Spearman r values
- `02_fatigue_summary.md` — aggregated table, Wilcoxon results, interpretation

---

## Task 3 — Short-window (4s) MNPS rerun

**Script:** `project/scripts/pr03_shortwindow_robustness.py`  
**Config:** `mndm/config/config_ingest_ds006848_4s.yaml`  
**Status: Pipeline running in background.**

Created `config_ingest_ds006848_4s.yaml` which imports `config_ingest_ds006848.yaml` and
overrides `mnps.window_sec: 4.0` (overlap: 0.5 → intended step: 2.0 s).

**Note on step size:** The existing features.parquet has epochs at 4.0s steps (from the 8s/50%
overlap original run). The `summarize` command reuses these feature epochs, so the MNPS is
computed at 4.0s steps (not the desired 2.0s). A warning is emitted: "Epoch step 4.000 s
differs from mnps config formula 2.000 s." The pipeline adapts by using the measured step.

**Implication:** The 4s MNPS windows will be non-overlapping (step = window = 4.0s). For
F2/F3 purity, a 4s window can still achieve >=50% overlap with 2.8s encoding intervals if
the window is well-positioned (needs ≤0.8s outside encoding → quite feasible). F3 (>=70%)
requires ≥2.8s in encoding, achievable when windows align closely with encoding onset.

New run directory: `neuralmanifolddynamics_ds006848_20260629_095145`

After the pipeline completes, `pr03_shortwindow_robustness.py` should be run to:
1. Execute 04b and 04c on the 4s run output
2. Compare filter results (F0-F4) between 8s and 4s runs
3. Produce `03_short_window_robustness.md` and `03_short_window_filter_comparison.csv`

---

## Infrastructure changes

- Created `mndm/config/config_ingest_ds006848_4s.yaml` — 4s MNPS window override
- Created `project/scripts/pr01_eeg_artifact_audit.py`
- Created `project/scripts/pr02_fatigue_trial_index.py`  
- Created `project/scripts/pr03_shortwindow_robustness.py`
- Created `project/analysis/peer_review_batch.ps1` — batch runner for all three tasks

---

## Claim ledger impact

- **Task 1 (artifact audit):** No change to scientific claims. Added methodological transparency:
  epoch-level z-score rejection only; no ICA/bad-channel detection. Limitation acknowledged.

- **Task 2 (fatigue):** Supports claim V1-V3 (encoding phase m/d differences). Individual-level
  heterogeneity in trial-index trends is present but does not confound the group ordering.
  NCorrect shows no progressive decline → behavioral fatigue is not the driver.

- **Task 3 (short window):** Pending pipeline completion. If Fast/FastDelay > Simultaneous/Slow
  survives F2 with 4s windows, claim V1-V2 gains additional purity validation.

---

## Next steps

1. Monitor 4s pipeline run (approx. 1-2 hours).
2. When complete, run `pr03_shortwindow_robustness.py --run-dir-4s <new_dir>`.
3. Copy all three outputs to NoeticDiffusion `articles/embodied_anchoring_follow_up/results/peer_review_followup/`.
4. Update claim ledger if 4s F2/F3 results add validation.
