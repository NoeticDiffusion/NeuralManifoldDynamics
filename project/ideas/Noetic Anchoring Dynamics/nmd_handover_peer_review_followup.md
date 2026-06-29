# NeuralManifoldDynamics LLM — Handover: ds006848 peer-review follow-up

**From:** NoeticDiffusion research assistant  
**Date:** 2026-06-29  
**Dataset:** `ds006848` verbal working-memory (30 subjects, 4 presentation modes)  
**Pipeline run:** `neuralmanifolddynamics_ds006848_20260626_114620`  
**Article:** `articles/embodied_anchoring_follow_up/Rapid item-updating elevates neural manifold dynamics during encoding.typ`

---

## Context

The article is undergoing peer review. Three analysis items were flagged as missing. Two require data you can deliver from the existing run; one requires a new pipeline run. All outputs should be placed in:

```
articles/embodied_anchoring_follow_up/results/peer_review_followup/
```

---

## Task 1 — EEG artifact transparency (clarification, no new computation)

**Reviewer concern (MC6):** "I did not see an equally explicit EEG QC section. For a paper whose primary result is EEG-derived MNPS, reviewers will need: bad channel counts, rejected/interpolated channels, epoch rejection rate, artifact handling."

**What we found:** In `data/raw/neuralmanifolddynamics_ds006848_20260626_114620/sub-001_verbal_wm/qc_summary.json`:
```json
"artifacts": {
  "methods": [],
  "bad_eeg_channels": [],
  "n_bad_eeg_channels": 0
}
```
Both fields are empty for sub-001 (and appear to be empty across subjects). This suggests the pipeline applied **no explicit bad-channel detection, interpolation, or ICA** for this run.

**Requested deliverable:**
1. Confirm whether `bad_eeg_channels` is empty for **all 30 verbal_wm subjects** (loop over all `sub-*_verbal_wm/qc_summary.json` files and report n_bad_eeg_channels per subject).
2. Confirm whether the BIDS `channels.tsv` or `coordsystem.json` from ds006848 was read at ingestion (check `config_ingest_ds006848.yaml` and `run_manifest.json` for any `bad_channels`, `channel_exclusion`, or `interpolation` entries).
3. Produce a short summary table: subject | n_bad_eeg_channels | artifact_methods
4. If no bad-channel step ran, state that explicitly in a `01_eeg_artifact_audit.md` file.

**Output file:** `articles/embodied_anchoring_follow_up/results/peer_review_followup/01_eeg_artifact_audit.md` + `01_bad_channels_per_subject.csv`

---

## Task 2 — Trial-index fatigue analysis

**Reviewer concern:** "Add trial index / practice/fatigue analysis if possible. Add condition-order/fatigue analysis for behavior and MNPS."

**What exists:** The handoff file `data/analysis/ds006848_verbal_wm_20260626/04d_behavioral/trial_level_merged.csv` has columns: `subject, condition, n_trials, m, d, e, m_dot, ..., NCorrect, partialScore`. However, it contains *per-condition aggregates*, not individual trial rows.

The raw per-trial data should be available in the block-native parquets:  
`data/raw/neuralmanifolddynamics_ds006848_20260626_114620/sub-*/sub-*_verbal_wm/block_native_windows.parquet`  
or in the handoff per-trial file:  
`data/analysis/ds006848_verbal_wm_20260626/04b_encoding_phase/per_trial_bin_medians.csv`

**Requested deliverable:**

Using `per_trial_bin_medians.csv` (which has `trial_id`, `condition`, `subject`, and MNPS values):

1. For each subject × condition, compute Spearman correlation between `trial_id` (0–49, encoding order within condition) and `m`, `d`, `NCorrect`.
2. Aggregate across subjects: group median Spearman r and fraction of subjects with |r| > 0.2.
3. Run a Friedman test on whether the per-subject median MNPS across the first 25 vs. last 25 trials differs within each condition.
4. Produce a summary table: condition | median_r_m | median_r_d | median_r_NCorrect | frac_subjects_|r|>0.2

**Interpretation guidance:** If r_m or r_d > 0.2 for fast conditions, it would suggest a learning or fatigue trend. If r_NCorrect < −0.2 for Fast, it suggests progressive fatigue.

**Output files:**
- `articles/embodied_anchoring_follow_up/results/peer_review_followup/02_fatigue_trial_index.csv`
- `articles/embodied_anchoring_follow_up/results/peer_review_followup/02_fatigue_summary.md`

---

## Task 3 — Short-window (2–4 s) MNPS rerun for F2/F3 purity validation

**Reviewer concern (MC2 + S3.4):** "F2/F3 purity would require a 2–4 s window rerun. Fast and Simultaneous have 2.8 s encoding intervals — shorter than half of the 8 s MNPS window, making F2/F3 geometrically impossible."

**Goal:** Re-run the MNDM pipeline on ds006848 with `window_sec: 4` (step: 2 s, 50% overlap). Then re-run the 04b encoding-phase analysis and 04c purity audit on the new output to check whether F2/F3 become achievable and whether the rank ordering (Fast/FastDelay > Simultaneous/Slow) survives.

**Current config reference:**  
`data/raw/neuralmanifolddynamics_ds006848_20260626_114620/config_ingest_ds006848.yaml`  
Key line to change: `window_sec: 8` → `window_sec: 4`

**Requested deliverable:**
1. New pipeline run with `window_sec: 4, step_sec: 2` — all other parameters identical to the 20260626 run.
2. Re-run 04b and 04c on the new parquets, with the same purity filter logic.
3. Report whether F2 (≥50% encoding overlap) and F3 (≥70%) are now achievable for Fast and Simultaneous.
4. Report Friedman results for m and d under F2/F3 for the 4 s window run.
5. Produce a comparison table: filter | window_8s_p_m | window_4s_p_m | window_8s_p_d | window_4s_p_d

**Output files:**
- New run folder: `data/raw/neuralmanifolddynamics_ds006848_YYYYMMDD_HHMMSS/` (4 s windows)
- `articles/embodied_anchoring_follow_up/results/peer_review_followup/03_short_window_robustness.md`
- `articles/embodied_anchoring_follow_up/results/peer_review_followup/03_short_window_filter_comparison.csv`

---

## Priority order

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 1 | EEG artifact audit | Low (JSON loop) | Required for reviewer transparency |
| 2 | Trial-index fatigue | Medium (per_trial_bin_medians.csv) | Addresses behavioral reviewer ask |
| 3 | 4 s window rerun | High (new pipeline run + 04b/04c) | Robustness extension; optional for submission |

Tasks 1 and 2 can be done immediately from existing data. Task 3 requires a new pipeline run.

---

## Key paths

```
Pipeline run (corrected):
  data/raw/neuralmanifolddynamics_ds006848_20260626_114620/

Per-subject QC JSONs:
  data/raw/neuralmanifolddynamics_ds006848_20260626_114620/sub-*/sub-*_verbal_wm/qc_summary.json

Handoff analysis package:
  data/analysis/ds006848_verbal_wm_20260626/

Per-trial MNPS data:
  data/analysis/ds006848_verbal_wm_20260626/04b_encoding_phase/per_trial_bin_medians.csv

Output destination:
  articles/embodied_anchoring_follow_up/results/peer_review_followup/
```
