# 138 — 2026-06-26 — 04b claim upgrade + next-analysis plan (science-lead response)

## Context

Science lead reviewed the 04b encoding-phase analysis (diary 136) and 04c purity
audit (diary 137). Response confirms the upgrade path and refines the interpretation
framing. This entry records the decisions and plans the next four analysis tracks.

---

## 1. Claim ledger — updated

### Internal validated results

- **Encoding-phase m and d differ across presentation modes (ds006848, n=30).**
  Friedman χ²(m)=14.52 p=0.0023, χ²(d)=18.68 p=0.0003. BH-FDR pairwise
  contrasts survive. Replicated across F0, F1, F4 purity filters.

- **Fast/FastDelay > Simultaneous/Slow for m and d survives applicable purity
  filters (F0/F1/F4).**
  F2/F3 (≥50%/≥70% encoding overlap) are geometrically impossible for the
  2.8 s Fast/Simultaneous conditions with 8 s windows — this is a hard
  physical constraint, not a test failure.

- **Maintenance-window MNPS (m, d, e) remains null across prior presentation
  modes.**
  The encoding-phase effect is encoding-specific, not a stable post-encoding
  maintenance-state separation.

- **WM-phase HRV remains gated due to 60 s superwindow contamination (87.7%).**
  No HRV-MNPS claims for the WM phase until shorter anchor windows are
  implemented or a different dataset is used.

### Plausible interpretation (not yet proven)

- Rapid item updating increases manifold mobility (`m`) and diffusivity (`d`)
  during encoding. This may reflect a more dynamically engaged, rapidly
  reconfiguring cortical state elicited by fast sequential digit delivery.

### Speculative (label explicitly if discussed)

- FastDelay encode–rehearse microcycles: the 400–1000 ms item/gap structure
  is unresolvable at 8 s windows. Speculative until shorter-window analysis
  or item-locked event analysis is run.
- Item-level MNPS dynamics: cannot be claimed from current window resolution.

---

## 2. Interpretation framing (science-lead correction applied)

**Do not use:** "sequential > simultaneous" — Slow is also sequential and
shows the lowest m/d.

**Use instead:** "Rapid item-updating conditions (Fast and FastDelay) show
higher m/d than simultaneous or slow-paced encoding."

The key variable is presentation *speed*, not sequentiality per se. The
Fast–Simultaneous contrast (identical duration, 2.8 s, different m/d) makes
a duration-artefact explanation untenable.

---

## 3. Next analysis tracks

### Track 1 — Behavioral condition review (priority)

Goal: determine whether the m/d elevation co-varies with encoding success,
task difficulty, or purely with presentation speed.

Variables to extract from `participants.tsv`, `events.tsv`, or behavioral
summary files:
- Accuracy (% correct digit recall)
- Partial score (digit-position scoring)
- Response time (time to enter recall sequence)
- Serial-position error (if reconstructable from recall sequence)

Analysis plan:
1. Compute per-subject, per-condition behavioral aggregates.
2. Correlate (Spearman, n=30) each behavioral metric with median m/d per
   condition.
3. Mixed model or rank test: does m/d track difficulty rank
   (Fast~Simult < FastDelay < Slow by difficulty) or speed rank
   (Slow < Simult < Fast < FastDelay by speed)?
4. Report: does behavioral performance explain away the m/d effect when
   controlled, or is it independent?

### Track 2 — Subject-level robustness

Goal: verify the encoding-phase m/d effect is not driven by a small subset
of subjects.

1. **Leave-one-out (LOO) Friedman**: rerun Friedman for all 30 leave-one-out
   folds; record χ², p-value, and which subject removal most attenuates.
2. **Bootstrap CIs**: for each condition's median m and d, bootstrap n=1000
   subject samples (with replacement from the 30-subject pool); report 95% CI.
3. **Spaghetti plots**: per-subject condition-rank lines for m and d; visualise
   whether the rank ordering (Fast/FastDelay > Simult/Slow) is consistent or
   bimodal.
4. **F1/F4 pairwise effect sizes**: Cohen's d or rank-biserial r for the key
   pairwise contrasts (Fast vs. Simult, FastDelay vs. Slow) under both F1 and
   F4 filters.

### Track 3 — Classical EEG comparator

Goal: benchmark the MNPS m/d effect against classical EEG measures.

Measures to compute (from existing features or new extraction):
- **Frontal theta power** (4–8 Hz, Fz / frontal cluster)
- **Alpha engagement** (8–13 Hz suppression during encoding vs. baseline)
- **EEG complexity** (Hjorth complexity, available from existing features)

Analysis plan:
1. Extract mean per-condition frontal theta, alpha, and complexity from the
   ds006848 H5 files (using encoding-window selection as in 04b).
2. Run Friedman across conditions for each classical measure.
3. Compare: do classical measures show the same Fast/FastDelay > Simult/Slow
   pattern? If yes: convergent validity. If no: MNPS captures something
   distinct.

### Track 4 — Optional: shorter-window ds006848 rerun

Goal: enable F2/F3 purity tests for Fast/Simultaneous and provide
temporal-isolation validation for the FastDelay microcycle hypothesis.

Window candidates:
- **4 s window / 2 s step**: allows ≥50% purity for Fast/Simult (enc_dur
  2.8 s → 70% max overlap in a 4 s window).
- **2 s window / 1 s step**: allows ≥70% purity and approaches per-item
  resolution.

Decision gate: run only if Tracks 1–2 suggest the 8 s result is robust and
not behaviorally confounded. This is the most expensive track (full pipeline
rerun, ~45 min per window size).

---

## 4. Files to produce

| Track | Script | Key outputs |
|-------|--------|-------------|
| T1    | `04d_behavioral_review.py` | `behavioral_summary.csv`, correlation table, mixed model output |
| T2    | `04e_subject_robustness.py` | LOO Friedman table, bootstrap CI CSV, spaghetti plots |
| T3    | `04f_eeg_comparator.py` | Classical feature Friedman table, convergence summary |
| T4    | `05_short_window_rerun.ps1` | 4s H5 files, updated purity audit |

---

## 5. Readiness checkpoint after Track 2

If LOO and bootstrap confirm the effect is robust (no single-subject driver,
CIs non-overlapping between Fast/FastDelay and Simult/Slow), the 04b finding
can be considered ready for article-section drafting.

If a strong outlier is identified, investigate that subject's behavioral
record and exclude with justification before proceeding.
