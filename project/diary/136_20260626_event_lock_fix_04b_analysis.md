# 136 — 2026-06-26 — ds006848 event-locking fix + 04b encoding-phase analysis

## Session summary

Continued from the infrastructure validation pass (diary 134–135). Two tasks
completed this session:

1. **Resolved the event-locked schema issue** — all `event_type` values were
   empty in the aggregated table.
2. **Implemented and ran `04b_encoding_phase_analysis.py`** — four analyses on
   the full 30-subject cohort.

---

## 1. Event-locked schema fix

### Root cause

`alignment_config_from_profile()` in `event_locked_config.py` defaulted to
`stage_transition_margin_sec = 30.0`. This margin was designed for sleep-spindle
analysis (exclude spindles near sleep-stage transitions). For cognitive BIDS
datasets, the events **are** the state transitions, so every event fell within
30 s of "itself" and was excluded. Result: `n_events_aligned = 0`, all rows
had empty `event_type`.

### Fix

Added a `kind`-aware default:

```python
default_margin = 0.0 if profile.event_source_kind == "bids_events" else 30.0
margin = float(ds_cfg.get("exclude_stage_transition_margin_sec", default_margin))
```

The config key `exclude_stage_transition_margin_sec` can still override this
per dataset if needed. The fix is generic — any future dataset using
`kind: bids_events` will get margin=0 automatically.

### Verification

After re-running `mndm.cli summarize --dataset ds006848`:

```
sub-001 rest:      9/11 events kept  →  36 event rows + 18 control
sub-001 verbalwm: 400/1710 events kept → 1400 event rows + 800 control
event_type breakdown (verbalwm):
  Digits_Retrieval          695
  Retention_Simultaneous    180
  Retention_FastDelay       177
  Retention_Fast            174
  Retention_Slow            174
```

All 30 subjects regenerated; `event_type` is fully populated.

---

## 2. Trial-structure reconstruction from events.tsv

`reconstruct_trials()` identifies each trial's encoding interval by scanning
backward from every `Retention_*` event for the preceding `Encoding_DigitValue_X`
or `Encoding_Set_Simultaneous` cluster. Encoding end = `Retention_*` onset.
Condition is read from the `Retention_*` label.

**Confirmed durations for sub-001:**

| Condition    | Trials | enc_dur (mean ± std) |
|-------------|--------|----------------------|
| Fast         | 50     | 2.80 ± 0.01 s        |
| FastDelay    | 50     | 6.40 ± 0.00 s        |
| Simultaneous | 50     | 2.80 ± 0.00 s        |
| Slow         | 49     | 7.00 ± 0.00 s        |

**Note:** BIDS `events.tsv` structure confirmed — `Baseline_2s` precedes
encoding (not follows). Encoding → Retention → Retrieval is the correct trial
sequence.

---

## 3. 04b analysis results (n = 30 subjects)

### Window resolution context

MNPS windows: 8 s duration, 4 s step. Encoding phases of 2.8–7 s produce 1–2
overlapping windows per trial. Statistical resolution is coarse; conclusions are
at the condition level, not item level.

### Test A — Full encoding episode

Friedman across Simultaneous / Fast / FastDelay / Slow:

| Metric | χ²    | p         |
|--------|-------|-----------|
| m      | 14.52 | **0.0023**|
| d      | 18.68 | **0.0003**|
| e      | 3.40  | 0.334     |
| m_dot  | 5.40  | 0.145     |

**Condition rank (median across subjects):**

| Condition    |  m     |  d     |  e    |
|-------------|--------|--------|-------|
| FastDelay   | 0.460  | 0.405  | 0.115 |
| Fast        | 0.411  | 0.335  | 0.180 |
| Simultaneous| 0.256  | 0.142  | 0.032 |
| Slow        | 0.201  | 0.075  | 0.025 |

**Significant BH-FDR pairwise (q < 0.05):**

For **m**: Simultaneous < {Fast, FastDelay}; FastDelay > Slow.
For **d**: {Fast, FastDelay} > Slow; Simultaneous < {Fast, FastDelay}.

### Test B — Common 2.8 s window (fair cross-mode comparison)

| Metric | χ²    | p         |
|--------|-------|-----------|
| m      | 17.12 | **0.0007**|
| d      | 20.12 | **0.0002**|
| m_dot  | 20.92 | **0.0001**|
| e_dot  |  8.04 | **0.045** |

The fast-presentation modes (Fast, FastDelay) show higher `m` and `d` even in
the first 2.8 s of encoding, suggesting the presentation-rate effect appears in
early trajectory features rather than requiring full-episode integration.

### Test C — Normalised encoding bins (exploratory)

Bin coverage per condition (all subjects, all trials):

| Condition    | Trials/bin (range) |
|-------------|---------------------|
| Fast         | 148–152            |
| FastDelay    | 188–194            |
| Simultaneous | 141–155            |
| Slow         | 189–192            |

Coverage is adequate but many trials share the same window across bins (because
the window is longer than the bin). Phase-bin Friedman tests require ≥ 3
subjects with data in all bins — results were computable but no phase × condition
effects reached significance in this run (needs full per-bin contrast table
review separately).

### Retrieval-by-prior-mode

Friedman: only `e_dot` (χ²=9.64, p=0.022) shows a significant overall effect.

**Median m / d / e at retrieval by prior mode:**

| Prior mode   |  m      |  d     |  e     |
|-------------|---------|--------|--------|
| Fast        |  0.022  |  0.028 |  0.003 |
| FastDelay   | -0.036  |  0.011 | -0.002 |
| Simultaneous| -0.024  |  0.087 | -0.022 |
| Slow        | -0.101  |  0.005 | -0.058 |

Pattern: prior-Slow retrieval has the lowest `m` and `e`, but differences are
small and largely non-significant. The `e_dot` finding is marginal.

---

## 4. Interpretation (science-lead validated framing)

**Status: Internal validated result** (n=30, Friedman + BH-FDR, survives F0/F1/F4
purity filters; F2/F3 geometrically inapplicable for 2.8 s encoding in 8 s windows).

**What this shows:**

- Rapid item-updating conditions (**Fast** and **FastDelay**) show higher `m`
  and `d` than Simultaneous or Slow-paced encoding. The effect is not a simple
  "sequential > simultaneous" story — Slow is also sequential yet shows the
  lowest `m` and `d`. The distinguishing factor is presentation *speed*, not
  sequentiality.
- The Fast–Simultaneous contrast is significant for both `m` and `d` despite
  identical total encoding durations (2.8 s): this rules out a pure duration
  effect. Rapid digit-by-digit delivery, not longer encoding time, drives the
  elevated manifold mobility.
- Maintenance-window MNPS (`m`, `d`, `e`) remains null across prior presentation
  modes. The effect is encoding-phase-specific, not a stable maintenance-state
  separation that persists after encoding concludes.

**What this does not show (explicit caveats):**

- The 8 s MNPS window prevents item-level interpretation: individual digit
  events (≈400 ms) are unresolvable at this window size.
- Purity filters ≥50% / ≥70% (F2/F3) are geometrically impossible for
  Fast/Simultaneous at 8 s windows — these conditions simply cannot dominate
  a window longer than their full encoding duration (2.8 s).
- The FastDelay encode–rehearse microcycle remains **speculative**: 8 s windows
  cannot resolve 400–1000 ms item/rehearsal cycles. Requires shorter windowing
  or a dedicated item-locked analysis.
- WM-phase HRV/anchor claims are gated (87.7 % contamination confirmed in
  prior session).

**Plausible interpretation (not yet proven):**

- Rapid item updating increases manifold mobility and diffusivity during
  encoding. This may reflect a more dynamically engaged, rapidly reconfiguring
  cortical state compared to the slower-paced or simultaneous presentation
  modes.

---

## 5. Files produced

| Path | Contents |
|------|----------|
| `mndm/src/mndm/pipeline/event_locked_config.py` | `bids_events` margin=0 fix |
| `project/scripts/04b_encoding_phase_analysis.py` | Full 04b script |
| `J:/.../04b_encoding_phase/A_full_encoding/` | Friedman, pairwise, medians CSVs |
| `J:/.../04b_encoding_phase/B_28s_window/` | Common-duration analysis |
| `J:/.../04b_encoding_phase/C_normalized_bins/` | Phase-bin coverage + Friedman |
| `J:/.../04b_encoding_phase/D_retrieval_by_prior_mode/` | Retrieval stratified by encoding mode |
| `logs/ds006848_summarize_final.txt` | Full re-summarize log (all 30 subjects) |

---

## 6. Open items (updated after science-lead review)

- **Test C FastDelay micro-cycle**: remains speculative; re-examine per-bin
  medians for FastDelay, but do not claim temporal resolution without shorter
  windows.
- **Behavioral condition review**: accuracy, partial score, RT, serial-position
  error — test whether m/d tracks encoding success vs. presentation speed.
- **Subject-level robustness**: leave-one-out Friedman, bootstrap CIs,
  spaghetti plots, F1/F4 pairwise effect sizes.
- **Classical EEG comparator**: frontal theta, alpha engagement, EEG complexity.
- **Optional**: ds006848 4 s/2 s window rerun for F2/F3 purity and
  temporal-isolation validation.
