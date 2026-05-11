# 008 — 2026-05-08 — Detector Calibration, Claim Ledger, and Architecture Freeze

## Session context

Following architect approval of the QC findings in [007], the architect gave clear direction:
1. Freeze the overpermissive baseline as a DQF (Detector QC Failure), not as a spindle result.
2. Calibrate YASA before any MNPS interpretation.
3. Prefer agreement-filtered events (F3∩C3) or calibrated single-channel.
4. Only run subject 2 after detector calibration gives plausible rates on sub-1.

This session implemented the claim ledger, calibration sweep, and the canonical calibrated detector.

---

## 1. Claim ledger — `project/claims/sleep_spindle_track_claims.md`

Created a formal claim ledger following the five-category discipline from the `role.mdc` rule:

| Category | Symbol |
|----------|--------|
| Established external result | EVR |
| Internal validated result | IVR |
| Plausible interpretation | PLI |
| Speculative extension | SPE |
| Rejected / falsified | REJ |
| Detector QC failure | DQF |

Six IVR entries established (pipeline, time-axis fix, provenance, staging, exclusion rate).  
Two DQF entries for the overpermissive baseline (rate 73–87/h, temporal overlap 39%).  
One PLI for the MNPS descriptive pattern (pending calibrated detector).  
Two REJ entries (default YASA not suitable; consensus filtering over-filters for this channel pair).  
Two EVR anchors (published spindle rate range; YASA parameter documentation).

---

## 2. YASA calibration sweep

**Target**: Single-channel rate in 5–20/h N2.

### Targeted sweep (12 combos, PSG_F3 only)

Best result without `remove_outliers`: `rel_pow=0.30, corr=0.70, rms=2.5, min_distance=700ms` → 70 events, **21.1/h** (just above range).

Key finding: rms and rel_pow together have limited effect; the `remove_outliers=True` flag is essential.

### Final calibration pass

Three strategies tested:

| Strategy | N | Rate |
|----------|---|------|
| A: best params + `remove_outliers=True` | 54 | **16.3/h** ✓ |
| B: best params + F3∩C3 consensus (0.3s) | 11 | 3.3/h (too low) |
| C: remove_outliers + consensus | 8 | 2.4/h (too low) |

**Strategy A wins**: `remove_outliers=True` is the key parameter that removes feature-space outlier spindle candidates.

Consensus filtering (B, C) over-filters because C3 detects only 47 events (even with calibrated params) vs F3's 70, and the intersection is too sparse.

---

## 3. Calibrated detector — canonical parameter set

**File**: `detect_spindles_yasa_calibrated.py`

**Parameters**:

```python
CALIBRATED_PARAMS = dict(
    freq_sp       = (12, 15),        # narrow sigma band
    freq_broad    = (1, 30),
    duration      = (0.5, 3.0),
    min_distance  = 700,             # ms
    thresh        = {"rel_pow": 0.30, "corr": 0.70, "rms": 3.0},
    remove_outliers = True,          # key parameter
    include       = (2,),            # N2 only
)
```

**Output**: `sub-1_task-Sleep_acq-psg_spindles_yasa_calibrated.csv`

| Metric | Value |
|--------|-------|
| N spindles | 54 |
| Rate | 16.3/h N2 |
| Duration | 0.69 ± 0.12 s |
| Frequency | 12.94 Hz |
| Amplitude | 40.2 µV |
| Stage | N2 only |

**Source tag**: `detector:yasa-0.7.0/calibrated/freq12-15Hz_relp0.30_corr0.70_rms3.0_md700ms_rmoutliers`

This tag will appear in every export row's `event_source` column for full provenance.

---

## 4. Consensus filtering — rejected for this dataset

`build_consensus_events.py` built and tested. The F3∩C3 consensus approach was architecturally sound but produced too few events (< 5/h) for this PSG channel pair. PSG_F3 and PSG_C3 are not truly homologous recording sites, and C3 is substantially less sensitive in this recording.

The consensus approach may be revisited if a more homologous electrode pair is available (e.g., C3+C4, or F3+F4 as bilateral frontal).

---

## 5. Architecture freeze — current state

The pipeline is now at the following state:

```
EDF + staging
  → detect_spindles_yasa_calibrated.py   [54 events, 16.3/h, IVR-006]
  → EventTable CSV (calibrated source tag)
  → event_locked_config.yaml             [profile sleep_spindle_event_locked_v1]
  → align_events_to_windows              [242→54 (fewer events now)]
  → build_matched_controls               [matched N2 controls]
  → build_event_locked_table             [Parquet with full provenance + SHA-256 hash]
  → descriptive_event_vs_control.py      [EXPLORATORY ONLY]
```

The next concrete step is to **re-run the event-locked export with the calibrated 54-event set** and produce a new descriptive summary. Only then, after replication on a second subject, would a formal statistical test be warranted.

---

## Claims status

### Valid as of 2026-05-08

**IVR-006**: YASA calibrated parameters achieve 16.3/h N2 — within plausible range for healthy adults.

**Valid claim**:
> "Using calibrated YASA 0.7.0 (remove_outliers=True, freq_sp=(12,15), rel_pow=0.30, corr=0.70, rms=3.0, min_distance=700ms) on ds005555 sub-1 PSG_F3, the detector produces 54 N2 spindle candidates at 16.3/h, within the published plausible range of 5–20/h."

### Formally rejected

> "Default YASA parameters on this PSG dataset produce reliable spindle annotations." — **REJ-001** (87/h, rejected).

> "F3∩C3 consensus filtering is a viable stricter event set for this recording." — **REJ-002** (too sparse, 3.3/h).

### Pending

Event-locked MNPS run with calibrated 54-event set → will update PLI-001.

---

## Open items for next session

1. **Re-run event-locked export** with `sub-1_task-Sleep_acq-psg_spindles_yasa_calibrated.csv` (54 events).
2. **Produce descriptive event-vs-control summary** on calibrated set — still exploratory.
3. **Second subject**: once above is done, run subject 2 from ds005555 to check reproducibility.
4. **Consider amplitude-stratified analysis**: within the 54 events, high-amplitude events may show a stronger MNPS pattern — but this is SPE-003, not yet testable.
