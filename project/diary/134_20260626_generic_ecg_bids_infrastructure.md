# 134 — 2026-06-26 — Generic ECG + BIDS Event-Locking Infrastructure

## Session goal

Implement three infrastructure items identified by the science lead as upstream
NMD issues that must not be locked to ds003838 or ds006848:

- **I2**: ECG QRS polarity auto-detection
- **I1**: Direct BIDS `events.tsv` event-locking source type
- **I3/I4**: HRV superwindow contamination reporting + retrieval-label gate

All implementations are dataset-agnostic — they activate automatically for any
BIDS dataset and require no dataset-specific code paths.

---

## I2 — ECG polarity auto-detection (`ecg.py`)

### Problem
Recordings with negative QRS deflections (inverted ECG lead) cause the
neurokit2/scipy_polarity detector to miss beats or double-detect T-waves.
This was identified as a likely cause of the unrealistic HR values in ds006848.

### Implementation
**`mndm/src/mndm/features/ecg.py`**

Added `_apply_polarity_correction(ecg_1d, sfreq, bandpass_*)` — a
detector-agnostic helper that:
1. Bandpass-filters the signal (5–20 Hz by default)
2. Compares the 99th percentile (positive excursion) against the absolute
   1st percentile (negative excursion), with a 1.2× threshold
3. Returns `(polarity: int, signal_for_detection: ndarray)` where polarity is
   `+1` or `-1`

`_detect_rpeaks` now returns a **3-tuple** `(peaks, polarity_inverted, detector_used)`
instead of just `peaks`. The polarity correction is applied before any detector
runs (neurokit2, scipy_polarity, scipy_abs), so it is fully generic.

Two new columns in every ECG epoch record:
- `ecg_polarity_inverted` — `True` when the signal was inverted before detection
- `ecg_peak_detector` — which detector was used (`neurokit2`, `scipy_polarity`, `scipy_abs`)

### Validation
Synthetic ECG test: positive-QRS signal → polarity=+1; inverted signal →
polarity=-1, detection on 10/10 beats. All existing imports pass.

---

## I1 — Direct BIDS events.tsv event-locking (`event_annotations.py`, `event_locked_runner.py`, `summary.py`)

### Problem
`event_locked` with `kind: csv` + `source_label: derived:task_state_label`
fails when:
- `task_state_label` is not generated (missing `within_run_labels` config), or
- The pre-window crosses an unlabeled recording segment, causing all events to
  be filtered out by transition exclusion logic.

### Implementation

**`mndm/src/mndm/pipeline/event_annotations.py`** — new function
`load_event_table_from_bids_events(path, event_types, trial_type_column,
onset_column, duration_column, exclude_types)`:
- Reads a BIDS `*_events.tsv` directly
- Filters by `event_types` whitelist and `exclude_types` blacklist
- Returns a standard `EventTable` with `event_type=trial_type`, `onset_sec`,
  `duration_sec`
- Logs diagnostics: all available trial_type values, how many events were kept
  vs excluded

**`mndm/src/mndm/pipeline/event_locked_runner.py`** — new dispatch in
`resolve_event_table_for_event_locked` for `source_cfg.kind == "bids_events"`:
- Reads `event_types`, `trial_type_column`, `onset_column`, `duration_column`,
  `exclude_types` from the config
- Calls `load_event_table_from_bids_events`
- No `task_state_label` dependency whatsoever

**`mndm/src/mndm/pipeline/summary.py`** — new dispatch branch
`event_source_kind == "bids_events"` that passes `stage_events_path` (already
resolved per-subject earlier in the summarize loop) directly to
`run_event_locked_export`.

**`mndm/config/config_ingest_ds006848.yaml`** — updated `event_locked.event_source.kind`
from `csv` to `bids_events` with `trial_type_column`, `onset_column`,
`duration_column`, and `exclude_types: [Boundary, Baseline_2s, n/a]`.

### Key design property
Event windows may freely cross unlabeled recording periods. The `stage_filter`
defaults to `[]` (accept any stage code). This matches the expected behaviour
for cognitive datasets where the pre-event baseline legitimately precedes the
first labeled event.

---

## I3/I4 — HRV superwindow contamination reporting + retrieval gate (`ecg.py`)

### Problem
For datasets with short discrete events (e.g., ds006848 digit retrieval),
60-second HRV superwindows span multiple task phases, including retrieval and
motor/speech artefacts. Previously there was no way to know which task labels
were active within each HRV window.

### Implementation

**`_build_task_label_intervals(raw_file_path, total_duration_s, onset_column, trial_type_column)`**  
A new helper function in `ecg.py` that:
1. Locates the companion BIDS `*_events.tsv` from the ECG file path (strips
   modality suffixes: `_eeg`, `_ecg`, `_meg`, `_ieeg`, `_bold`, `_physio`)
2. Applies LVCF (last-value-carried-forward) to build per-label intervals
3. Returns `Dict[str, List[tuple[float, float]]]` — fully generic, works for
   any BIDS recording

**`_compute_hrv_superwindow_metrics`** gains an optional
`task_label_intervals` parameter. When provided, for each HRV superwindow it
computes the overlap fraction with every task label and exports:

| Column | Description |
|---|---|
| `ecg_hrv_dominant_stage_label` | Label with largest overlap in this window |
| `ecg_hrv_dominant_stage_frac` | Fraction of window covered by dominant label |
| `ecg_hrv_n_stage_labels` | Number of labels present (>1% overlap) |
| `ecg_hrv_contains_excluded_label` | True if any `exclude_labels` label is present |

**Config key** (per-dataset, opt-in):
```yaml
features:
  ecg:
    hrv:
      exclude_labels:
        - Digits_Retrieval
```

---

## Documentation

**`mndm/CONFIG_GUIDE.md`** updated with:
- ECG polarity auto-detection section (output columns, override option)
- HRV contamination reporting section (new columns, `exclude_labels` config)
- Direct BIDS event-locking section (`kind: bids_events` full example with
  all configurable keys and a troubleshooting note)

---

## Status

| Item | Status |
|---|---|
| I2 ECG polarity | Done — generic, all detectors |
| I1 BIDS event-locking | Done — `kind: bids_events` implemented |
| I3/I4 HRV contamination | Done — automatic from events.tsv |
| Config documentation | Done — CONFIG_GUIDE.md updated |
| ds006848 config updated | Done — `kind: bids_events` now active |

All changes are dataset-agnostic. Any BIDS recording benefits automatically.
No ds003838- or ds006848-specific code was added.

## Next steps

- Re-run `summarize` for ds006848 to test the new `bids_events` event-locking
  in practice
- Validate `ecg_polarity_inverted` distribution across ds006848 subjects
- Check `ecg_hrv_contains_excluded_label` stats for `Digits_Retrieval` windows
- Consider A1–A3 (04b encoding-phase analysis) once infrastructure is verified
