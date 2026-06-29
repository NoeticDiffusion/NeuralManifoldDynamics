# Bug Report — `block_windows/source_window_index` constant in digit_span runs

**Dataset**: ds003838 (`neuralmanifolddynamics_ds003838_20260607_111755`)  
**Discovered**: 2026-06-07 during embodied anchoring analysis  
**Severity**: High — makes `m`, `d`, `e` columns zero/constant in block_native sidecars for 64/65 subjects  

---

## Summary

In the block_native HDF5 outputs for the **digit_span** task, the dataset
`block_windows/source_window_index` is a constant array — all entries are the
same integer.  Because the block_native CSV/Parquet exporter uses this index to
look up each window's MNPS coordinates in `coords_3d_subject_anchored/values`,
every window receives the same coordinate, and all exported `m`, `d`, `e`
columns are **identical constants** (typically near 0).

The **rest** task runs are **unaffected**: every subject's rest run has a
correctly varying `source_window_index`.

One digit_span subject (**sub-094**) is also unaffected.

---

## Observed behaviour

### Affected — digit_span (64/65 subjects)

```
sub-032_digit_span
  block_windows total:        343
  coords_3d_subject_anchored: (4461, 3)   ← 13× more coords than windows
  source_window_index unique: 1           ← BUG: all 343 windows → index 1847
  source_window_index range:  1847 – 1847
  c[1847]:                    [0.000, 0.000, 0.120]
  c[0]:                       [-0.211,  0.136,  0.409]
  c std (full series):        [0.514,  0.431,  0.586]
  → exported m = 0.0 for all 343 windows

sub-050_digit_span
  block_windows total:        317
  coords_3d_subject_anchored: (5329, 3)   ← 17× more coords than windows
  source_window_index unique: 1           ← BUG: all 317 windows → index 1962
  source_window_index range:  1962 – 1962
  c[1962]:                    [0.000, 0.000, 0.120]
  c std (full series):        [0.542,  0.804,  0.332]
  → exported m = 0.0 for all 317 windows

sub-080_digit_span
  block_windows total:        382
  coords_3d_subject_anchored: (5039, 3)
  source_window_index unique: 1           ← BUG: all 382 windows → index 1938
```

Constant SWI values across subjects: **1847, 1962, 1938** — these appear to be
the epoch index at which the digit_span task begins within the full-session
coordinate array (rest + digit_span concatenated).  The coordinate at that
index happens to be near `[0, 0, 0.12]` for multiple subjects, causing the
block_native CSV to report `m = d = 0` throughout.

### Unaffected — rest task (65/65 subjects)

```
sub-032_rest
  block_windows total:        55
  coords_3d_subject_anchored: (55, 3)   ← 1:1 ratio
  source_window_index range:  0 – 54   ← sequential, correct

sub-094_rest
  block_windows total:        61
  coords_3d_subject_anchored: (61, 3)   ← 1:1 ratio
  source_window_index range:  0 – 60   ← sequential, correct
```

### Unaffected — sub-094 digit_span (only correctly processed digit_span run)

```
sub-094_digit_span
  block_windows total:        341
  coords_3d_subject_anchored: (2091, 3)   ← 6× more coords than windows
  source_window_index unique: 341         ← CORRECT: all different
  source_window_index range:  233 – 2086
  c[233]:                     [1.007, 0.314, 1.123]
  c std (full series):        [0.708, 0.754, 0.806]
  → m varies correctly between -1.011 and +2.446, std = 0.576
```

---

## Root cause hypothesis

The exporter computes `source_window_index` by looking up the epoch index in
the full-session coordinate array that corresponds to each block window's
**center time**.  For the rest task, the coordinate array was stored for
*only the rest segment*, making indices 0..N−1 directly usable.

For the digit_span task, the coordinate array covers *the full EEG session*
(rest + digit_span combined), roughly 13–17× longer.  The lookup formula
appears to resolve all block windows to the **same fixed offset** — most
likely the epoch index at `floor(T_run_start / T_epoch_dt)`, i.e. the epoch
index at the start of the digit_span segment within the full session.  This
value is then used for every window instead of the per-window center time.

Sub-094's digit_span being correct suggests the bug is conditional — possibly
related to a timing reference mismatch that doesn't affect this subject (e.g.
sub-094 may have been processed with the run starting at a different reference
epoch, or processed under a different pipeline version).

---

## Impact

| Item | Count |
|------|-------|
| Affected digit_span runs | 64 / 65 |
| Unaffected rest runs | 65 / 65 |
| Unaffected digit_span runs | 1 / 65 (sub-094) |
| Consequence | `m`, `d`, `e` constant (≈ 0) in block_native CSV/Parquet |
| Data still valid | `coords_3d_subject_anchored/values` in H5 — correct for all subjects |
| Data still valid | `anchor_state/values` in H5 — correct for all subjects |
| Data still valid | `regional_mnps_subjects_111755.csv` — epoch-level aggregates, correct |

**Note**: the `anchor_state` data in H5 is valid but similarly constant in
block_native for most subjects (anchor_index std ≈ 0 in parquet), suggesting
the anchor exporter may have the same source-window lookup bug.

---

## Suggested fix

In the block_native exporter, verify that `source_window_index[i]` is
computed as:

```python
t_center = block_window_center_times[i]          # absolute time, seconds
epoch_idx = np.searchsorted(epoch_times, t_center) - 1  # or round to nearest
source_window_index[i] = epoch_idx
```

and that `epoch_times` is the **full coordinate array's time axis** (not a
per-segment offset).  The rest task works because its coordinate array starts
at t=0 with 1:1 ratio; for digit_span the array starts at an offset into the
full session.

A quick validation check would be to assert `len(np.unique(source_window_index)) > 1`
before writing the block_native outputs.

---

## Workaround

Read MNPS coordinates directly from H5:

```python
import h5py, numpy as np

with h5py.File("sub-032_digit_span.h5", "r") as h:
    coords = np.array(h["coords_3d_subject_anchored/values"])  # (N_epochs, 3)
    times  = np.array(h["block_windows/window_center_sec"])    # (N_windows,)
    # compute epoch times independently, then match by time
```

Alternatively, use `regional_mnps_subjects_111755.csv` for task-level MNPS
aggregates, which are computed correctly for all subjects.

---

## Reproduction

```python
import h5py, numpy as np
from pathlib import Path

run = Path("neuralmanifolddynamics_ds003838_20260607_111755")
for task in ["digit_span", "rest"]:
    d = run / f"sub-032_{task}"
    with h5py.File(d / f"sub-032_{task}.h5", "r") as h:
        swi = np.array(h["block_windows/source_window_index"], dtype=int)
        print(f"{task}: unique SWI values = {len(np.unique(swi))}")
        # Expected: many unique values
        # Actual for digit_span: 1
```
