# 059 - ds006036 stage contract + temporal rerun

Date: 2026-05-28

## Context

Follow-up after ds006036 QC review:
- temporal regional surface did not materialize (channel naming mismatch),
- stage contract needed to include additional protocol/artifact labels,
- full rerun requested (`features` + subject-anchor `summarize` + cohort-anchor `summarize`).

## What was changed

### 1) ds006036 config updates

Updated `mndm/config/config_ingest_ds006036.yaml`:

- Added temporal legacy channel names to ensemble group:
  - `temporal: ["T7", "T8", "TP7", "TP8", "T3", "T4", "T5", "T6"]`
- Expanded stage contract in both:
  - `epoching.datasets.ds006036.sampling.stage_map`
  - `mnps.stage_codebook`
- Added/explicitly mapped:
  - `PHOTO 3Hz -> 57`
  - `PHOTO 7Hz -> 58`
  - `Swallowing -> 63`
  - `speech -> 64`
  - `head movement -> 65`
  - `muscle -> 66`
  - `CAL mode -> 67`
  - `RESET condition -> 68`
  - `PAT 2 CAL -> 69`
  - `PAT 2 EEG -> 70`
  - `PAT Mon-Cz EEG -> 71`
  - `PAT ComAv EEG -> 72`
- Updated stage-blocking expected frequencies:
  - `expected_stage_frequencies_hz: [3, 5, 7, 10, 15, 20, 25, 30]`

### 2) Framework-robust label matching

Updated:
- `mndm/src/mndm/pipeline/summary_events.py`
- `mndm/src/mndm/features/epoch_selection.py`

Both paths now normalize labels by:
- replacing non-printable/control characters with spaces,
- collapsing repeated whitespace,
- lower-casing before stage-map key matching.

This improves mapping stability for noisy raw labels (e.g., control characters embedded in event strings).

### 3) Test coverage

Updated `mndm/tests/test_epoch_selection_point_events.py` with:
- `test_label_epochs_with_stages_normalizes_noisy_event_labels`

Validation:
- `pytest mndm/tests/test_epoch_selection_point_events.py mndm/tests/test_sleep_stage_labels.py`
- Result: 10 passed.

## Rerun commands

Initial rerun in previous output base skipped due cache (`88 already processed`), so full recompute was run in a fresh output base:

- Output base: `E:/Science_Datasets/openneuro/processed_ds006036_subject_then_cohort_anchor_v2`

Commands:

1. Features
   - `python -m mndm.cli features --dataset ds006036 --config mndm/config/config_ingest_ds006036.yaml --out-dir E:/Science_Datasets/openneuro/processed_ds006036_subject_then_cohort_anchor_v2 --n-jobs 6`
2. Summarize (subject-anchor)
   - `python -m mndm.cli summarize --dataset ds006036 --config mndm/config/config_ingest_ds006036.yaml --out-dir E:/Science_Datasets/openneuro/processed_ds006036_subject_then_cohort_anchor_v2 --n-jobs 6`
3. Summarize (cohort-anchor)
   - `python -m mndm.cli summarize --dataset ds006036 --config mndm/config/config_ingest_ds006036.yaml --out-dir E:/Science_Datasets/openneuro/processed_ds006036_subject_then_cohort_anchor_v2 --n-jobs 6 --fit-anchor --anchor-id ds006036_cohort_anchor_v2_3`

## New run outputs

- Subject-anchor run:
  - `.../neuralmanifolddynamics_ds006036_20260528_093207/`
- Cohort-anchor run:
  - `.../neuralmanifolddynamics_ds006036_20260528_093859/`
  - Anchor: `anchors/ds006036_cohort_anchor_v2_3.json`

Both runs produced:
- `run_manifest.json`
- `stage_mapping_qc.json`
- `regional_mnps_subjects_*.csv`
- `regional_block_jacobians_subjects_*.csv`
- per-subject H5 outputs

## Verification notes

- `features.csv` now includes regional groups:
  - `frontal`, `central`, `parietal_occipital`, `temporal`
- `regional_mnps_subjects_093207.csv` contains rows for `region=temporal`.
- Stage mapping QC (both runs) reports:
  - detected raw frequencies: `[3, 5, 7, 10, 15, 20, 25, 30]`
  - missing expected frequencies: `[]`
  - subjects with raw 25Hz: `6`
  - subjects with raw 30Hz: `2`
  - mean `stage_frac_labeled`: `0.5845`
- Newly added stage codes are actively used in mapping:
  - mapped code set includes `57, 58, 63..72` in addition to prior photic/eyes codes.

## Remaining caveat

Top unmapped labels are now dominated by garbled/noise strings with control characters and very low-count leftovers (e.g., malformed `Y...` variants, one-off strings like `TINAGMA`). These appear to be protocol/noise artifacts rather than intended analysis stages.
