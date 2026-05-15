# 021 — 2026-05-12 — I-CARE Pseudo-Stage Proxy (N2-like)

## Session goal

Add a biologically motivated pseudo-stage fallback for I-CARE so stage-aware
sleep-spindle analyses can run even when explicit sleep staging labels are absent.

## Implemented

1. Added pseudo-stage derivation in pipeline extractors:
   - `mndm/src/mndm/pipeline/extractors.py`
   - New function: `derive_pseudo_stage_array(...)`
   - Resolves config from:
     - `pseudo_stage` (global defaults)
     - `pseudo_stage.datasets.<dataset_id>` (per-dataset overrides)
   - Combines criteria:
     - spindle-density proxy (explicit spindle columns if available)
     - sigma-like activity proxy
     - low EMG/high-frequency proxy
   - Supports fallback behavior:
     - if spindle columns are missing, uses sigma column as spindle-burst source
   - Outputs:
     - stage array (e.g. `N2_PROXY -> code 2`)
     - diagnostic metadata (`status`, used columns, labeled fraction, criteria used)

2. Wired pseudo-stage into summarize flow:
   - `mndm/src/mndm/pipeline/summary.py`
   - Execution order for stage labels now:
     1) explicit stage columns in features
     2) inferred BIDS events stages
     3) within-run labels
     4) **pseudo-stage proxy fallback**
   - Adds pseudo-stage provenance fields to H5 attrs/manifest, including:
     - `pseudo_stage_status`
     - `pseudo_stage_label`
     - `pseudo_stage_code`
     - `pseudo_stage_labeled_fraction`
     - source columns used for sigma/emg/spindle proxies

3. Updated I-CARE sleep-spindle config to use proxy stages:
   - `mndm/config/config_ingest_physionet_i-care_2_1_sleep_spindles.yaml`
   - Added:
     - `pseudo_stage.datasets.physionet_icare_2_1` block
     - `mnps.stage_codebook.N2_PROXY: 2`
   - Switched event-locked stage target:
     - `stage_filter: ["N2_PROXY"]`
   - Added channel-aware proxy settings for currently observed I-CARE features.

4. Added tests:
   - `mndm/tests/test_pseudo_stage_proxy.py`
   - Verifies:
     - N2 proxy labeling works on synthetic spindle/sigma/emg signals
     - graceful no-op when required proxy columns are missing

## Validation

- Unit tests:
  - `pytest mndm/tests/test_pseudo_stage_proxy.py -q` → passed
- Preflight:
  - `mndm.cli prerequisite-check` with sleep-spindle I-CARE config → OK
- Summarize smoke run:
  - `mndm.cli summarize --subject 0332 ...config_ingest_physionet_i-care_2_1_sleep_spindles.yaml`
  - H5 attrs confirmed:
    - `stage_source = pseudo_stage_proxy`
    - `pseudo_stage_status = ok`
    - `pseudo_stage_label = N2_PROXY`
  - Stage labels present in H5:
    - unique values `[-1, 2]`

## Notes

- This is an **internal proxy**, not ground-truth sleep staging.
- Interpretation should remain cautious and be reported as a derived heuristic.
