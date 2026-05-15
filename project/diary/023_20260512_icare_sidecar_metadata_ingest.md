# 023 — 2026-05-12 — I-CARE Sidecar Metadata Ingest to H5

## Session goal

Inject per-subject clinical metadata from PhysioNet sidecar text files
(`0332.txt`-style key/value files) into `mndm` H5 outputs, with a reusable
YAML-driven mechanism for similar datasets.

## Implemented

1. Extended participant metadata loading with generic sidecar support:
   - `mndm/src/mndm/pipeline/extractors.py`
   - `load_participant_table(...)` now supports:
     - regular participants tables (`participants.tsv/csv/txt` or configured path),
     - optional `participants.sidecar_files` key/value parsing,
     - merge of table + sidecar metadata (`sidecar_merge`: prefer sidecar or existing).

2. Added sidecar parser and controls (YAML):
   - file discovery via `file_glob` / `file_globs`
   - parser type: `key_value`
   - delimiter: `key_value_separator`
   - key shaping: `key_normalization` (`none`, `lower`, `snake_case`)
   - value casting: `value_cast` (`none`, `auto`)
   - participant-id extraction from sidecars via `subject_id` config
     (`from_key`, `regex`, `strip_prefixes`, `pad`, `prefix`).

3. Added robust participant-id matching during summarize:
   - `mndm/src/mndm/pipeline/summary.py`
   - `participant_meta_for(...)` now tries multiple candidate forms
     (`sub-0332`, `0332`, padded numeric variants), reducing ID-format fragility.

4. Unified prerequisite-check participant loading path:
   - `mndm/src/mndm/prerequisite_check.py`
   - now delegates to shared `load_participant_table(...)` logic (including sidecars).

5. Enabled sidecar metadata for PhysioNet I-CARE config:
   - `mndm/config/config_ingest_physionet_i-care_2_1.yaml`
   - Added `metadata_extraction.datasets.physionet_icare_2_1.participants.sidecar_files`
     block targeting `*/[0-9][0-9][0-9][0-9].txt`.

## Tests

- Ran:
  - `pytest mndm/tests/test_metadata_extraction.py mndm/tests/test_prerequisite_check.py mndm/tests/test_dataset_subject_runner.py`
- Result:
  - `38 passed`
- Added/updated test coverage:
  - sidecar-only participant metadata load,
  - merge between participants table and sidecar metadata.

## Runtime validation

### Base config run (smoke output)

- Features:
  - `mndm.cli features --config mndm/config/config_ingest_physionet_i-care_2_1.yaml --out-dir E:/Science_Datasets/physionet/processed_smoke_wfdb --subject 0332`
- Summarize:
  - `mndm.cli summarize --config mndm/config/config_ingest_physionet_i-care_2_1.yaml --out-dir E:/Science_Datasets/physionet/processed_smoke_wfdb --subject 0332`
- Output run:
  - `E:/Science_Datasets/physionet/processed_smoke_wfdb/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260512_191310`

### Sleep-spindle config run (smoke output)

- Features:
  - `mndm.cli features --config mndm/config/config_ingest_physionet_i-care_2_1_sleep_spindles.yaml --out-dir E:/Science_Datasets/physionet/processed_smoke_wfdb_spindles --subject 0332`
- Summarize:
  - `mndm.cli summarize --config mndm/config/config_ingest_physionet_i-care_2_1_sleep_spindles.yaml --out-dir E:/Science_Datasets/physionet/processed_smoke_wfdb_spindles --subject 0332`
- Output run:
  - `E:/Science_Datasets/physionet/processed_smoke_wfdb_spindles/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260512_191339`

### Metadata injection check

- Confirmed in H5 `participant/row_json` and `summary.json`:
  - `participant_id: sub-0332`
  - `patient: 332`
  - `hospital: D`
  - `age: 68`
  - `sex: Female`
  - `rosc: 60`
  - `ohca: false`
  - `shockable_rhythm: false`
  - `ttm: 33`
  - `outcome: Good`
  - `cpc: 1`
- Provenance:
  - `participant_meta_source.source_format = sidecar_key_value`

## Notes

- A full summarize run against a larger refreshed base output directory
  (`processed_meta_sidecar_base`) progressed far but failed on one later run due
  a pre-existing stratified 9D normalization issue unrelated to sidecar metadata.
- Smoke output runs above completed successfully and verify metadata ingestion
  end-to-end for both requested configs.
