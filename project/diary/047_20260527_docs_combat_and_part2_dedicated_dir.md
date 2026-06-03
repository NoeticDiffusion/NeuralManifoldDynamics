## Session: Docs update + dedicated part2 ComBat path

Date: 2026-05-27

### Question
1) Update README/docs so ComBat and batch-harmonization ideas are clearly documented, including Layer-0 invariance context.
2) Reconfigure I-CARE part2 (`next_140`, 0-12h) to use a dedicated processed directory to avoid part1/part2 feature-table collisions, then run part2 with ComBat.

### Documentation updates
- Updated root README:
  - `README.md`
  - Added section: **Normalization and Batch Harmonization (ComBat pilot)**
  - Documented runtime provenance locations (`run_manifest.json`, `features_snapshot.json`) and Layer-0 invariance framing.

- Updated package README:
  - `mndm/README.md`
  - Added config example and runtime notes for `normalization.method: combat` in `post_features` scope.
  - Clarified participant-metadata batch mapping and fallback behavior.

- Updated output guide:
  - `mndm/Output_variables_guide.md`
  - Added run-level JSON provenance section and explicit `extra.normalization` fields.

### Part2 config hardening (separate output root)
- Updated:
  - `mndm/config/config_ingest_physionet_i-care_2_1_next_140_0_12h_no_regional.yaml`
- Changes:
  - set dedicated `paths.processed_dir`:
    - `E:/Science_Datasets/physionet/processed_next140_0_12h_no_regional_combat`
  - kept ComBat enabled:
    - `normalization.enabled: true`
    - `batch_key: hospital`
    - `covariates: [group, age, sex]`
    - dataset override enabled for `physionet_icare_2_1`

### Runtime actions
- Stopped prior incorrect run that reused part1 feature table in shared processed root.
- Started clean chained run with updated config:
  - `features` -> `summarize`
- Early validation from logs:
  - index built from next_140 received root:
    - `Indexed 1900 files (140 subjects)`
  - file index saved under dedicated processed root:
    - `.../processed_next140_0_12h_no_regional_combat/physionet_icare_2_1/file_index.csv`
  - features workers started:
    - `Using 16 workers ...`

### Note
- This separation removes the core failure mode where part2 summarize accidentally reads part1 `features.parquet` from a shared dataset folder.
