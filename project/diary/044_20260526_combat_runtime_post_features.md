## Session: v2.2 ComBat runtime in summarize

Date: 2026-05-26

### Question
Wire ComBat into runtime (`summarize`) so we can test whether i-CARE outputs change under site/hospital harmonization, and include a lightweight Layer-0 style outlier damping step.

### Implemented changes
- Runtime ComBat integration in `mndm/src/mndm/pipeline/summary.py`:
  - Added dataset-aware normalization config resolver (`normalization.datasets.<dataset_id>` merge).
  - Added `post_features` normalization hook in dataset summarize flow:
    - `read_features -> subject filter -> qc filter -> combat -> grouping`.
  - Added participant-metadata driven batch/covariate wiring:
    - rows map to subject IDs,
    - subject IDs map to participant sidecar metadata,
    - `batch_key` and `covariates` are resolved case-insensitively with aliases (`site_or_hospital` fallback set).
  - Added chunked ComBat execution to control memory on large epoch tables.
  - Added optional winsorization (`winsorize_quantiles`) before ComBat fit as a Layer-0 outlier damping step.
  - Added robust fallbacks:
    - retry batch-only model if covariate model fails,
    - skip failing chunks in non-strict mode instead of aborting summarize.
  - Added normalization provenance/report payload:
    - `self._normalization_report`,
    - included in `features_snapshot.json` and `run_manifest.json` (`extra.normalization`).

- Config updates (new optional `normalization.combat` sub-block):
  - `mndm/config/config_ingest_common_eeg.yaml`
  - `mndm/config/config_ingest_physionet_i-care_2_1.yaml`
  - `mndm/config/config_ingest_physionet_i-care_2_1_next_140_0_12h_no_regional.yaml`
  - `mndm/config/config_template.yaml`
  - `mndm/config/eeg_config_ingest_template.yaml`
  - Keys:
    - `chunk_size`
    - `min_batch_size`
    - `min_feature_observations`
    - `winsorize_quantiles`

- Test added:
  - `mndm/tests/test_dataset_subject_runner.py::test_dataset_runner_applies_combat_normalization`
  - Verifies that synthetic site offset in `eeg_alpha` is reduced after `_apply_feature_normalization`.

### Validation
- `.venv/Scripts/python -m pytest mndm/tests/test_dataset_subject_runner.py -k "combat_normalization"` -> pass.
- `.venv/Scripts/python -m pytest mndm/tests/test_dataset_subject_runner.py` -> all pass.
- `ReadLints` on edited Python files -> no linter errors.

### Notes / limits
- This is standard ComBat (`neuroCombat`), not longComBat/mixed-effects; repeated-measure structure is not explicitly modeled yet.
- Harmonization currently runs on post-feature tables and affects eligible numeric feature columns.
- Layer-0 items in `combat_ideas.md` (ratio features, 1/f slope, entropy-family invariants, etc.) are mostly feature-contract topics; only outlier damping was added here as a minimal runtime-safe first step.
