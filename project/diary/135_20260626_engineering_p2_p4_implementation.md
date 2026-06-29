# 135 — 2026-06-26: Engineering Pass — Priorities 2–4 Implementation

## Session overview

Implemented three structural engineering improvements in response to the science lead's priority list (from diary 133/134):

- **P1 complete**: Audit table for SampEn independence confirmed all 30 runs pass.
- **P2**: Explicit row provenance (`row_source` group) added to H5 output.
- **P3**: Transform-aware `features_projection_z` export surface added to H5.
- **P4**: `--force-features` CLI flag + `.meta.json` sidecar for cache invalidation.

---

## P1 — SampEn audit table (verified complete)

Produced `p1_sampent_em_audit.csv` in `meg_eeg_comparison/`. All 30 pilot runs pass:

| Gate | Result |
|------|--------|
| SampEn non-constant (std > 0.1) | **PASS** (mean std=1.30, min=0.92) |
| SampEn ≠ PE (|r| < 0.95) | **PASS** (mean r=−0.13, max r=+0.11) |
| e_m non-constant in coords_9d | **PASS** (mean std=1.22, min=0.92) |
| e_m non-nan fraction > 95% | **PASS** (1.000 in all runs) |

e_m_face_minus_scrambled mean = −0.177 (faces show lower sample entropy, consistent with d_l/e_s effects).

---

## P2 — Explicit row provenance in H5

**Problem**: Analysis notebooks used `n_half = n_total // 2` positional slicing to identify FIF (MEG) rows vs `.set` (EEG-only) rows. This was fragile and caused the original A3 bug.

**Change**: Added `/row_source/` group to every H5 output:
- `row_source` (str): `"set_eeg"` | `"fif_meeg"` | `"unknown"`
- `has_meg` (int8): 1 if file was `.fif`, else 0
- `has_eeg` (int8): 1 for all rows
- `has_mag`, `has_grad` (int8): derived from file extension
- `raw_file` (str): basename of the source file
- `source_format` (str): `"neuromag_fif"` | `"eeglab_set"` | `"unknown"`

**Files changed**:
- `mndm/src/mndm/schema.py`: Added `row_source_columns` field to `MNPSPayload`
- `mndm/src/mndm/pipeline/summary.py`: Extracts provenance from `sub_frame["file"]` and populates `row_source_columns`
- `core/src/core/io/h5_writer.py`: Writes `row_source_columns` under `/row_source/` (schema `mndm.row_source.v1`); also improved object-dtype string array handling in `_create_dataset`

**Validation notebook updated**: `load_h5_split()` now uses `_get_fif_mask()` which reads `/row_source/has_meg` when available, with fallback to positional slicing for backward compatibility. Cell 31 (C1 at 4s) similarly updated.

Smoke test confirmed: `row_source[0]=b'set_eeg'`, `row_source[-1]=b'fif_meeg'`, set_rows=121, fif_rows=121.

---

## P3 — Transform-aware `features_projection_z` export surface

**Problem**: `/features_robust_z` applies raw-space robust-z to physical MEG power values (~1e-25 T²/Hz). The MAD is so small relative to the eps=1e-9 floor that z-scores collapse to ~1e-17 — effectively zero. This makes the exported surface useless for MEG spectral features, while `/coords_9d` is correct (it applies log10 first).

**Change**: Added `/features_projection_z/` — a third export surface that applies each feature's configured transform pipeline (e.g., `log10 → robust_z → clip`) before standardisation.

The three surfaces now are:
```
/features_raw              — raw physical values (no transform)
/features_robust_z         — strict raw-space robust-z (correct for EEG/Hjorth)
/features_projection_z     — pipeline-aware robust-z (correct for MEG spectral power)
```

**Files changed**:
- `mndm/src/mndm/projection.py`: Added `_apply_column_pipeline()` helper; extended `build_feature_export_bundle()` to compute and return `projection_z_values`/`projection_z_names`
- `mndm/src/mndm/schema.py`: Added `features_projection_z_values`, `features_projection_z_names` fields to `MNPSPayload`
- `mndm/src/mndm/pipeline/summary.py`: Extracts `projection_z_values` from bundle and passes to payload
- `core/src/core/io/h5_writer.py`: Writes `/features_projection_z/` with `export_transform="projection_z"`

**Verification**:
```
meg_delta features_robust_z std  = 2.73e-17  ← degenerate (near-zero)
meg_delta features_projection_z std = 1.2230  ← informative (after log10)
hjorth_mobility: both surfaces identical (no log10 → same values)
```

**Validation notebook updated**: `load_h5_split()` now loads `projz_fif` and `projz_set` from `features_projection_z`, with fallback to `robz_fif` for old H5s.

---

## P4 — Cache invalidation

**Problem**: Stale intermediate JSONs can silently carry old feature computation (e.g., the `meg_sample_entropy=meg_permutation_entropy` alias bug). Detecting and forcing recomputation was previously a manual process (delete files by hand).

**Changes**:

1. **`--force-features` CLI flag**: Added to `all` and `features` sub-commands. When set, the `already_processed_stems` check is bypassed entirely — all files are re-extracted regardless of cached parquet/intermediate JSON.

   ```
   python -m mndm.cli all --dataset ds003645 --force-features ...
   ```

2. **`.meta.json` sidecar**: `write_intermediate_json()` now accepts `cache_meta` dict and writes `{stem}.meta.json` alongside the main JSON. Fields: `config_hash`, `mndm_version`, `written_at`. Future pipeline code can compare `config_hash` against the current config to detect staleness without `--force-features`.

**Files changed**:
- `mndm/src/mndm/cli.py`: Added `--force-features` argument to `_add_common_args()`; passed through to `cmd_features()` calls
- `mndm/src/mndm/orchestrate.py`: Added `force_features` parameter to `cmd_features()`; wraps `already_processed_stems` population in `if not force_features:` guard; computes `_config_hash` for sidecar
- `mndm/src/mndm/parallel.py`: Updated `write_intermediate_json()` to accept optional `cache_meta` and write sidecar

---

## Next steps (P5)

To get fresh H5s with all new features:

1. **Run re-summarize** (30–45 min):
   ```powershell
   pwsh -File "H:\SourceRepo2\NeuralManifoldDynamics\project\analysis\run_p2_p5_rerun.ps1"
   ```
   This creates a new timestamped run directory with H5s containing `row_source` + `features_projection_z`.

2. **Validate**: The script also runs the updated validation notebook.

3. **Full re-extraction of non-pilot subjects**: Run with `--force-features` to ensure all 18 subjects have fresh `meg_sample_entropy`:
   ```powershell
   python -m mndm.cli all --dataset ds003645 --force-features \
       --config mndm\config\config_ingest_ds003645.yaml \
       --data-dir K:\ds003645 \
       --out-dir E:\Science_Datasets\openneuro\processed
   ```
   Expected: ~3–5 hours for all 18 subjects.

---

## Mandatory gates status

| Gate | Status |
|------|--------|
| H5 contract pass = 1.0 | needs validation run |
| feature completeness = 1.0 | needs validation run |
| row provenance implemented | ✅ DONE (P2) |
| real SampEn/e_m rerun | ✅ DONE (previous session) |
| stale intermediate cache risk | ✅ MITIGATED (P4 --force-features) |
| 4s/2s C1/C2 subject-level | needs validation run |
| transform-aware export issue | ✅ FIXED (P3 features_projection_z) |

Score estimate after re-run: targeting ≥0.80 (production-ready) once validation confirms.

---

## Research claim status

**Validated (engineering)**:
- `row_source` correctly identifies FIF vs `.set` rows via file extension
- `features_projection_z` correctly de-collapses MEG spectral features (std 1e-17 → 1.22)
- `--force-features` bypasses all caching

**Pending (scientific)**:
- Validation notebook results on fresh H5s
- Subject-level C1/C2 aggregation
- Updated readiness score
