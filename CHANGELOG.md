# Changelog

---

## v2.5.0 — Multimodal ingest expansion, phase-aware features, and contract hardening

This release consolidates the post-v2.4.0 working-tree additions into the next
versioned measurement-contract line. The detailed MEG, HRV v0.1, block-native
v2, geometry-contract, anchoring, and validation notes retained under v2.4.0
below remain part of the cumulative contract history. This section records the
additional functionality promoted into the v2.5.0 release.

### Major additions

**BDF / Figshare infant EEG**

- Added BDF adapters and privacy-safe metadata extraction for Figshare infant
  EEG workflows.
- Added BAD_ masking, coverage-aware cohort-anchor export, and a repair for
  connectivity features becoming all-NaN after masked segments.
- The full internal run processed 71 BDF files, retained 5,273 epochs, and
  produced 70 coverage-passing HDF5 summaries. These are ingest/QC results;
  infant behavioural condition labels were not established by the available
  Status field.

**NWB / Neuropixels / ephys and LFP**

- Added DANDI/NWB ElectricalSeries and Units-to-rate paths with probe discovery,
  streaming QC, state/stimulation annotations, and explicit geometry limits.
- Added Neuropixels ephys feature extraction and smoke validation for DANDI
  000006.
- Added LFP channel selection, contact/reference sensitivity, and state-aware
  QC for DANDI 000458. These are one-session descriptive foundations, not
  cross-probe geometric or circuit-level validation.

**Phase anchor and sleep-EAP extensions**

- Added optional cardiac/respiratory phase, RR, HR, respiratory-rate, inhale
  fraction, and HEP-related feature extraction through `phase_anchor`.
- Added the sleep-EAP phase-2 contract and associated quality/provenance
  surfaces.
- These features are now release-bound as optional v2.5.0 capabilities; their
  dataset-scale scientific interpretation remains modality- and cohort-
  dependent.

**Contract, provenance, and QC hardening**

- Added `/epoch_id` as an explicit window join key where available.
- Added stricter simultaneous-MEEG row-lineage checks and expanded provenance.
- Added robust-z degenerate-scale safeguards and documented the
  `degenerate_scale_policy = "nan"` compatibility boundary.
- Added configuration overlay replacement semantics and additional source
  adapters, tests, and QC artifacts.

### Validation and claim boundaries

| Dataset or stream | Evidence in v2.5.0 | Claim ceiling |
|---|---|---|
| ds003838 | 130 completed HDF5 exports and 27,670 block-native windows; corrected stage statistics use `N = 62`, with `vagal_index` listen--mem13 `d = 1.995` | Internal task-contrast validation; no independent replication |
| ds006848 | ECG polarity and contamination audits; encoding `m`/`d` analyses on `n = 30` | Working-memory HRV claims withheld because 87.7% of 60 s windows overlap retrieval |
| ds003645 | Five-subject MEG pilot, readiness `0.7879` | Exploratory pilot; not full-cohort production validation |
| DANDI 000006 / 000458 | Units and LFP smoke/QC paths | Transport and descriptive sensitivity only |
| Figshare infant EEG | BDF ingest, masking, connectivity repair, and cohort outputs | No infant behavioural condition or clinical inference |

### Release boundary

The v2.5.0 release must be identified by the final commit and package version,
not by the presence of local generated data. Exploratory, negative, and
dataset-specific findings remain labelled as such. The v2.4.0 section below
retains the historical release surface that v2.5.0 extends.

---

## v2.4.0 — MEG support, HRV v0.1, block-native v2, geometry contract

### Major additions

**MEG ingest (exploratory/beta)**

- `meg_mag_*`, `meg_grad_*`, `meg_*` feature columns extracted from Neuromag FIF files via MNE-Python.
- Shadow mapping routes MEG features through the existing 9D coordinate contract — each `meg_*` type maps to the same subcoordinate slot as its `eeg_*` counterpart; no changes to the projection machinery.
- For simultaneous MEEG recordings: explicit row-source provenance under `row_source/` (schema: `mndm.row_source.v1`), replacing the implicit positional half-split assumption.
- New `features_projection_z` HDF5 export surface applies the configured transform pipeline (log10 → robust-z → clip) before export. Required for MEG spectral features, where raw physical power (~10⁻²⁵ W) collapses to near-zero under raw-space robust-z.
- Validated on a 5-subject pilot from ds003645 (readiness score 0.7879). Labeled **exploratory** — not yet confirmed at full 18-subject scale.

**Embodied Anchoring — concrete modality implementations** *(v2.3 introduced the principle)*

- *ECG / HRV v0.1*: Superwindow time-domain surface (`ecg_hrv_hr_mean_bpm`, `ecg_hrv_ibi_mean_ms`, `ecg_hrv_sdnn_ms`, `ecg_hrv_rmssd_ms`, `ecg_hrv_pnn50`, `ecg_hrv_nn_count`, plus artifact/coverage/quality flags) via configurable centered window (default 60 s). Optional complexity columns (`ecg_hrv_sampen`, `ecg_hrv_dfa_alpha1`) when enabled. Manifest tag: `anchor_hrv_v0_1`. Note: frequency-domain metrics (HF power, LF/HF) are not part of the v0.1 surface.
- *PPG surface*: Per-epoch rate, amplitude, variability, and quality flags when PPG channels are present; feeds `vascular_index` in `anchor_state`.
- *Pupillometry surface*: Per-epoch diameter, volatility, blink-rate proxy, and quality score when pupil traces are present.
- Automatic ECG polarity correction (validated on ds006848: 92.7% of epochs had inverted polarity; after correction, population median HR 76.2 bpm, RMSSD 40.7 ms).
- HRV contamination gating: `ecg_hrv_*` columns carry contamination flags when the superwindow overlaps task events.

**Block-native v2 sidecar ecosystem**

- `block_native_qc.json`, named window profiles, `source_window_index` provenance.
- Built-in parquet/CSV sidecars alongside HDF5.
- Inter-network Jacobian coupling columns (`coupl_*`) in block-native sidecars; stage-level pooling fallback for short-trial datasets.

### Other additions

- `anchor_auto_fit`: one-shot per-run cohort anchor fitting — resolves most `cohort_anchored` skip cases without manual anchor preparation.
- `standard_invalidity_v1` geometry contract: versioned policy for `coords_9d` duplicate-subcoordinate tolerance with per-subject diagnostics; always-on time-grid contract auditing.
- `participants.extra_tables`: generic clinical TSV join (UPDRS items, longitudinal tables) embedded into per-subject H5 output. Demonstrated on ds007526.
- Conventional EEG coma pack extended: suppression ratio, burst-suppression proxy, spectral ratios, and reactivity proxies for clinical ICU datasets.
- `openneuro_ingest` downloads now run through `uvx openneuro-py@latest` by default (`download.use_uvx`, default true; `OPENNEURO_PREFER_UVX` env override). Works around installed openneuro-py versions broken by upstream OpenNeuro GraphQL schema changes (e.g. 2026.3.0 `Cannot query field "key" on type "DatasetFile"`). `uv` added to `requirements.txt`; `openneuro-py` pin bumped to `>=2026.4.1`.

### Production validation — six additional cohorts

| Dataset | n | Notes |
|---------|---|-------|
| ds003838 | 130 subjects | HRV v0.1 + block-native; 27,670 block windows; vagal_index listen–mem13 Cohen's d = 1.995 (Wilcoxon p = 1.2×10⁻¹⁰) |
| ds006036 | 88 subjects | Block-native |
| ds007526 | 277 recordings | Parkinson gait/rest + clinical TSV join |
| ds003490 | 75 subjects | Dual-anchor rerun |
| ds003506 | 84 subjects | Dual-anchor rerun |
| ds003509 | 75 subjects | Dual-anchor rerun |

---

## v2.3.0 — Embodied Anchoring Principle, event/block layers

### Major additions

- **Embodied Anchoring Principle (EAP)**: additive body-state surface (`anchor_state`, `anchor_state_dot`, `anchor_quality`) aligned to the same epoch grid as `mnps_3d`, without redefining the canonical chart. Optional `anchor_coupling` for downstream body-brain covariation diagnostics. Four index slots: `vagal_index`, `sympathetic_index`, `anchor_index`, `vascular_index`.
- **`geometry_contract`**: always-on mathematical validity reporting for canonical geometry exports.
- **`event_locked`**: generic derived analysis layer for short-event-centered questions.
- **`block_native`**: generic derived analysis layer for sustained-block / task-segment questions.
- **Explicit coordinate anchoring** (formalized from the 2.1 line):
  - `coords_3d_subject_anchored` / `coords_9d_subject_anchored`: preserves within-subject/session-relative geometry.
  - `coords_3d_cohort_anchored` / `coords_9d_cohort_anchored`: uses a frozen feature anchor for cross-subject and cross-group comparisons.
  - `/feature_anchors/*`: per-feature center/scale statistics with release-bound `anchor_id` / `anchor_hash` provenance.

### Other additions

- DANDI and PhysioNet ingest/download support.
- Sleep-spindle detection support (YASA-based annotation alignment).
- NWB and WFDB source format support.
- Conventional EEG comparator packs alongside the MNPS contract.
- HRV-oriented embodied-anchor features and task-segment-driven block-native export (demonstrated on ds003838).
- Regional EEG via channel-group trajectories (frontal, central, parietal-occipital, temporal) with optional CSD preprocessing (λ² = 1e-5, stiffness = 4.0).

---

## v2.1.0 — Explicit coordinate layers, MNDM contract formalization

Major change: MNDM 2.1 explicitly separates exported coordinate layers:

- `subject_anchored`: preserves subject/session-relative geometry.
- `cohort_anchored`: uses a frozen feature anchor for cross-subject and cross-group comparisons.

### Additional functionality

- Added DANDI and PhysioNet ingest/download support.
- Added sleep-spindle detection support.
- Added support for NWB and WFDB source formats.
