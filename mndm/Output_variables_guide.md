### HDF5 output schema (Noetic Ingest)

This file documents **all nodes (datasets + groups) and relevant HDF5 attributes** written by the MNDM summarization pipeline into each `*.h5`.

Notation:
- **T**: number of MNPS timepoints (per-window/epoch on the MNPS grid)
- **D**: MNPS dimension (typically 3: `[m,d,e]`)
- **K**: Stratified MNPS v2 dimension (typically 9)
- **W / W2**: number of Jacobian windows (for 3D vs v2)

---

### Run-level JSON provenance (run directory sidecars)

Each summarize run directory also writes JSON sidecars outside HDF5, notably:

- `run_manifest.json` (run capabilities + provenance + config digest)
- `features_snapshot.json` (feature-table snapshot for the run)
- `run_errors.json` (captured grouping/runtime failures, when present)
- `normalization_report.json` (normalization runtime summary + pre/post probe results)
- `stage_mapping_qc.json` (run-level aggregate + per-subject stage/event mapping QC, when available)

Normalization-specific provenance (ComBat pilot) is recorded in:

- `run_manifest.json` -> `extra.normalization`
  - `status`, `method`, `scope`, `batch_key`
  - `batch_counts`, `rows_harmonized`, `feature_columns_harmonized`
  - `covariates_used`, `covariate_coverage`
  - `family_wise` (family grouping strategy + per-family chunk/harmonization counts)
  - `validation` (pre/post probes: `batch_eta2`, `target_eta2`, `perturbation`)
- `run_manifest.json` -> `extra.normalization_report`
  - sidecar write status/path for `normalization_report.json`
- `features_snapshot.json` -> `normalization`

This is the primary place to verify whether batch harmonization was applied for a run.

Event/stage provenance additions:

- `summary.json` -> `stage_mapping_qc`
  - per-subject mapping QC (raw label counts, mapped/unmapped counts, stage-window counts)
  - explicit expected vs detected stage-frequency coverage
- `summary.json` -> `event_provenance`
  - status + source events path + exported event table columns + row count
- `run_manifest.json` -> `extra.stage_mapping_qc`
  - sidecar write status/path for `stage_mapping_qc.json` and run-level aggregate coverage

Relevant config knobs (dataset override under `epoching.datasets.<id>.sampling`):

- `prefer_events_stage_in_summary` (bool): allows summarize-only reruns to re-infer stage
  from raw events and override stale stage columns in `features.csv`.
- `stage_blocking.enabled` (bool): enables continuous block inference from sparse event markers.
- `stage_blocking.stage_event_regex`: regex for block-start event labels.
- `stage_blocking.bridge_marker_labels`: optional dense in-block marker labels.
- `stage_blocking.bridge_tail_sec` / `stage_blocking.bridge_tail_cap_sec`: control
  how far bridge markers extend an inferred block when no explicit duration is present.
- `stage_blocking.window_membership.mode`: how inferred absolute blocks claim MNPS windows.
  Common values: `midpoint_in_interval`, `fully_contained`, `overlap_frac_ge`.
- `stage_blocking.window_membership.min_overlap_fraction`: overlap threshold used when
  `window_membership.mode: overlap_frac_ge`.
- `stage_blocking.expected_stage_frequencies_hz`: optional expected frequency/intensity ids for explicit QC presence/absence reporting.
- legacy aliases still accepted: `photic_regex`, `hv_mark_labels`, `use_hv_marks`,
  `hv_tail_sec`, `hv_tail_cap_sec`, `preserve_photic_blocks`, `expected_frequencies_hz`.
- `mnps.overlap` vs `stage_blocking.window_membership.min_overlap_fraction`: the former is
  MNPS stride overlap during window construction; the latter is interval geometry when labeling
  already-constructed windows.

---

### Root: HDF5 attributes (top-level `h5.attrs`)

- **`dataset_id`** *(str)*: dataset label used throughout the pipeline (often `dsXXXX:sub-YYY:<condition>_<task>_<run>[_acq]`).
- **`manifest`** *(str, JSON)*: JSON string containing the same general information as `summary.json` (meta-indices, samples, coords_v2 metadata, etc.). This is also where Tier 1/2 “new measurements” are stored (see below).
- **`subject_id`** *(str)*: always attempted; prefers payload `subject_id`, otherwise derived from `dataset_id`.

Payload attributes (`payload.attrs`) are also copied into `h5.attrs` when not `None`, e.g.:
- **`dataset`** *(str)*: dataset id (e.g. `ds005555`)
- **`subject_id`** *(str)*, **`session`** *(str|None)*, **`condition`** *(str|None)*, **`task`** *(str|None)*, **`run`** *(str|None)*, **`acq`** *(str|None)*
- **`fs_out`** *(float)*: MNPS sampling rate (Hz) on the MNPS grid
- **`window_sec`** *(float)*, **`overlap`** *(float)*: MNPS windowing (for MNPS projection / derivative grid)
- **`stage_codebook`** *(obj)*: codebook for stage labels (often serialized)
- **`stage_source`** *(str|None)*, **`stage_column`** *(str|None)*
- **`coords_v2_names`** *(list[str]|None)*: v2 coordinate names (if v2 exists)
- **`schema_version`** *(str)*: `mnps_tensor_spec_v2_1` when MNDM 2.1 coordinate layers or embedded feature anchors are present.
- **`mndm_version`** *(str)*: `2.1` for anchored-coordinate contract outputs.
- **`primary_coordinate_layer`** *(str)*: usually `coords_3d_cohort_anchored` when a cohort/external anchor is configured, otherwise `coords_3d_subject_anchored`.
- **`primary_coordinate_contract`** *(str)*: `cohort_anchored` or `subject_anchored`.
- **`anchor_id`**, **`anchor_hash`** *(str|None)*: identity and hash of the feature-anchor artifact used for cohort-anchored coordinates.

Derived convenience attrs (best-effort; may be absent):
- **`meta_<field>`** *(str/int/float)*: flattens scalar fields from `participant_meta` (participants.tsv) into top-level attrs.
- **`group`** *(str)*: may be derived/normalized from `participant_meta` if not already set.
- **`condition`** *(str)*: may be derived from session-/meta-fields if not already set.

---

### Manifest (JSON in `h5.attrs["manifest"]`): Tier 0/1/2 measurement blocks

These are **analysis-agnostic descriptive blocks** embedded in the manifest JSON (and also written to `summary.json` / `qc_summary.json`).

- **`dist_summary`** *(object)*: distributional geometry per coordinate.
  - **`dist_summary.axes.<coord>`** where `<coord>` ∈ `{m,d,e}`:
    - `n`, `mean`, `median`, `std`, `iqr`, `mad`, `mad_sigma`, `skewness`, `kurtosis_excess`, `delta_mean_median`
  - **`dist_summary.subcoords.<name>`** (only if v2 exists): same fields for each v2 subcoordinate.

- **`tau_summary`** *(object)*: autocorrelation length per coordinate (seconds).
  - **`tau_summary.axes.<coord>.tau_sec`**
  - **`tau_summary.subcoords.<name>.tau_sec`** (only if v2 exists)
  - Each entry also includes: `dt_sec`, `max_lag_sec`, `threshold`

- **`tier2_jacobian`** *(object)*: MNJ-adjacent metrics from the primary Jacobian (typically 3×3).
  - **`tier2_jacobian.jacobian_condition_number`**: descriptives for κ(J)=σmax/σmin as `dist_summary`-style fields.
  - **`tier2_jacobian.signed_divergence_balance`**: `frac_pos`, `frac_neg`, `mean_pos`, `mean_neg`, `mean_abs` on `trace(J)`.
  - **`tier2_jacobian.rotation_coherence`** *(3D only)*: `mean_resultant_length` and `mean_axis` for the antisymmetric rotation axis.

- **`tier2_emmi`** *(object)*: derived indices from MNPS + speed.
  - `speed_mean`, `speed_median`, `mv_median`, `emmi_e_over_m_median`, `mv_over_speed_median`

- **`conventional_eeg`** *(object; EEG only, optional)*: config-driven conventional qEEG comparator summaries.
  - `schema_version = "mndm.conventional_eeg.v1"`
  - `packs`: enabled comparator packs, currently including `tier1`, `complexity`, and `connectivity`
  - `columns`: emitted feature-table columns such as:
    - `eeg_conventional_relative_<band>`
    - `eeg_conventional_ratio_<name>`
    - `eeg_conventional_peak_<name>`
    - `eeg_conventional_complexity_<name>`
    - `eeg_conventional_connectivity_<name>`
  - `families.<family>.<feature>`: descriptives for each comparator feature, including
    `column`, `n`, `nan_frac`, `mean`, `median`, `std`, `iqr`, `mad`
  - current family mapping:
    - `relative` -> relative bandpower outputs
    - `ratio` -> slowing/ratio outputs
    - `peak` -> alpha-peak / median-frequency / spectral-edge outputs
    - `complexity` -> spectral-entropy / permutation-entropy / Hjorth outputs
    - `connectivity` -> synchrony summaries such as `alpha_FP_plv_mean` or `alpha_FB_coh_mean`
  - granularity note:
    - `relative`, `ratio`, `peak`, and `complexity` are epoch-aligned comparator surfaces
    - `connectivity` is currently a recording-level summary surface broadcast across epochs in the feature table

---

### Root datasets

- **`/time`** *(float64, shape `[T]`)*: monotonically increasing time (seconds) on the MNPS grid.
- **`/mnps_3d`** *(float32, shape `[T,3]`)*: MNPS coordinates \([m,d,e]\).
- **`/mnps_3d_dot`** *(float32, shape `[T,3]`)*: derivatives of `mnps_3d` on the MNPS grid.

Optional root datasets:
- **`/z`** *(float32, shape `[T,Kz]`)*: embodied/interoceptive channels (if enabled).
- **`/window_start`** *(float32, shape `[T]`)*: start time (sec) for each MNPS window.
- **`/window_end`** *(float32, shape `[T]`)*: end time (sec) for each MNPS window.

---

### Group: `/labels`

Created if `stage` or other label arrays exist.

- **`/labels/stage`** *(int8, shape `[T]`)*: stage code per MNPS timepoint.
- **`/labels/<name>`** *(int8, shape `[T]`)*: optional binary labels (e.g. event→MNPS-mapped labels).

---

### Group: `/events`

Created if `payload.events` and/or `payload.event_table_columns` exists.

- Legacy arrays:
  - **`/events/<name>`** *(int64 or float64, shape `[N]`)*: event series (either indices or timestamps; ingest treats them as 1D arrays).
- Columnar event-provenance table (when available):
  - **`/events/onset_sec`**, **`/events/duration_sec`** *(float64, shape `[N]`)*
  - **`/events/raw_event_label`**, **`/events/normalized_event_label`** *(utf-8 strings, shape `[N]`)*
  - **`/events/mapped_stage_code`** *(float64, shape `[N]`)*
  - **`/events/mapping_mode`**, **`/events/mapping_rule`** *(utf-8 strings, shape `[N]`)*
  - **`/events/source_event_column`** *(utf-8 strings, shape `[N]`)*
  - **`/events/inferred_block_id`**, **`/events/window_assignment_count`** *(int32, shape `[N]`)*
  - optional stage-block helpers (e.g. `stage_block_frequency_hz`, `is_stage_block_event`);
    historical photic aliases such as `photic_frequency_hz` may also be present for backward compatibility
- Interpretation note:
  - `window_assignment_count` depends on `stage_blocking.window_membership.mode`
  - stricter modes like `fully_contained` usually reduce assigned photic/block windows
    without changing raw frequency detection or the presence of `/labels/stage`
- Group attrs:
  - **`_has_event_table`** *(bool-like)*: indicates columnar event table is present.
  - **`_schema_version`** *(str)*: event-provenance schema tag when exported.

---

### Group: `/event_windows`

Created when summarize emits the additive EEG event-window contract.

- **`/event_windows/event_id`**, **`/event_windows/window_id`** *(int32, shape `[R]`)*
- **`/event_windows/rel_time_sec`** *(float32, shape `[R]`)*
- **`/event_windows/bin_label`** *(utf-8 strings, shape `[R]`)*
- **`/event_windows/overlap_sec`**, **`/event_windows/overlap_frac`** *(float32, shape `[R]`)*
- **`/event_windows/event_label`**, **`/event_windows/event_label_key`** *(utf-8 strings, shape `[R]`)*
- **`/event_windows/event_onset_sec`**, **`/event_windows/event_duration_sec`** *(float32, shape `[R]`)*
- **`/event_windows/window_start_sec`**, **`/event_windows/window_end_sec`** *(float32, shape `[R]`)*
- **`/event_windows/window_contains_event_onset`** *(int8, shape `[R]`)*
- **`/event_windows/event_start_window_index`**, **`/event_windows/event_stop_window_index`** *(int32, shape `[R]`)*
- Group attrs:
  - **`_schema_version`** = `mndm.event_windows.v1`
  - **`reference`** *(str)*: event timestamp used as `t=0`
  - **`bins_json`** *(str, JSON)*: exact alignment bins used to generate the rows
  - **`source_events_path`** *(str|None)*: resolved BIDS `*_events.tsv` path when known

This group is additive: `/events` remains the source event table, while
`/event_windows` provides the exact join contract needed for event-locked
downstream analysis.

---

### Derived Event-Locked Sidecars

These are not written into the subject H5 contract in v1. Instead, the
event-locked pipeline writes flat Parquet/CSV sidecars that can be joined back
to H5 by identifiers such as `subject_id`, `run_id`, `window_id`, and
`matched_event_id`.

Common sidecar columns include:

- **`condition`** *(str)*: generic row type, typically `event` or `matched_control`
- **`event_type`** *(str)*: semantic event label such as `sleep_spindle` or `stage_block_end`
- **`event_source`** *(str)*: provenance source such as `annotation:*` or `derived:stage_blocking`
- **`event_onset_sec`**, **`event_duration_sec`** *(float64)*: aligned source-event timing
- **`bin_label`**, **`rel_time_sec`**, **`overlap_sec`**, **`overlap_frac`**: event-to-window alignment fields
- **`match_rank`**, **`match_distance`**, **`matched_event_id`**: matched-control provenance fields

When `event_source.kind: "derived_stage_block_end"` is used, the exported event
rows are synthesized from inferred `stage_blocking` intervals:

- one point-event is emitted per inferred block end
- the synthetic event uses `event_type = "stage_block_end"` unless overridden
- metadata/provenance is stored in the sidecar event columns rather than as a
  new H5 group
- the synthetic event metadata includes audit fields such as
  `derived_from`, `is_inferred`, `end_reason`, `membership_mode`,
  `bridge_tail_sec`, `bridge_tail_cap_sec`, `bridge_tail_ms`,
  `block_start_ms`, `block_end_ms`, and `block_duration_ms`

---

### Group: `/codebooks`

Created when summarize exports explicit codebooks.

- **`/codebooks/stage/codes`** *(int32, shape `[C]`)*
- **`/codebooks/stage/labels`** *(utf-8 strings, shape `[C]`)*
- **`/codebooks/stage/label_keys`** *(utf-8 strings, shape `[C]`)*: concise helper keys such as `eyes_closed`
- Group attrs:
  - **`_schema_version`** = `mndm.codebook.v1`
  - optional source metadata such as `source`, `column`, `events_path`

---

### Group: `/nn`

Created if `payload.nn_indices` exists.

- **`/nn/indices`** *(int32, shape `[T,k]`)*: kNN neighbor indices in MNPS space.

---

### Group: `/jacobian`

Always created (may be empty if Jacobians were not computed).

- **`/jacobian/J_hat`** *(float32, shape `[W,D,D]`)*: MNPS Jacobian estimates.
- **`/jacobian/J_dot`** *(float32, shape `[W-1,D,D]`)*: temporal difference of the Jacobian.
- **`/jacobian/centers`** *(int32, shape `[W]`)*: center index for each Jacobian window.

---

### Group: `/features_raw`

Created when summarize exports the per-epoch empirical feature surface.

- **`/features_raw/values`** *(float32, shape `[T,K]`)*: raw feature matrix in original scale.
- **`/features_raw/names`** *(utf-8 strings, shape `[K]`)*: feature column names aligned to `values`.
- **`/features_raw/metadata/*`** *(shape `[K]` per field)*: machine-readable per-feature metadata, including usage flags and provenance.

When `conventional_eeg.enabled: true`, the exported feature surface may also
contain Tier 1 qEEG comparator columns prefixed with `eeg_conventional_`.
Current generic EEG comparator families are `relative`, `ratio`, `peak`, and
`complexity`. When the connectivity pack is enabled, the feature surface may
also include `eeg_conventional_connectivity_*` columns.

---

### Group: `/features_robust_z`

Created when summarize exports the strict robust-z feature surface.

- **`/features_robust_z/values`** *(float32, shape `[T,K]`)*: strict robust-z feature matrix.
- **`/features_robust_z/names`** *(utf-8 strings, shape `[K]`)*: feature column names aligned to `values`.
- **`/features_robust_z/metadata/*`** *(shape `[K]` per field)*: machine-readable per-feature metadata aligned to `names`.

Important:
- this surface is **strict robust-z only**
- projection-only steps such as `log10` and `clip` remain represented in provenance metadata and `feature_baselines`, not baked into `features_robust_z`

---

### Groups: MNDM 2.1 coordinate layers

MNDM 2.1 makes coordinate anchoring explicit. The legacy `/mnps_3d` and
`/coords_9d` paths may still exist, but new analyses should consult
`h5.attrs["primary_coordinate_layer"]` and the layer attrs.

These anchored layer groups are additive and may be selectively omitted when
`mnps_projection.export_contracts.subject_anchored` or
`mnps_projection.export_contracts.cohort_anchored` is set to `false`.

- **`/coords_3d_subject_anchored/values`** *(float32, shape `[T,3]`)*:
  subject/session-relative 3D coordinates. This is the right layer for
  within-subject geometry, local Jacobians, trajectory shape, and reachability
  diagnostics.
- **`/coords_3d_subject_anchored/names`** *(utf-8 strings, shape `[3]`)*:
  coordinate names, usually `[m,d,e]`.
- **`/coords_9d_subject_anchored/values`** *(float32, shape `[T,K]`)*:
  subject/session-relative stratified coordinates when `mnps_9d` is enabled.
- **`/coords_9d_subject_anchored/names`** *(utf-8 strings, shape `[K]`)*.
- **`/coords_3d_cohort_anchored/values`** *(float32, shape `[T,3]`)*:
  cohort/external-anchored 3D coordinates when `mnps_projection.anchor.enabled`
  is active. This is the preferred layer for clinical group comparisons.
- **`/coords_3d_cohort_anchored/names`** *(utf-8 strings, shape `[3]`)*.
- **`/coords_9d_cohort_anchored/values`** *(float32, shape `[T,K]`)*:
  cohort/external-anchored stratified coordinates when both `mnps_9d` and an
  anchor are available.
- **`/coords_9d_cohort_anchored/names`** *(utf-8 strings, shape `[K]`)*.

Coordinate-layer group attrs include:

- **`schema_version`** = `mndm.coordinate_layer.v2.1`
- **`coordinate_contract`** = `subject_anchored` or `cohort_anchored`
- **`anchor_id`**, **`anchor_hash`**, **`anchor_source`** for cohort-anchored layers
- **`role`**: human-readable intended use, e.g. `within_subject_geometry` or `clinical_group_comparison`

---

### Group: `/feature_anchors`

Created when a cohort/external anchor artifact is embedded.

This group is typically present only when the `cohort_anchored` export contract
is enabled for the run.

- **`/feature_anchors/spec`** *(attrs)*:
  - `schema_version = "mndm.feature_anchors.v2.1"`
  - `anchor_id`
  - `anchor_hash`
  - `anchor_source`
  - `cohort_filter`
  - `scale_method`
  - `subject_balanced`
  - `n_subjects`, `n_files`, `min_subjects`
- **`/feature_anchors/per_feature/feature_name`** *(str[K])*: feature names.
- **`/feature_anchors/per_feature/center`**, **`scale`**, **`q25`**, **`q50`**, **`q75`**, **`iqr_sigma`**, **`mad_sigma`**, **`qn_sigma`** *(float32[K])*: anchor statistics.
- **`/feature_anchors/per_feature/n_subjects`**, **`n_epochs`** *(float/int-like arrays)*: support counts.

Anchor fitting is subject-balanced: each subject contributes one summary value
per feature, preventing long recordings from dominating the anchor.

---

### Group: `/participant`

Participant metadata remain embedded as JSON and attrs for low-friction joins.

- **`/participant/row_json`** *(utf-8 JSON dataset)*: raw participant-table row
- **`/participant/mapped_json`** *(utf-8 JSON dataset)*: canonical derived metadata such as `group`, `condition`, `task`
- **`/participant/source_json`** *(utf-8 JSON dataset)*: lookup provenance for the participant table
- **`/participant/clinical_json`** *(utf-8 JSON dataset, optional)*: richer additive participant/session metadata carried into H5 for analysis convenience
- Attr families:
  - **`field_*`**: scalar raw participant fields
  - **`mapped_*`**: scalar derived metadata
  - **`source_*`**: scalar participant-source provenance
  - **`clinical_*`**: scalar convenience fields mirrored from `clinical_json`

---

### Group: `/coverage`

Created when summarize exports explicit cross-layer coverage metadata.

- **`/coverage/axis_fraction`** *(float32, shape `[T,3]`)*: direct-axis coverage per MNPS window for `[m,d,e]`
- **`/coverage/axis_names`** *(utf-8 strings, shape `[3]`)*
- **`/coverage/min_axis_coverage`** *(scalar float32)*
- **`/coverage/coordinate_layers_present`** *(utf-8 strings, shape `[L]`)*
- **`/coverage/coordinate_contracts_present`** *(utf-8 strings, shape `[Lc]`)*
- **`/coverage/jacobian_centers`**, **`/coverage/jacobian_9d_centers`** *(int32, optional)*: explicit mappings back to the shared MNPS time index
- Group attrs:
  - **`_schema_version`** = `mndm.coverage.v1`

---

### Group: `/provenance`

Created when summarize exports structured additive provenance blocks.

- **`/provenance/contract/*`**: export-contract metadata such as `export_contract_version`, `config_digest_sha256`, `run_manifest_ref`
- **`/provenance/anchoring/*`**: explicit coordinate contracts/layers available in this H5 plus primary contract/layer and optional anchor identity
- **`/provenance/normalization/*`**: concise normalization status/method/scope and sidecar references
- **`/provenance/event_stage_mapping/*`**: event/stage mapping versioning, source column/path, and codebook hash
- Group attrs:
  - **`_schema_version`** = `mndm.provenance.v1`

---

### Group: `/qc`

Created when summarize exports additive per-window QC.

- **`/qc/windows/retained_after_qc`** *(int8, shape `[T]`)*
- **`/qc/windows/rejected_flag`** *(int8, shape `[T]`)*
- **`/qc/windows/qc_ok_eeg`**, **`/qc/windows/qc_ok_ecg`**, **`/qc/windows/qc_ok_eog`** *(int8, optional)*
- **`/qc/windows/coverage_ok`** *(int8, shape `[T]`, optional)*
- **`/qc/windows/stage_transition_flag`** *(int8, shape `[T]`, optional)*
- Group attrs:
  - **`_schema_version`** = `mndm.qc.windows.v1`

This group intentionally carries only the light-weight, per-window contract.
Heavier QC summaries still live in `qc_summary.json` and `qc_reliability.json`.

---

### Group: `/jacobian_9D`

Created if v2 Jacobians exist.

- **`/jacobian_9D/J_hat`** *(float32, shape `[W2,K,K]`)*: Jacobian in Stratified MNPS v2 space.
- **`/jacobian_9D/J_dot`** *(float32, shape `[W2-1,K,K]`)*: temporal difference.
- **`/jacobian_9D/centers`** *(int32, shape `[W2]`)*: center index.

Optional subgroup:
- **`/jacobian_9D/cross_partials/<name>`** *(float32, shape `[W2]`)*: selected elements from the v2 Jacobian as 1D series (dataset names are sanitized; `/` is replaced with `_`).

---

### Group: `/coords_9d`

Created if stratified coordinates exist.

- **`/coords_9d/values`** *(float32, shape `[T,9]`)*: Stratified MNPS subcoords in canonical order.
- **`/coords_9d/names`** *(utf-8 strings, shape `[9]`)*: subcoord names (canonical order: `m_a,m_e,m_o,d_n,d_l,d_s,e_e,e_s,e_m`).
- **`/coords_9d` attrs**:
  - **`version`** = `"9d"`

---

### Group: `/extensions`

Created if extensions exist. The structure is **free-form** and mirrors nested dicts:

- **`/extensions/<extension_name>/...`**: subgroups/datasets for extension payloads (e.g. `e_kappa`, `rfm`, `o_koh`, `tig`).
- Conventional EEG comparator summaries are written under:
  - **`/extensions/conventional_eeg/...`**
  - families are currently grouped as `relative`, `ratio`, `peak`, `complexity`, and `connectivity`

Rule:
- dict → subgroup
- scalar/array → dataset (gzip-compressed unless scalar)

---

### Group: `/regions`

Created if raw regional signals exist (typically fMRI parcellation).
This is a supporting-input contract, not the canonical regional output contract.

- **`/regions/bold`** *(float32, shape `[n_regions, n_times]`)*: ROI×time matrix.
- **`/regions/names`** *(utf-8 strings, shape `[n_regions]`)*: ROI names/labels.
- **`/regions` attrs**:
  - **`sfreq`** *(float)*: sampling rate (Hz) for the `bold` time axis.

---

### Group: `/regional_mnps`

Created if regional MNPS/MNJ has been computed and attached to the payload.
This is the canonical modality-agnostic regional output path for both EEG and fMRI.

Per-network structure:
- **`/regional_mnps/<network_label>/mnps`** *(float32, shape `[Tr,3]`)*: regional MNPS.
- **`/regional_mnps/<network_label>/mnps_dot`** *(float32, shape `[Tr,3]`)*: derivatives.
- **`/regional_mnps/<network_label>/jacobian`** *(float32, shape `[Wr,3,3]`)*: regional Jacobian.

Per-network attrs (best-effort):
- various **metrics** as HDF5 attrs (float; `nan` may appear)
- **`n_timepoints`** *(int)*

---

### Compression / dtypes (practical details)

- All non-scalar datasets are written with **gzip compression** (`compression_opts=4`).
- `time` is `float64`; most other numeric arrays are `float32` for disk/IO.
- `labels/*` is `int8`; `events/*` is `int64` or `float64`; `nn/indices` is `int32`.

