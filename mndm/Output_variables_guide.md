### HDF5 output schema (MNDM 3.0 release line)

This file documents the **canonical groups/datasets and the attributes needed to
interpret them**, written by the MNDM summarization pipeline into each `*.h5`.
`payload.attrs` are copied through to `h5.attrs` largely as-is and are **not**
fully enumerated here (see "Root: HDF5 attributes" below) — treat the attribute
list there as illustrative, not exhaustive. Verified 2026-09-11 against
`core/src/core/io/h5_writer.py` and two real `ds005555` exports; see
`project/diary/` for the audit that produced this revision.

Release-version note:
- This document describes the **NeuralManifoldDynamics 3.0** measurement surface.
- Some embedded sub-schema identifiers intentionally still carry `v2.1` names
  (for example the explicit anchored-coordinate layer schema) because those
  subcontracts were introduced in the 2.1 release line and remain valid in 3.0.
- The root `h5.attrs["schema_version"]` tensor-spec identifier itself has moved
  past `v2_1` in current 3.0 exports (observed `mnps_tensor_spec_v2_4`); do not
  assume a fixed literal value — read it from the file. The **coordinate-layer**
  group attr `schema_version = "mndm.coordinate_layer.v2.1"` is a separate,
  intentionally stable sub-schema id (see "Groups: MNDM 2.1 coordinate layers").

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
- `skipped_recordings.json` (`mndm.skipped_recordings.v1`, added 2026-09-11; explicit, non-error, policy-driven skips such as `coverage_too_low`, `coverage_too_low_after_nan_cov_mask`, `all_epochs_dropped_by_missing_axis_policy`, `coverage_too_low_after_geometry_invalidity_policy`, `all_epochs_dropped_by_geometry_invalidity_policy` — written whenever any grouping reaches `SubjectSummaryRunner.run` but is deliberately not exported, so an empty `sub-*/` output directory always has a matching record instead of requiring analysis repos to rediscover empty stems. Distinct from `run_errors.json`: entries here are expected outcomes of coverage/geometry policy, not exceptions, and do not flip `run_manifest.json`'s `run_status` to `completed_with_errors`. Mirrored into `run_manifest.json` extra as `skipped_recordings`.)
- `normalization_report.json` (normalization runtime summary + pre/post probe results)
- `stage_mapping_qc.json` (run-level aggregate + per-subject stage/event mapping QC, when available)
- `block_native_qc.json` (run-level aggregate + per-subject block-native QC, when available)

Block-native provenance is embedded in `run_manifest.json` under:

- `counts.h5_with_block_native_windows` (int): number of H5 outputs that contain `/block_windows/`
- `capabilities.has_block_native_windows` (bool)
- `capabilities.block_native_windows_path` = `"/block_windows"`
- `capabilities.coordinate_contracts`:
  - `requested_contracts`
  - `realized_contracts`
  - `skipped_contracts_with_reason`

Local-dynamics capability discovery is available under
`capabilities.jacobian_metrics_v1`, `capabilities.finite_time_response_v1`,
`capabilities.transition_residuals_v1`,
`capabilities.transition_residual_covariance_proxy_v1`,
`capabilities.residual_covariance_proxy_v1`, and
`capabilities.stochastic_reachability_v1`, with corresponding `*_path` and
`counts.h5_with_*` fields. Dynamical-family discovery is
`capabilities.dynamical_families` with
`capabilities.dynamical_families_path = "/dynamical_families"` and
`counts.h5_with_dynamical_families`. The probe counts both the canonical tree
and legacy `/orthogonal_dynamics/` containers. A false capability means no
serialized surface was emitted; it is not evidence of a zero-valued measurement.

Per-subject block-native injection summary is available in `summary.json` under top-level `block_native`:

- `n_blocks` (int): inferred block count
- `n_windows` (int): generated window count
- `status` (`ok` | `no_blocks` | `error`)
- `block_source_kind` (str): source kind used
- `window_profile_kind` (str): profile kind used
- `named_window_profile` (str|None): named profile alias when configured
- `block_counts_by_stage`, `window_counts_by_stage`
- `block_counts_by_frequency_hz`
- `source_window_match_count`, `source_window_total`
- `derived_from` (str|None): whether blocks came from raw source events, stage-block inference, or derived label segments such as `task_state_label_segments`
- `raw_hrv_feature_columns` (list[str], optional): raw `ecg_hrv_*`/`qc_ok_ecg_hrv` columns exported directly into block-native sidecars when available
- `export_paths` (list[str]): emitted block-native sidecars for that subject/run

Per-subject block-native QC is available in `summary.json.block_native_qc` and run-level aggregate QC in `block_native_qc.json`.

Per-subject embodied-anchor manifest additions may also be present in `summary.json`:

- `anchor_state`: source features, realized anchor names, and support/quality provenance
- `anchor_coupling`: optional policy/config/status block for additive body-brain coupling diagnostics
- `anchor_hrv_v0_1`: optional ECG-HRV superwindow configuration and emitted feature names

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
- **`manifest`** *(str, JSON)*: JSON string containing the same general information as `summary.json` (meta-indices, samples, coords_9d metadata, etc.). This is also where Tier 1/2 “new measurements” are stored (see below). Only present when the JSON is small enough (`manifest_bytes <= 65535`); the full manifest always exists as the sibling root **dataset** `/manifest_json` (UTF-8 JSON string), which has no size cap.
- **`manifest_bytes`** *(int)*: byte length of the manifest JSON; use this to decide whether `h5.attrs["manifest"]` was populated or whether to read `/manifest_json` instead.
- **`subject_id`** *(str)*: always attempted; prefers payload `subject_id`, otherwise derived from `dataset_id`.
- **`mnps_axis_names`** *(list[str])*: unconditionally set to `[m, d, e]`; the canonical MNPS axis order (v1.2 spec).

Payload attributes (`payload.attrs`) are also copied into `h5.attrs` when not `None`, e.g.:
- **`dataset`** *(str)*: dataset id (e.g. `ds005555`)
- **`subject_id`** *(str)*, **`session`** *(str|None)*, **`condition`** *(str|None)*, **`task`** *(str|None)*, **`run`** *(str|None)*, **`acq`** *(str|None)*
- **`fs_out`** *(float)*: MNPS sampling rate (Hz) on the MNPS grid
- **`window_sec`** *(float)*, **`overlap`** *(float)*: MNPS windowing (for MNPS projection / derivative grid)
- **`stage_codebook`** *(obj)*: codebook for stage labels (often serialized)
- **`stage_source`** *(str|None)*, **`stage_column`** *(str|None)*
- **`coords_9d_names`** *(list[str]|None)*: 9D coordinate names (if `coords_9d` exists); this is the actual attr name (an earlier revision of this document called it `coords_v2_names`, which does not exist in current exports).
- **`schema_version`** *(str)*: the root tensor-spec identifier. Current 3.0 exports write `mnps_tensor_spec_v2_4`; do not hardcode `v2_1` — always read this value from the file rather than assuming it.
- **`mndm_version`** *(str)*: software/measurement-contract release line recorded in the export metadata; current documentation tracks `3.0.1`.
- **`primary_coordinate_layer`** *(str)*: usually `coords_3d_cohort_anchored` when a cohort/external anchor is configured, otherwise `coords_3d_subject_anchored`.
- **`primary_coordinate_contract`** *(str)*: `cohort_anchored` or `subject_anchored`.
- **`available_jacobian_layers`** *(list[str])*: names of the additive anchored-Jacobian-layer groups written for this file (e.g. `[jacobian_subject_anchored, jacobian_9D_subject_anchored]`); see "Groups: MNDM 2.1 Jacobian layers" below. Mirrored under `/provenance/anchoring/available_jacobian_layers`.
- **`anchor_id`**, **`anchor_hash`** *(str|None)*: identity and hash of the feature-anchor artifact used for cohort-anchored coordinates.
- **`geometry_invalidity_policy`** *(str)*: always-on hard-invalidity contract version for canonical MNPS/MNJ exports, currently `standard_invalidity_v1`.
- **`geometry_contract_status`** *(str)*: status of the always-on geometry contract for this export, typically `ok` or `adjusted`.

Derived convenience attrs (best-effort; may be absent):
- **`meta_<field>`** *(str/int/float)*: flattens scalar fields from `participant_meta` (participants.tsv) into top-level attrs.
- **`group`** *(str)*: may be derived/normalized from `participant_meta` if not already set.
- **`condition`** *(str)*: may be derived from session-/meta-fields if not already set.

Additional **structural** (non-hash) root attrs observed in real 3.0 exports that
are part of the measurement contract and worth knowing about, even though the
list above is illustrative rather than exhaustive:
- **`coverage_seconds_{assumed,effective,measured}`**, **`coverage_seconds_method`**, **`coverage_min_seconds_effective`**, **`coverage_min_epochs_effective`**, **`coverage_rule_tag`**: coverage-threshold provenance for this export.
- **`epochs_raw`**, **`epochs_after_nan_mask`**, **`epochs_after_qc`**, **`epochs_after_geometry_policy`**: epoch-count funnel through the pipeline stages.
- **`direct_axis_coverage_{m,d,e}_{mean,min}`**, **`direct_axis_renorm`**, **`missing_axis_policy`**, **`missing_weighted_feature_rate_{direct,v2}`**: per-axis direct-feature support and missing-data policy.
- **`dropped_geometry_invalid_epochs`**, **`dropped_missing_axis_epochs`**, **`geometry_jacobian_invalid_windows`**, **`geometry_jacobian_9d_invalid_windows`**: counts of rows/windows removed by the standard invalidity policy (summarized in `/provenance/geometry_contract`).
- **`mde_from_v2_*`** (`aggregation`, `aggregation_requested`, `map`, `pooling_legacy`, `v1_mapping_hash`, `v1_mapping_input`, `v1_mapping_matrix`, `v1_mapping_matrix_rows`, `v1_mapping_normalized`, `v1_mapping_source`) and **`mde_mode_{effective,requested}`**: provenance for deriving the 3D `[m,d,e]` surface from the 9D construction when applicable.
- **`mnps_9d_constructs`**, **`mnps_9d_definition_version`**, **`x_definition`**, **`v2_definition`**, **`v2_missing_policy`**, **`subcoords_hash_v2`**, **`weights_hash_direct`**: definitional/version provenance for the 3D and 9D coordinate construction.
- **`time_reference_{enabled,status,source,anchor_mode,schema_version}`**: whether/how an explicit external time reference was applied to this export's time base.
- **`reproducibility_seed`**, **`reproducibility_seed_source`**, **`pip_freeze_hash`**, **`env_hash`**, **`python_version`**, **`platform`**: run-environment reproducibility provenance.
- **`*_hash_saved`** (`x_hash_saved`, `x_hash_jacobian_input`, `x_hash_knn_input`, `jacobian_hash_saved`, `jacobian_dot_hash_saved`, `jacobian_9d_hash_saved`, `jacobian_9d_dot_hash_saved`, `coords_9d_hash_saved`, `coords_9d_hash_jacobian_input`, `coords_9d_hash_knn_input`, `nn_indices_hash_saved`, `features_raw_hash_saved`, `features_robust_z_hash_saved`, `feature_export_names_hash`): content-hash provenance per exported surface, for deterministic-replay checks.
- **`feature_export_scope`**, **`feature_metadata_fields`**, **`features_raw_column_count`**, **`features_robust_z_column_count`**: feature-table export scope/shape provenance.
- **`coords_9d_{allow_all_non_finite_columns,allow_duplicate_columns,allow_duplicate_constant_columns,degraded_mode,duplicate_count,duplicate_constant_count,all_non_finite_count}`**, **`e_e_{backend,construct,degraded_mode,metric}`**: 9D-construction policy flags and per-axis fallback provenance (`e_e` is the entropy-energy subcoordinate).
- **`normalize_mode`**, **`export_contract_version`**: export-contract identifiers.
- **`anchor_state_names`**: present even when the `/anchor_state` values matrix itself is empty for this run (see "Groups: anchored coordinates vs embodied anchors").

---

### Manifest (JSON in `h5.attrs["manifest"]`): Tier 0/1/2 measurement blocks

These are **analysis-agnostic descriptive blocks** embedded in the manifest JSON (and also written to `summary.json` / `qc_summary.json`).

- **`geometry_contract`** *(object)*: always-on mathematical invalidity contract for canonical geometry export.
  - **`policy_version`**: currently `standard_invalidity_v1`.
  - **`status`**: `ok` when no mathematically invalid geometry had to be removed, `adjusted` when invalid epochs or Jacobian windows were dropped or retained surfaces are degraded.
  - **`shared_time_grid`**: counts for `epochs_before_policy`, `epochs_retained`, `epochs_dropped`, `drop_fraction`, and `drop_reason_counts`. This object is serialized verbatim under **`/provenance/geometry_contract/shared_time_grid/*`**. Do **not** confuse it with the unrelated, same-named **`/coverage/shared_time_grid`** dataset, which is a plain scalar `int8` flag (1 = the exported surfaces share one time grid), not this epoch-drop-count object.
  - **`time_grid`**: realized time-base audit for the exported grid, including recovered inter-window `dt`, recovered window lengths, match/mismatch booleans against runtime/config values, and any warnings about non-finite or non-positive bounds.
  - **`mnps_3d`**: finite-row fraction before policy plus any degenerate/all-NaN axes.
  - **`coords_9d`**: whether the 9D surface is available, shape-matched to the shared time grid, and whether non-finite rows remained on the retained shared grid.
  - **`jacobian`** / **`jacobian_9d`**: retained vs invalid local-window counts after the hard Jacobian policy.
  - Interpretation rule:
    - this block is part of the canonical measurement contract
    - downstream analyses should gate interpretation on this block before using Jacobian- or reachability-style summaries

- **`mnps_mnj_sanity.derivative_self_consistency`** *(object, optional)*: reviewer-facing QA that compares finite-difference `diff(mnps_3d)/dt` against the exported `/mnps_3d_dot` field on the realized time grid.
  - Reports finite interval fraction, vector relative error, symmetric speed-ratio summaries, and edge-vs-interior behavior near derivative/filter boundaries.
  - This is a diagnostic block, not a hard invalidity gate.

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
  - `packs`: enabled comparator packs, currently including `tier1`, `complexity`, `connectivity`, and `coma`
  - `artifact_qc` *(added 2026-09-11)*: `{status, reason}` where `status` is
    `confirmed_applied` (an artifact-reduction method, e.g. ICA/EOG
    regression, was confirmed to have actually run for this recording),
    `not_confirmed` (a method was configured but did not confirmedly run,
    or ran with no sidecar evidence either way), or `not_assessed` (no
    artifact-QC sidecar evidence at all). This is an explicit, **non-blocking**
    provenance flag: the extension is still computed and written when
    `status != confirmed_applied` (withholding it entirely would make this
    family unusable on every dataset that has not yet enabled ICA/EOG
    regression, e.g. PhysioNet I-CARE, whose `preprocess.artifacts.method`
    is `none`). Readers must consult this field before trusting
    family-average band-power / connectivity descriptives, which can be
    outlier-heavy without artifact rejection (see
    `project/mnps_v3/tests/ingest_jacobian_fidelity_handover_2.md` items 9
    and 12). Mirrors the same evidence used for `/qc/windows/qc_ok_eeg`'s
    `-1` (not assessed) state.
  - `columns`: emitted feature-table columns such as:
    - `eeg_conventional_relative_<band>`
    - `eeg_conventional_ratio_<name>`
    - `eeg_conventional_peak_<name>`
    - `eeg_conventional_complexity_<name>`
    - `eeg_conventional_connectivity_<name>`
    - `eeg_conventional_coma_<name>`
  - `families.<family>.<feature>`: descriptives for each comparator feature, including
    `column`, `n`, `nan_frac`, `mean`, `median`, `std`, `iqr`, `mad`
  - current family mapping:
    - `relative` -> relative bandpower outputs
    - `ratio` -> slowing/ratio outputs
    - `peak` -> alpha-peak / median-frequency / spectral-edge outputs
    - `complexity` -> spectral-entropy / permutation-entropy / Hjorth outputs
    - `connectivity` -> synchrony summaries such as `alpha_FP_plv_mean` or `alpha_FB_coh_mean`
    - `coma` -> ICU/coma EEG proxies such as `suppression_ratio`, `burst_suppression_proxy`,
      `continuity_proxy`, `alpha_delta_ratio`, and `reactivity_proxy`
  - `clinical_markers` *(object; optional, coma pack)*:
    - explicit availability metadata for non-EEG biomarkers
    - `mode = "eeg_only_proxy"` when only EEG-derived proxies are present
    - `markers.ssep|nse|gcs|s100b.status = "unavailable"` until external clinical data are ingested
  - granularity note:
    - `relative`, `ratio`, `peak`, `complexity`, and `coma` are epoch-aligned comparator surfaces
    - `connectivity` is currently a recording-level summary surface broadcast across epochs in the feature table

---

### Root datasets

- **`/time`** *(float64, shape `[T]`)*: monotonically increasing time (seconds) on the MNPS grid.
- **`/mnps_3d`** *(float32, shape `[T,3]`)*: MNPS coordinates \([m,d,e]\).
  - Standard contract note:
    - rows that are mathematically invalid are dropped before export rather than clamped
    - `T` therefore reflects the retained valid MNPS grid, not necessarily the raw pre-policy epoch count
- **`/mnps_3d_dot`** *(float32, shape `[T,3]`)*: derivatives of `mnps_3d` on the MNPS grid.

Optional root datasets:
- **`/z`** *(float32, shape `[T,Kz]`)*: embodied/interoceptive channels (if enabled).
- **`/window_start`** *(float32, shape `[T]`)*: start time (sec) for each MNPS window.
- **`/window_end`** *(float32, shape `[T]`)*: end time (sec) for each MNPS window.

---

### Group: `/labels`

Created if `stage` or other label arrays exist.

- **`/labels/stage`** *(int8, shape `[T]`)*: stage code per MNPS timepoint.
- **`/labels/<name>`** *(shape `[T]`)*: optional aligned label series. Depending on the source, this may be integer-coded, binary, or UTF-8 string labels such as task-state labels. For sleep-stage runs this commonly includes one-hot-style per-stage series such as `/labels/wake`, `/labels/n1`, `/labels/n2`, `/labels/n3`, `/labels/rem`, `/labels/r` alongside `/labels/stage`.
- Group attrs on `/labels`: **`alignment`** = `per_timepoint`, and (when the payload carries them) **`stage_source`**, **`stage_column`**, **`stage_codebook`** (JSON string) — the same values also copied to the root attrs of the same names.

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
- when anchor and within-run label surfaces are present, sidecars may also include joined
  `anchor_state_*`, `anchor_state_dot_*`, `anchor_quality_*`, `task_state_label`,
  and `task_load_n` columns

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

**`event_metadata_json`** *(str, JSON blob)*: any CSV columns not recognized
by the standard `EventTable` schema (see `event_annotations.py`'s
`_FLOAT_COLS`/`_STR_COLS`) are not dropped when loading a `kind: "csv"`
event source — `load_event_table_from_csv` folds them into a single JSON
string per event, and `build_event_locked_table` copies it verbatim into
every corresponding output row as `event_metadata_json`. For example,
ds004587's Illusion Game trial CSVs (see below) carry `illusion_strength`,
`type`, `correct`, `rt`, `block_number`, `trial_number`,
`qc_ok_event_sync`, and `within_sync_bracket` this way; downstream analysis
must `json.loads()` this column to recover them as typed values (see
`illusionGame_EAP/src/ds004587_support.py`'s `load_event_locked_trial_table`).

**ds004587 (Illusion Game) trial-level event-locking.** ds004587's raw EEG
`events.tsv` carries only a single "recording start" marker — there is no
hardware trial trigger. `project/scripts/28_ds004587_lux_trial_sync.py`
(backed by `mndm.pipeline.ds004587_lux_sync`) recovers per-trial onsets from
the injected `LUX` photosensor channel using landmark-consensus clock-offset
estimation against distinctive block-break landmarks (validated: ~72/100
runs pass the sync quality gate with sub-30ms residuals). It writes one
`EventTable`-compatible CSV per run
(`{events_core}_ig_trials_v1.csv`) plus a cohort-level
`ds004587_ig_trial_sync_quality.csv` audit. `event_locked.datasets.ds004587`
(`config_ingest_ds004587.yaml`) then consumes those CSVs the normal
`kind: "csv"` way, producing per-trial `event_locked.parquet` rows with `m`/
`d`/`e` MNPS geometry at the nearest ~8s/4s-step window to each recovered
trial onset. Trials whose run failed the sync quality gate, or that fall
outside the run's landmark-bracketed interval, are never fabricated: their
source-CSV `onset_sec` is `NaN`, so the alignment step silently excludes them
(counted in `n_events_excluded_non_finite`) rather than guessing. Because
several trials commonly land in the same MNPS window at this cadence,
consumers must collapse each trial (`event_id`) to its single nearest window
(minimum `abs(rel_time_sec)`) before running trial-level statistics — see
`illusionGame_EAP/src/09_trial_level_event_locked_eap.py`. This still cannot
produce a rest-vs-illusion neural contrast (ds004587 has no rest-EEG in the
source BIDS release) or trial-locked Jacobian/MNJ metrics (only `m`/`d`/`e`
are exported by `build_event_locked_table`).

---

### Group: `/codebooks`

Created when summarize exports explicit codebooks.

- **`/codebooks/stage/codes`** *(int32, shape `[C]`)*
- **`/codebooks/stage/labels`** *(utf-8 strings, shape `[C]`)*
- **`/codebooks/stage/label_keys`** *(utf-8 strings, shape `[C]`)*: concise helper keys such as `eyes_closed`
- Group attrs (on `/codebooks/stage`, and each other emitted codebook subgroup):
  - **`_schema_version`** = `mndm.codebook.v1`
  - optional source metadata such as `source`, `column`, `events_path`
- The parent **`/codebooks`** group itself also carries **`_schema_version`** = `mndm.codebooks.v1` (plural; distinct from the per-codebook `mndm.codebook.v1` singular tag on each child group).

---

### Group: `/nn`

Created if `payload.nn_indices` exists.

- **`/nn/indices`** *(int32, shape `[T,k]`)*: kNN neighbor indices in MNPS space.

---

### Group: `/jacobian`

Always created (may be empty if Jacobians were not computed).

- **`/jacobian/J_hat`** *(float32, shape `[W,D,D]`)*: MNPS Jacobian estimates.
- **`/jacobian/J_dot`** *(float32, shape `[W,D,D]`)*: per-window Jacobian derivative estimate. **Not** `[W-1,D,D]`: `J_dot` is required to have the same shape as `J_hat` (one derivative estimate per retained window, not a temporal finite difference between consecutive windows). The payload contract enforces `J_dot.shape == J_hat.shape` (see `mndm/src/mndm/schema.py`).
- **`/jacobian/centers`** *(int32, shape `[W]`)*: center index for each Jacobian window.
- **`/jacobian/affine_reference`** *(float32, shape `[W,D]`, optional)*: local fit-neighborhood reference used by the affine derivative model, when the payload sets it directly.
- **`/jacobian/affine_intercept`** *(float32, shape `[W,D]`, optional)*: local derivative intercept paired with `affine_reference`; it does not alter `J_hat` semantics.
  - In practice, the current summarize pipeline routes the affine fit outputs through `/jacobian/diagnostics/affine_reference_windows` and `/jacobian/diagnostics/affine_intercept_windows` (same `[W,D]` shape) rather than the top-level `affine_reference`/`affine_intercept` paths; check both locations when reading a file.
- **`/jacobian/diagnostics/*`** *(optional)*: local-fit diagnostics alongside the hard-invalidity fields below. Observed additional fields include `attempted_centers` `[W]`, `condition_number_windows` `[W]`, `failed_centers`, `hard_invalid_centers`, `hard_invalid_window_mask` `[W]` (int8), `local_fit_mse_windows`, `local_fit_mse_baseline_windows`, `rel_mse_baseline_windows` `[W]`, plus group attrs such as `failed`, `failed_insufficient_neighbours`, `failed_nonfinite_samples`, `hard_invalid_condition_number_windows`, `hard_invalid_nonfinite_windows`, and `j_dot_dt`. Additive holdout fields (2026-09-13): `rel_mse_baseline_oos_windows` `[W]`, `n_holdout_samples` `[W]`, and attrs `rel_mse_baseline_oos_median`, `oos_holdout_stride`. These score an auxiliary train/holdout refit of the same neighborhood (every 4th sorted index held out). Baseline is the holdout-set mean ẋ, matching in-sample `rel_mse_baseline`. They do **not** replace the in-sample fit-fidelity gate; `computed` still uses `rel_mse_baseline_median`. NaN means the split had too few train or holdout rows. An experimental discrete one-step map `x_{t+h} ≈ Φ(x_t − x̄) + b` lives in `mndm.jacobian_discrete` and is **not** serialized on `/jacobian`. That experimental module does not use `x_dot`, is not wired into summarize, and a passing next-state `rel_mse` is not a Family B `computed` claim. The writable family `/dynamical_families/one_step/v1` (`mndm.affine_one_step.v1`, estimator `recording_affine_discrete_map`) is a separate recording-level object; it is not `jacobian_discrete` and is not this diagnostics group.
- **`/jacobian/diagnostics/hard_invalid_centers`** *(int32, optional)*: center indices of Jacobian windows removed by the standard invalidity policy.
- **`/jacobian/diagnostics/hard_invalid_windows`** *(scalar attr, optional)*: number of canonical windows removed after Jacobian estimation.
- **`/jacobian/diagnostics/hard_invalid_condition_number_threshold`** *(scalar attr, optional)*: hard condition-number threshold used by the standard policy.
- **`/jacobian/derived_metrics/v1/`** *(optional)*: `mndm.jacobian_metrics.v1` semantic local-dynamics metrics. Its `series/` datasets are per valid-Jacobian-window fields (`spectral_abscissa`, `numerical_abscissa`, `symmetric_rate_min`, `reactivity_gap`, `stable_reactive_flag`, magnitude/deformation/rotation diagnostics, `rel_mse_baseline`); `summary/` carries support counts, `stable_reactive_fraction`, `rel_mse_baseline_median`, and `fit_identified`; `provenance/` carries zero tolerances, metric semantics, `operator_semantics`, `abscissa_units`, `nominal_dt_sec`, and the fit-fidelity gate fields below. Sibling certificate fields: `computation_status` (`computed` only if the fit-fidelity gate passes **and** any finite metric windows exist, else `insufficient_support` with a `failure_reason`), `measurement_validity` (`not_assessed` when computed, else `not_applicable`), `claim_status` (`no_biological_claim` on new writes). These are mathematical provenance on `J_hat`, not S3-licensed empirical NDT \(\alpha/\omega\). Window-level `spectral_abscissa` / `numerical_abscissa` here are chart-Jacobian metrics under the I-CARE `fit_fidelity_gate`; they are not the recording-level generator proxies `spectral_abscissa_level3` / `numerical_abscissa_level3` under `/dynamical_families/one_step/v1`. Series `reactivity_gap` is \(\omega-\alpha\) of `J_hat`, not `reactivity_gap_level3`.
  - Full `series/` fields are `spectral_abscissa`, `numerical_abscissa`,
    `symmetric_rate_min`, `symmetric_rate_max`, `reactivity_gap`,
    `stable_reactive_flag`, `dynamical_regime`, `spectral_radius`,
    `frobenius_norm`, `trace`, `rotation_norm`, `henrici_departure`, and
    `rel_mse_baseline` (per-window local-fit fidelity, when the estimator's
    diagnostics were supplied to the gate; else `NaN`).
  - Theory \(\Omega\) appears only as the scalar `rotation_norm`, not as a matrix. Theory \(G_{\mathrm{peak}}\) is **not** in this group. FTR `g_peak_over_horizons` is a peak-gain analogue, not licensed NDT \(G_{\mathrm{peak}}\).
  - **Fit-fidelity gate (added 2026-09-11, provisional threshold, opt-in per dataset overlay):** `alpha`/`omega` and the regime/flag classification derived from them require the local affine fit behind `J_hat` to beat the no-dynamics (trajectory-mean) baseline. This gate is **disabled by default** (`local_dynamics.jacobian_metrics.fit_fidelity_gate.enabled: false`, see `config_ingest_common_dynamical_families.yaml`) and only evaluated on dataset overlays that explicitly opt in (currently the PhysioNet I-CARE 2.1 overlays only — see `config_ingest_physionet_i-care_2_1_dynamical_families.yaml`); every other dataset's `computation_status`/series/summary are byte-for-byte unchanged from the prior release. `provenance/rel_mse_baseline_median` mirrors `/jacobian/diagnostics/rel_mse_baseline_median` — but is recomputed from the *retained* (post-geometry-filter) `rel_mse_baseline_windows` array whenever both are supplied, rather than trusting a possibly-stale pre-filter scalar. `provenance/fit_fidelity_threshold` (default `0.9`) and `provenance/fit_fidelity_gate` record the gate state: `not_evaluated` (disabled, or the caller supplied neither diagnostic at all), `rel_mse_baseline_median` (evaluated, resolved to a finite value), or `rel_mse_baseline_unknown` (evaluated but unresolvable — e.g. all-NaN windows — which **fails closed**, not open, the same as a value that fails the threshold). When the recording-level `rel_mse_baseline_median >= fit_fidelity_threshold`, or is unresolvable, `summary/fit_identified=False`, `computation_status="insufficient_support"`, `failure_reason` is `local_linear_fit_not_better_than_baseline` or `fit_fidelity_unknown` respectively, and every window's `stable_reactive_flag`/`dynamical_regime` stays at its invalid/not-testable fill value (`-1`). **Even when the recording-level median passes**, individual windows whose own `rel_mse_baseline` (from the per-window array) fails the same threshold are still withheld one-by-one (`summary/n_windows_fit_unidentified`, `series/spectral_abscissa` etc. `NaN`, `stable_reactive_flag=-1`, `dynamical_regime=-1` for exactly those windows) — the recording-level median is a coarse admission gate, not a license to classify every window in an admitted recording. `fit_fidelity_threshold=0.9` is the provisional value from the I-CARE CPC1-vs-CPC5 audit (`project/mnps_v3/tests/ingest_jacobian_fidelity_handover.md`); it is **not yet frozen** against a non-clinical qualification set. When neither `rel_mse_baseline_median` nor `rel_mse_baseline_windows` diagnostics are supplied at all (the default everywhere except the opted-in overlays above), `summary/fit_identified` is `None` (gate not evaluated) and prior-release unconditional behavior is preserved exactly.

Interpretation note:

- Jacobian windows are not clamped when mathematically unusable
- instead, invalid windows are removed from the canonical export and recorded in the diagnostics / `geometry_contract`

---

### Group: `/features_raw`

Created when summarize exports the per-epoch empirical feature surface.

- **`/features_raw/values`** *(float32, shape `[T,K]`)*: raw feature matrix in original scale.
- **`/features_raw/names`** *(utf-8 strings, shape `[K]`)*: feature column names aligned to `values`.
- **`/features_raw/metadata/*`** *(shape `[K]` per field)*: machine-readable per-feature metadata, including usage flags and provenance.
- Group attrs on `/features_raw` (and identically on `/features_robust_z`, `/features_projection_z`): **`alignment`** = `per_timepoint`, **`export_transform`** (`none` / `strict_robust_z` / `projection_z`), **`feature_contract_version`** = `v1`, **`feature_order_hash`** (sha256 of the JSON-serialized `names` order), **`n_features`**.

When `conventional_eeg.enabled: true`, the exported feature surface may also
contain Tier 1 qEEG comparator columns prefixed with `eeg_conventional_`.
Current generic EEG comparator families are `relative`, `ratio`, `peak`, and
`complexity`. When the connectivity pack is enabled, the feature surface may
also include `eeg_conventional_connectivity_*` columns. When the coma pack is
enabled, the surface may also include `eeg_conventional_coma_*` columns.

For `ds003645` MEG shadow-mapping runs, `features_raw` may additionally expose
paired diagnostic and combined MEG columns:

### MEG measurement-safeguard surfaces

`/features_projection_z` is the transform-aware export surface: each row is
aligned exactly to `/time`, `/mnps_3d`, and `/features_raw`, while values have
the configured feature pipeline applied (for example `log10 → robust_z → clip`
for MEG power). It is written for **any** modality when the projection-transform
pipeline runs, not only MEG shadow-mapping runs — an EEG-only run with no MEG
channels at all can still emit `/features_projection_z` alongside
`/features_raw` and `/features_robust_z`. `/epoch_id` is an additive per-row
identity when available.

`/row_source` contains additive row-aligned source lineage. Standard columns
(`schema.py`'s `row_source_columns`, also confirmed in real exports):
- **`row_source`** *(str)*: e.g. `"set_eeg"`, `"fif_meeg"`, `"unknown"`.
- **`raw_file`** *(str)*: basename of the source file for each row.
- **`source_format`** *(str)*: `"neuromag_fif"`, `"eeglab_set"`, `"unknown"`.
- **`has_meg`**, **`has_eeg`**, **`has_mag`**, **`has_grad`** *(int8, 0/1)*: per-row modality/channel-family presence flags.
- **`is_simultaneous`** *(int8, 0/1)*: 1 when the row's EEG and MEG (or other paired modality) source signals were recorded simultaneously rather than paired from separate sessions/files (written by `mndm/src/mndm/pipeline/summary.py`).

Sensor-topographic QC reports are external JSON/CSV artifacts. They report
frozen helmet-sector coverage and reliability only. They do not add regional
labels to MNPS outputs and must not be read as cortical localization or
EEG–MEG coordinate harmonization.

- diagnostic sensor-family columns such as `meg_mag_alpha`,
  `meg_grad_alpha`, `meg_mag_permutation_entropy`
- combined shadow columns used by the ds003645 9D mapping such as
  `meg_delta`, `meg_alpha`, `meg_beta_alpha`, `meg_hjorth_mobility`,
  `meg_permutation_entropy`, `meg_highfreq_power_30_45`

The combined `meg_*` surface is built from MAG and GRAD diagnostics using the
configured `robust_z_then_median` policy and then flows through the standard
`project_features_v2` contract like any other feature family.

For embodied-anchoring runs, `features_raw` may also include additive
interoceptive/raw anchor columns such as:

- `ecg_hrv_hr_mean_bpm`
- `ecg_hrv_ibi_mean_ms`
- `ecg_hrv_sdnn_ms`
- `ecg_hrv_rmssd_ms`
- `ecg_hrv_pnn50`
- `ecg_hrv_nn_count`
- `ecg_hrv_artifact_fraction`
- `ecg_hrv_coverage_fraction`
- `ecg_hrv_quality_score`
- `qc_ok_ecg_hrv`

When `phase_anchor.enabled: true` in the dataset config, `features_raw` also
contains per-epoch **phase anchor** columns (Mål B / C of the Embodied Anchoring
Principle). These land directly in `features.parquet` during `mndm.cli features`
and are automatically carried into HDF5 by `mndm.cli summarize`:

| Column | Type | Description |
|--------|------|-------------|
| `phi_cardiac_mean` | float32 | Circular mean cardiac phase ∈ [−π, π) via linear RR interpolation |
| `phi_resp_mean` | float32 | Circular mean respiratory phase ∈ [−π, π) via Hilbert transform |
| `rr_interval_ms` | float32 | Mean RR interval in epoch (ms) |
| `hr_bpm` | float32 | Mean heart rate (bpm) |
| `resp_rate_bpm` | float32 | Respiratory rate (bpm) from Hilbert phase advance |
| `inhale_fraction` | float32 | Fraction of epoch samples in inhale phase (0–1) |
| `hep_amplitude` | float32 | HEP mean frontal-EEG amplitude 200–600 ms post-R-peak |
| `n_rpeaks_in_epoch` | int | R-peak count in epoch |
| `pa_cardiac_quality` | float32 | Quality: fraction of expected beats detected (0–1, baseline 75 bpm) |
| `pa_resp_quality` | float32 | Quality: finite fraction of phi_resp samples (0–1) |

Missing modalities produce NaN-filled columns rather than errors:
- **RichSleep**: ECG ✓ + Resp ✓ → full output
- **ANPHY**: ECG ✓, no Resp → cardiac arm only; resp columns → NaN
- **BOAS**: no ECG, Resp ✓ → respiratory arm only; cardiac columns → NaN

Config key: top-level `phase_anchor:` in the dataset YAML (see `CONFIG_GUIDE.md`).

When `features.ecg.hrv.complexity.enabled: true` (requires `antropy` + `nolds`),
two additional nonlinear complexity columns are appended:

- `ecg_hrv_sampen` — Sample Entropy (order 2, tolerance 0.2 × std(nn)).
  Measures RR-interval irregularity; higher → more complex, less predictable
  beat-to-beat pattern.  Returns `NaN` when fewer than `min_nn_for_sampen`
  (default 50) valid intervals are available.
- `ecg_hrv_dfa_alpha1` — Short-range DFA scaling exponent α₁, computed over
  lag range 4–11.  α₁ ≈ 0.5 → uncorrelated noise; α₁ ≈ 1.0 → healthy 1/f
  long-range correlation; α₁ > 1.5 → over-correlated / pathological.
  Returns `NaN` when fewer than `min_nn_for_dfa` (default 16) intervals are
  available.

---

### Group: `/features_robust_z`

Created when summarize exports the strict robust-z feature surface.

- **`/features_robust_z/values`** *(float32, shape `[T,K]`)*: strict robust-z feature matrix.
- **`/features_robust_z/names`** *(utf-8 strings, shape `[K]`)*: feature column names aligned to `values`.
- **`/features_robust_z/metadata/*`** *(shape `[K]` per field)*: machine-readable per-feature metadata aligned to `names`.

Important:
- this surface is **strict robust-z only**
- projection-only steps such as `log10` and `clip` remain represented in provenance metadata and `feature_baselines`, not baked into `features_robust_z`
- new guarded exports add `robust_z_valid` (**`int32`**, not `int8` — see
  "Compression / dtypes" below), `robust_z_invalid_reason`
  (`""`, `degenerate_scale`, or `insufficient_support`), and
  `robust_z_finite_count` to `/features_robust_z/metadata/*`
- under the default `degenerate_scale_policy: nan`, a low-support or
  MAD-degenerate column is all `NaN`; `eps_floor` is a legacy-reproduction
  opt-in, not a coordinate-normalization setting

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
- **`alignment`** = `per_timepoint`, and (best-effort) **`normalize_mode`** copied from the layer payload when set.

---

### Groups: MNDM 2.1 Jacobian layers

Parallel to the coordinate layers above, the writer additively serializes
Jacobian estimates keyed by coordinate contract, via `payload.jacobian_layers`
(`_write_jacobian_layers_group` in `core/src/core/io/h5_writer.py`). This group
family is also recorded in the root attr `available_jacobian_layers` (see
"Root: HDF5 attributes" above), and is written whenever the pipeline computes
an anchored Jacobian.

- **`/jacobian_subject_anchored/J_hat`** *(float32, shape `[W,D,D]`)*, **`/jacobian_subject_anchored/J_dot`** *(float32, shape `[W,D,D]`)*, **`/jacobian_subject_anchored/centers`** *(int32, shape `[W]`)*: Jacobian estimated directly on `/coords_3d_subject_anchored` values.
- **`/jacobian_9D_subject_anchored/J_hat`**, **`/jacobian_9D_subject_anchored/J_dot`** *(float32, shape `[W2,K,K]`)*, **`/jacobian_9D_subject_anchored/centers`** *(int32, shape `[W2]`)*: Jacobian estimated directly on `/coords_9d_subject_anchored` values.
- Cohort-anchored counterparts (`/jacobian_cohort_anchored/*`, `/jacobian_9D_cohort_anchored/*`) follow the same pattern when a cohort/external anchor is configured, mirroring `/coords_3d_cohort_anchored` / `/coords_9d_cohort_anchored`.
- Group attrs: **`schema_version`** = `mndm.jacobian_layer.v2.1` (defaulted if absent).
- The set of layer names actually written for a given file is recorded in the root attr **`available_jacobian_layers`** and in **`/provenance/anchoring/available_jacobian_layers`** (see "Group: `/provenance`").
- These layers are additive copies alongside the canonical `/jacobian` and `/jacobian_9D` groups (which are estimated on the un-anchored, legacy `/mnps_3d` / `/coords_9d` values); they do not replace or alter `/jacobian` or `/jacobian_9D` semantics.

---

### Groups: anchored coordinates vs embodied anchors

The repository now exposes two different kinds of "anchor" surfaces that should
not be conflated:

- **`/feature_anchors/*`**: frozen cohort/external anchor statistics used to
  construct cohort-anchored coordinate layers
- **`/anchor_state/*`, `/anchor_quality/*`, `/anchor_coupling/*`**: additive
  embodied/interoceptive state aligned to the MNPS time grid

The first is a coordinate-normalization contract. The second is the MNDM 2.3
Embodied Anchoring Principle.

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

This group is about cohort/external feature scaling for anchored coordinate
contracts. It is intentionally distinct from the embodied-anchoring groups
below, which carry time-aligned physiological/interoceptive state on `/time`.

- **`/anchor_state/values`** *(float32[T, Qa], optional)*: aligned anchor-state matrix on the MNPS grid.
- **`/anchor_state/names`** *(str[Qa], optional)*: anchor-state column names.
- **`/anchor_state_dot/values`** *(float32[T, Qa], optional)*: finite-difference or derivative companion to `anchor_state`.
- **`/anchor_quality/values`** *(float32[T, Qq], optional)*: aligned anchor quality/support matrix.
- **`/anchor_quality/names`** *(str[Qq], optional)*: anchor-quality column names.
- **`/anchor_coupling/*`** *(optional)*: additive body-brain coupling diagnostics such as cross-block Jacobian estimates or summary metrics.

Typical `anchor_state` sources include ECG-, PPG-, pupil-, or respiration-derived
families when those modalities are available. In the current HRV v0.1 path,
`anchor_state` prefers superwindow HRV columns over older short-window ECG
surfaces when both exist.

Guarded AnchorState exports retain `mndm.anchor_quality.v1` for compatibility
and append a `quality_surface = "v2"` attr. For every emitted component and
composite, `<name>_eligible` records raw-modality eligibility and
`<name>_valid` records post-provenance/post-scale validity (both 0/1 float32).
`anchor_valid_fraction` is the fraction of valid v0.1 components. Invalid
components and all-invalid composites are `NaN`, so event-locked exports carry
the same missing value rather than a numerical sentinel. `summary.json` also
records `anchor_state_validation` (finite support, IQR, maximum magnitude,
scale warnings, thresholds, and guard-policy version).

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

- **`/coverage/axis_fraction`** *(float32, shape `[T,3]`)*: **measured** per-window direct-axis coverage for `[m,d,e]`, in `[0,1]`. This is the achieved-coverage quantity; on fully-supported files it can be identically `1.0`.
- **`/coverage/axis_names`** *(utf-8 strings, shape `[3]`)*
- **`/coverage/min_axis_coverage`** *(scalar float32)*: **not a measurement** — a policy floor copied verbatim from config (`mnps_projection.min_axis_coverage`, default `0.30`), matching the existing `min_*`-prefixed admission-policy convention (`coverage_min_seconds_effective`, `coverage_min_epochs_effective`). Windows with `axis_fraction < min_axis_coverage` on any axis fail `/qc/windows/coverage_ok` and may be dropped under `missing_axis_policy: nan_mask_v1`. Do not read `axis_fraction == 1.0` alongside `min_axis_coverage == 0.30` as "only 30% coverage was achieved" — those are two different, independently-scaled fields. Also copied as the root attr `min_axis_coverage`. Distinct from root attrs `direct_axis_coverage_{m,d,e}_min`, which *are* measured per-axis minima of `axis_fraction`.
- **`/coverage/coordinate_layers_present`** *(utf-8 strings, shape `[L]`)*
- **`/coverage/coordinate_contracts_present`** *(utf-8 strings, shape `[Lc]`)*
- **`/coverage/jacobian_centers`**, **`/coverage/jacobian_9d_centers`** *(int32, optional)*: explicit mappings back to the shared MNPS time index
- Group attrs:
  - **`_schema_version`** = `mndm.coverage.v1`

---

### Group: `/provenance`

Created when summarize exports structured additive provenance blocks.

- **`/provenance/contract/*`**: export-contract metadata such as `export_contract_version`, `config_digest_sha256`, `config_filename`, `run_manifest_ref`, `geometry_contract_status`, `geometry_invalidity_policy`
- **`/provenance/geometry_contract/*`**: always-on mathematical invalidity contract for canonical geometry exports (same object as `manifest.geometry_contract`, serialized as a first-class H5 group: `policy_version`, `primary_requires_coords_9d`, `status`, `shared_time_grid/*`, `time_grid/*`, `mnps_3d/*`, `coords_9d/*`, `jacobian/*`, `jacobian_9d/*`)
- **`/provenance/anchoring/*`**: explicit coordinate contracts/layers available in this H5 plus primary contract/layer and optional anchor identity. Observed fields: `available_coordinate_contracts`, `available_coordinate_layers`, `available_jacobian_layers`, `primary_coordinate_contract`, `primary_coordinate_layer`, `realized_contracts`, `requested_contracts`, `skipped_contracts_with_reason`
- **`/provenance/anchor_state`** *(optional, additive)*: reserved for future embodied-anchor provenance mirrored from `payload.provenance["anchor_state"]`; may be written as an **empty group** (no children) when no embodied-anchor provenance is populated for a run — this is expected, not an error. See the separate root `/anchor_state/*` values matrix under "Groups: anchored coordinates vs embodied anchors", which is a different, unrelated node despite the similar name.
- **`/provenance/normalization/*`**: concise normalization status/method/scope and sidecar references
- **`/provenance/event_stage_mapping/*`**: event/stage mapping versioning, source column/path, and codebook hash
- **`/provenance/signal_support_provenance/*`** *(optional)*: signal-support / temporal-continuity provenance for this export, distinct from `geometry_contract` and `coverage`. Observed fields: `schema` (e.g. `mndm.signal_support_export.v1`), `status`, `temporal_support_status` (e.g. `unknown` when effective filter/continuity support is not certified by epoch bounds alone), `temporal_support_reason`, `source_quality_status`, `source_quality_intervals_json` (JSON blob of source-signal quality intervals), `execution_records_json` (JSON blob of the concrete preprocessing/extraction steps executed for this file), `per_epoch_input_extent_json` (JSON blob of per-epoch raw-input extent), and `reference_fit_population/{status,features/*}` when a reference-population fit was used. This is validity-relevant provenance — a `temporal_support_status` other than a fully-certified value means downstream continuity-sensitive analyses should treat the epoch bounds alone as insufficient evidence of continuous coverage.
- **`/provenance/mapping/*`** *(optional)*: modality-specific mapping contract
  metadata for runs such as ds003645 MEG shadow mapping. Typical fields include
  `modality`, `mapping_family`, `mapping_reference`, `sensor_types`,
  `feature_combination`, `primary_surface`, and `validation_pilot`.
- Group attrs:
  - **`_schema_version`** = `mndm.provenance.v1`

---

### Group: `/qc`

Created when summarize exports additive per-window QC.

- **`/qc/windows/retained_after_qc`** *(int8, shape `[T]`)*
- **`/qc/windows/rejected_flag`** *(int8, shape `[T]`)*
- **`/qc/windows/qc_ok_eeg`**, **`/qc/windows/qc_ok_ecg`**, **`/qc/windows/qc_ok_eog`** *(int8, optional)*: two-stage gate, added 2026-09-11. Stage 1 (feature-table level, computed unconditionally): `1` = core bands finite, `0` = core bands not finite. Stage 2 (export level, per-recording): if an artifact-reduction method (ICA / EOG regression) is **not confirmed** to have actually run and modified the signal for this recording (per the preprocess QC sidecar's `artifact.applied`), the *entire* exported array is overwritten to **`-1` (not assessed)**, regardless of whether the underlying stage-1 value was `0` or `1` — a `0` can therefore never appear in the export unless `artifact.applied is True` for every underlying file in this recording's grouping. `1` only ever appears in the export when both stages pass: core bands finite AND artifact-reduction confirmed. Do not read `qc_ok_eeg=1` as "an artifact detector passed this window" unless it is actually `1` under this rule, and do not read `qc_ok_eeg=0` as "this dataset has an artifact detector but this window failed it" -- `-1` is the far more common outcome for any dataset that has not configured/confirmed artifact rejection (e.g. I-CARE, whose `preprocess.artifacts.method` is `"none"`). This does not change epoch retention: the `eeg_only` QC filter policy reads the unmodified stage-1 feature-table column, not this export, and is unaffected by the `-1` state (see `Output_variables_guide.md`'s coverage/QC filter notes), matching pre-2026-09-11 behavior for datasets that never configure artifact rejection at all.
- **`/qc/windows/coverage_ok`** *(int8, shape `[T]`, optional)*: `1` iff `axis_fraction >= min_axis_coverage` (both defined above) on all three axes for that window, else `0`.
- **`/qc/windows/mnps_3d_valid`** *(int8, shape `[T]`, optional)*: per-window finite validity for retained 3D MNPS rows.
- **`/qc/windows/coords_9d_valid`** *(int8, shape `[T]`, optional)*: per-window finite validity for retained stratified coordinates.
- **`/qc/windows/geometry_valid`** *(int8, shape `[T]`, optional)*: joint retained-window validity across available geometry surfaces.
- **`/qc/windows/stage_transition_flag`** *(int8, shape `[T]`, optional)*
- Group attrs (on `/qc/windows`):
  - **`_schema_version`** = `mndm.qc.windows.v1`
- The parent **`/qc`** group itself also carries **`_schema_version`** = `mndm.qc.v1` (distinct from the child `/qc/windows` tag).

This group intentionally carries only the light-weight, per-window contract.
Heavier QC summaries still live in `qc_summary.json` and `qc_reliability.json`.

#### ICA and bad-channel provenance in `qc_summary.json`

When `preprocess.artifacts.method: "ica"` is active, the following fields are
written into `qc_summary.json` under `artifacts`:

| Field | Type | Description |
|-------|------|-------------|
| `artifacts.method` | str | Artifact method used: `"ica"`, `"none"`, or `"autoreject"` |
| `artifacts.ica_n_components_fit` | int | Actual number of ICA components fitted |
| `artifacts.ica_n_excluded` | int | Number of components excluded (eye + cardiac) |
| `artifacts.ica_eog_proxy_channels` | list[str] | Channel names temporarily retyped as EOG |
| `artifacts.ica_ecg_channel` | str\|null | ECG channel used for cardiac component detection |
| `artifacts.bad_eeg_channels` | list[str] | Channel names flagged and dropped/interpolated |
| `artifacts.n_bad_eeg_channels` | int | Count of bad EEG channels detected |

When ICA is disabled (`method: "none"`), `artifacts.ica_n_excluded` is absent
and `artifacts.method` = `"none"`.

The per-subject bad-channel list is also echoed into `run_manifest.json` under
`extra.preprocess_artifacts` for run-level aggregation.

---

### Group: `/jacobian_9D`

Created if v2 Jacobians exist.

- **`/jacobian_9D/J_hat`** *(float32, shape `[W2,K,K]`)*: Jacobian in Stratified MNPS v2 space.
- **`/jacobian_9D/J_dot`** *(float32, shape `[W2,K,K]`)*: per-window Jacobian derivative estimate. Same shape as `J_hat` (see the `/jacobian/J_dot` note above — this is **not** a `[W2-1,K,K]` temporal difference).
- **`/jacobian_9D/centers`** *(int32, shape `[W2]`)*: center index.
- **`/jacobian_9D/affine_reference`**, **`/jacobian_9D/affine_intercept`** *(float32, shape `[W2,K]`, optional)*: affine fit parameters for the directly estimated 9D Jacobian.
- **`/jacobian_9D/derived_metrics/v1/`** *(optional)*: the same Jacobian Metrics v1 contract evaluated directly on the 9D Jacobian; it is not synthesized from 3D outputs. Same Round-2 certificate siblings as `/jacobian/derived_metrics/v1/`.

Optional subgroup:
- **`/jacobian_9D/cross_partials/<name>`** *(float32, shape `[W2]`)*: selected elements from the v2 Jacobian as 1D series (dataset names are sanitized; `/` is replaced with `_`).

### Local-dynamics extension groups (optional)

- **`/finite_time_response/v1/{primary,stratified_9d}/`**: opt-in finite-time response summaries with explicit horizon, continuity, timebase, propagator, computation, and validation semantics. A branch is emitted only when it was computed. `computation_status` is one of `not_requested`, `unavailable`, `invalid`, or `computed`; `validation_level` is independent (method-validation; default `model_derived` is not `measurement_validity`). Sibling certificate: `measurement_validity` is `not_assessed` when computed and `not_applicable` otherwise; `claim_status` is `no_biological_claim` on new writes. The peak-gain analogue of theory \(G_{\mathrm{peak}}\) is `summary.g_peak_over_horizons` (with `log_g_peak_over_horizons`, `tau_peak_over_horizons`, `peak_horizon_steps`). FTR is not FAR, not perturbational, and not S3-licensed empirical NDT.
- **`/residual_covariance_proxy/v1/`**: PSD-regularized residual covariance proxy and mandatory time-semantic/QC provenance. The current subject summarize pipeline does not emit this surface because it lacks an exported one-step transition residual. New writes carry the same three-way certificate; computed proxies are `not_assessed`, not NDT-licensed.
- **`/transition_residuals/v1/{primary,stratified_9d}/`**: opt-in, cross-fitted one-step state prediction and residual series. Each record carries source/target window and center indices, observed `dt_sec`, predicted state, residual, and coordinate/fit provenance. Sibling certificate fields as above (`not_assessed` only when `computation_status=computed`).
- **`/transition_residual_covariance_proxy/v1/{primary,stratified_9d}/`**: recording-level covariance of accepted cross-fitted transition residuals. It is unavailable for materially irregular transition steps or insufficient support. This proxy is not biological process noise. Under the Gate F freeze it is the only admissible Q for opt-in discrete `W_Q`.
  - Each branch carries `covariance` (`[D,D]` or `[K,K]` float64) plus a wider provenance field set that determines whether `covariance` should be trusted as a usable Q. The most validity-relevant fields (not exhaustive): `crossfit_status`, `degrees_of_freedom_policy`, `prediction_fit_policy`, `effective_rank`, `q_rank`, `q_min_eigenvalue`, `q_psd_correction` (bool: whether a PSD projection/floor was applied), `q_psd_post_dtype` / `q_floor_met_post_dtype` (bool: whether the PSD/floor condition still holds after the final output dtype cast), `q_regularization`, `q_shrinkage`, `q_requested_min_eigenvalue`, `q_max_dt_deviation_sec`, `q_n_samples`, `q_scope`, `q_semantics`, `q_time_semantics`, `q_units`, `q_input_scale` / `q_output_scale` / `q_output_dtype`, `coordinate_contract`, `coordinate_layer`, `conversion_model`, `numerical_precision`, `residual_mean` / `residual_mean_norm`. Reading `covariance` without checking `q_psd_correction`/`q_psd_post_dtype`/`crossfit_status` risks treating an ill-conditioned or non-PSD-corrected residual estimate as a clean Q.
  - `/stochastic_reachability/v1/*` carries the same `q_*`/`w_q_projection_qc/*` provenance vocabulary (prefixed `w_q_*` there) alongside its own `a_q`, `c_1_q`, `d_eff`, `v_norm`, `conversion_model`, `n_propagator_steps` fields — see the `failure_reason` note below for how a branch reports non-computability instead of a degraded numeric `w_q`.
- **`/stochastic_reachability/v1/{primary,stratified_9d}/`**: opt-in (`local_dynamics.stochastic_reachability.enabled`, default false) discrete reachability \(W\leftarrow\Phi W\Phi^\top+Q\) from Gate E Q and \(\Phi=\mathrm{expm}(J_{\mathrm{crossfit}}\Delta t)\). Not `/dynamical_families/spread`. Documented identity: `finite_time_reachability_level4` when `n_propagator_steps>1`; one-step `w_q` is `transition_reachability_covariance_level2`. Predictive spread, not controllability, occupancy, or empirical future covariance. `d_eff` is a derived scalar of `W_Q`, not `observed_future_effective_dimension_level0` and not `reachability_effective_dimension_level4`. `observed_future_spread_level0` is superseded (state-matched spread is level1 and is not written). Certificate when computed: `not_assessed` / `no_biological_claim`. Grain: `recording_horizon`. Analysis-repo I-CARE / coma reachability products (for example `tube_d_eff_median`) are **not** this schema.
  - **`primary` and `stratified_9d` fail independently.** Each branch carries its own `computation_status`; one branch can be `computed` (with a full numeric `w_q`, `a_q`, `c_1_q`, `d_eff`, `v_norm`, and `w_q_projection_qc/*` block) while the other is `invalid`/`unavailable` with a `failure_reason` string (observed values include `post_dtype_psd_failure`, `reachability_numerical_overflow`, and `upstream_jacobian_local_fit_not_identified`) and **no** `w_q`/numeric fields at all. Do not assume both branches succeed or fail together, and do not treat a missing `w_q` on one branch as evidence about the other branch's validity.
  - **Upstream Jacobian fit-fidelity gate (added 2026-09-11):** the summarize pipeline passes the sibling `/jacobian/derived_metrics/v1` (or `/jacobian_9D/derived_metrics/v1` for `stratified_9d`) fit-fidelity gate (`summary/fit_identified`, `summary/rel_mse_baseline_median`) into this computation. When that gate says the local affine fit behind Φ did not beat the no-dynamics baseline (`fit_identified=False`), reachability reports `computation_status="unavailable"`, `failure_reason="upstream_jacobian_local_fit_not_identified"`, and `provenance/upstream_rel_mse_baseline_median`, **before** attempting the \(W\leftarrow\Phi W\Phi^\top+Q\) recursion. This targets the failure mode where an unidentified/expanding generator (\(\alpha>0\)) pushed through `expm` makes `reachability_numerical_overflow` the modal outcome instead of a rare numerical accident (see `project/mnps_v3/tests/ingest_jacobian_fidelity_handover.md`, evidence S2). When the upstream gate was not evaluated (older callers, or the Jacobian family unavailable), this check is skipped and prior-release behavior is unchanged.

### `failure_reason` field

Several v1 groups above carry an additional sibling dataset `failure_reason`
(a UTF-8 string) whenever `computation_status` is **not** `computed`. It is
populated on:
- `/jacobian/derived_metrics/v1/failure_reason` and `/jacobian_9D/derived_metrics/v1/failure_reason` when `computation_status != computed` (observed reasons include `no_finite_metric_windows`, and, when the opt-in fit-fidelity gate is enabled and evaluated, `local_linear_fit_not_better_than_baseline` and `fit_fidelity_unknown` — see the fit-fidelity gate note above);
- `/dynamical_families/destination/v1/failure_reason` and `/dynamical_families/resilience/v1/failure_reason` when their `computation_status` is `not_testable` (observed reasons include `explicit_A_B_reaction_coordinate_contract_not_configured` and `no_perturbation_protocol`);
- `/dynamical_families/drift/v1/{realized_velocity_level0,conditional_mean_rate_level1/*}/failure_reason` when a nested identity is not computed (observed reasons include `insufficient_local_support`); `lag_inconsistent` is a diagnostic label on `lag_diagnostics` and does not erase `pooled`; `ito_not_qualified` is a qualification label, not a substitute for `ito_qualified`;
- `/dynamical_families/one_step/v1/` parent and nested identity `failure_reason` when the map is not computed or not identified (observed reasons include `insufficient_local_support`, `one_step_fit_not_better_than_baseline`, `one_step_subject_anchored_3d_only`, `materially_irregular_increment_timestep`, `non_positive_nominal_dt`, `one_step_requires_two_temporal_blocks`, `one_step_embargo_shorter_than_declared_lag`; level-3 leaves use `upstream_one_step_not_identified` or `generator_matrix_log_not_real`; level-2 operator functionals use `not_testable` with `upstream_one_step_not_identified` when a map fit was denied, the parent refusal reason on early refusal, or `operator_phi_rank_deficient` / `operator_polar_reflection_not_rotation` / `operator_polar_rotation_log_not_real` when \(\Phi\) is identified but the functional fails closed). Denied identification still writes nested map leaves (`affine_one_step_map_level2` and siblings) as `insufficient_support` and level-3 as `not_testable`. Early lag-1 refusal synthesizes the three operator-functional leaves as `not_testable` (never zeros); map leaves may still be omitted on short recordings that cannot form two embargoed folds. When `declared_lags` includes 2, the nested `declared_lag_2/` group synthesizes map, functional, and level-3 leaves on early refusal.
- `/dynamical_families/diffusion/v1/summary.drift_alignment_failure_reason` for the specific `A_bD`/`R_b_over_a` sub-metrics (`independent_drift_not_supplied`) even while the rest of diffusion is `computed`;
- `/stochastic_reachability/v1/{primary,stratified_9d}/failure_reason` when that branch's `computation_status=invalid` (see above).
`failure_reason` is absent (not an empty string) when the branch is `computed`. This is the explicit-invalid contract in practice: prefer reading `failure_reason` over inferring failure from a missing numeric field.

### Validity certificate (v3 R2)

Every family and local-dynamics v1 group above may carry three sibling datasets:

- `computation_status`: possible values include `computed`, `insufficient_support` (Jacobian metrics, when no finite metric windows exist **or** the fit-fidelity gate rejects the recording's local linear fit, `failure_reason="local_linear_fit_not_better_than_baseline"`), `not_testable` (observed on destination/resilience families when their required contract inputs — e.g. explicit reaction coordinate, perturbation protocol — are not configured), `unavailable` (observed on `/stochastic_reachability/v1/*` when a required upstream input, including the Jacobian fit-fidelity gate, is missing or rejected), and `invalid` (observed on local-dynamics branches such as `/stochastic_reachability/v1/*` that attempted computation but failed numerically). When `computation_status` is anything other than `computed`, the group typically also carries a sibling `failure_reason` string (see the `failure_reason` note under "Local-dynamics extension groups" above).
- `measurement_validity`
- `claim_status`

These are additive keys on family and local-dynamics schema IDs (`mndm.jacobian_metrics.v1`,
`mndm.finite_time_response.v1`, `mndm.transition_residuals.v1`,
`mndm.diffusion_geometry.v1`, `mndm.chart_drift.v1`, `mndm.affine_one_step.v1`,
`mndm.committor.v1`, `mndm.finite_amplitude_resilience.v1`, residual-covariance
and reachability v1). Certificate keys do not rename those schema IDs.

On **new writes**, `claim_status` is always `no_biological_claim`.
Diffusion that the estimator can support is `computed` /
`measurement_validity=not_assessed`; OD-TQ1 id/hash in `provenance` are
method tags, not regime validity. `measurement_validity` is
`translation_qualified` only for a computed **destination or resilience**
family that already records a TQ id and contract hash. Jacobian metrics and
FTR stay `not_assessed` even when series are finite. `validation_level` is
not `measurement_validity`.

**Legacy read:** if a field is absent, readers must report `not_recorded`.
Do not infer `translation_qualified` from `qualification_id` or
`validation_level`. Do not default a missing `claim_status` to
`no_biological_claim` (copy `provenance/claim_status` only when that dataset
exists). Always record `certificate_origin`: `canonical`,
`legacy_promoted_provenance`, or `legacy_absent`.

### Inferential grain (v3 R3)

The same v1 groups carry a nested `grain/` mapping:

- `native`, `parent` ∈ {`window`, `transition`, `recording_horizon`, `event`, `recording`, `subject`}
- `biological_unit` = `subject` on new writes
- `repeated_measure` = `true` or `false`
- `direct_between_subject_inference` = `forbidden` on new writes

Typical natives: Jacobian metrics, diffusion, destination, and chart-drift
`window`; FTR and reachability `recording_horizon`; transition residuals
`transition`; Q-proxy and affine one-step `recording` (`repeated_measure=false`
on both; one-step window-shaped series are broadcasts of one operator, not
\(T\) replicates); FAR `event`. Grain is written even when
`computation_status` is not `computed`.

**Legacy read:** missing grain fields are `not_recorded`. Do not infer
`window` because `series/` exists, and do not infer `subject` from the path.
`grain_origin` is `canonical` or `legacy_absent`.

### Group: `/support_signature/v1/` (metadata; not a dynamical family)

Schema id: `mndm.support_signature.v1`. Additive file-level support /
capability signature. Does not alter `/mnps_3d`, `/coords_9d`, `/jacobian`,
or `geometry_contract`. Family schema IDs are unchanged.

```text
/support_signature/v1/
  modality                         eeg | ieeg | fmri | meg | unknown
  coordinates/{m_a,…,e_m}/
    source                         direct | fallback | mixed | not_applicable
    fallback_feature               feature name, or none
    semantic_equivalence           true | false | not_applicable
  axes_3d/{m,d,e}/
    source                         direct | mixed | not_applicable
  capability/
    chart_3d, chart_9d, mnj_3d, mnj_9d
    diffusion, spread, destination, resilience
```

Coordinate `source` is built from existing `mnps_9d.metric_policies` plus
already-recorded fallback metadata (entropy `actual_metric_used` /
`degraded_mode`, `embodied_arousal_proxy_source`). It is **not** a new
coverage fraction. `axes_3d` is `direct` when all three child 9D
coordinates are `direct`, else `mixed`.

Capability cells are the contract class for this file's modality (one row
of the CONFIG_GUIDE table), not HDF5 presence. `spread` is `gated` (Gate F).
`resilience` is `perturbational_only`. `destination` is
`no_generic_ingest`. `diffusion` is `overlay_only`. Capability `yes` is not
an NDT license.

**`mnj_9d` per-file downgrade (added 2026-09-11):** the static per-modality
class (EEG/MEG `conditional`, iEEG `yes`, fMRI `limited`) describes the
*modality*, not this file's actual 9D subcoordinate support. When
`coordinates/e_m/{source,semantic_equivalence}` is `fallback`/`false` (e.g.
EEG `e_m` falling back to `eeg_highfreq_power_30_45` instead of
`ecg_rmssd`/`eog_blink_rate`), `capability/mnj_9d` is downgraded to `gated`
for that file regardless of the static class, so a reader consulting only
`capability/mnj_9d` cannot miss the non-equivalent axis. `mnj_3d` is
unaffected (the 3D canonical chart does not consume the 9D split). An
already-`not_assessed` `mnj_9d` (unknown modality) is left as-is, never
"upgraded" to `gated`.

**Legacy read:** missing `/support_signature/v1` or missing children are
`not_recorded`. Do not infer `chart_3d=yes` from `/mnps_3d`. Do not infer
`spread=gated` from a missing reachability group. `signature_origin` is
`canonical` or `legacy_absent`.

### Group: `/dynamical_families` (optional, overlay-gated)

Written only when `dynamical_families.enabled` is true in an explicit overlay.
New files write the canonical tree only. Readers may load a legacy
`/orthogonal_dynamics/` group when the canonical path is absent and must then
set `provenance.namespace_origin = "legacy_orthogonal_dynamics"`. Dual-write
is not performed.

Diffusion, destination, and resilience schema IDs are unchanged from v3.0.
Chart-drift (`mndm.chart_drift.v1`) and affine one-step (`mndm.affine_one_step.v1`)
are additional writable families. Family YAML `spread` remains `gate_closed` and is
not written here. Opt-in `mndm.stochastic_reachability.v1` lives at
`/stochastic_reachability/v1`, not under `/dynamical_families`. Attractor,
basin, persistence, recurrence, hysteresis, and observational recovery are
withheld register identities, not HDF5 groups. Recurrence is not an
attractor. Hysteresis is not FAR. Matching is not automatically level2.
\(-\log(P_{RR})/\Delta t\) is transformed retention, not an escape rate.

- **`/dynamical_families/diffusion/v1/`**: `mndm.diffusion_geometry.v1`. Local increment-covariance diffusion geometry. `contract_status=standard` is the schema class, not an empirical license. Ingest uses C1 defaults (`drift=None`, `residualize_increments=False`): `a_hat` is the documented identity `conditional_covariance_rate_level1` (centered increment covariance over nominal \(\Delta t\), convention \(a\) not \(D=a/2\); `a_semantics=raw_increment_covariance` means not residualized). The series name `a_hat` is not renamed. Nested `increment_covariance_level0` is the unconditional centered \(\mathrm{Cov}(\Delta X)\) (not divided by \(\Delta t\), not local kNN, not `a_hat`; YAML nested name is refused). `series/source_idx` and `summary/transition_support_id` identify lag-1 source transitions (not kNN neighborhoods). `source_idx` has length `n_increment_pairs`, not \(T\), and is not aligned to `a_hat[t]`. `D_total` / `d_diff` / `c_diff` may be `computed`, but `A_bD` and `R_b_over_a` are NaN with `summary.A_bD_computation_status=not_testable`, `summary.R_b_over_a_computation_status=not_testable`, and `drift_alignment_failure_reason=independent_drift_not_supplied`. Do not read those NaNs as zero alignment. `d_diff` is a derived scalar of `a_hat`, not `diffusion_effective_dimension_level1`. `ito_diffusion_tensor_level3` is not written. Jacobian residuals, MNPS \(\dot x\), and Jacobian intercepts are not diffusion \(a\) or SDE drift. Library C1 may consume an externally qualified chart \(b\) without changing `a_hat` (Gate C1-A). The opt-in chart-drift family is not an independent \(b\) for these alignment scalars. C2 residualization is not authorized. Legacy read path: `/orthogonal_dynamics/diffusion_geometry/v1`.
- **`/dynamical_families/destination/v1/`**: `mndm.committor.v1`. Production adapter is 1-D O2b (`local_law_dense_grid_o2b`) with explicit first-hit A/B + reaction coordinate. Ingest `q_A_to_B` is the documented identity `restricted_1d_local_law_quadrature_q` (not `generator_committor_level3`; `interpretation_level` is unnumbered and recorded as `summary/interpretation_level_token=not_numbered`). The first-hit estimator uses the same series name as `destination_first_hit_fraction_resolved_level1` (x-conditioned, not level0). That resolved-only mean is not `destination_hit_probability_level1` (including unresolved) and not `destination_unresolved_fraction_level1`. `series/resolved_first_hit_outcome` is a 0/1/NaN encoding, not those leaves. O2b does not serialize \(V_{1/2}\) or \(\lvert\nabla q\rvert\). Stage labels are not committor truth. Legacy read path: `/orthogonal_dynamics/committor/v1`.
- **`/dynamical_families/resilience/v1/`**: `mndm.finite_amplitude_resilience.v1`. Protocol-gated FAR object. Existing `amplitude_curve` / `basin_return_probability` is the documented identity `far_recovery_probability_level4`. Spontaneous return and excursion recovery are not FAR and are not written. Existing `r50_discrete_first_bin_at_or_below_half` is not `far_threshold_p50_level4`. Observational return / \(R(\rho)\) from spontaneous trajectories is not an ingest product. Legacy read path: `/orthogonal_dynamics/finite_amplitude_resilience/v1`.
- **`/dynamical_families/drift/v1/`**: `mndm.chart_drift.v1`. Opt-in (default off) 3D subject-anchored identities under `realized_velocity_level0/` and `conditional_mean_rate_level1/{pooled,blocked_crossfit,lag_diagnostics}/`. The pooled field is a local conditional increment mean over nominal \(\Delta t\), not identified Itô \(b\). Pooled `series/source_idx` shares `transition_support_id` with diffusion lag-1 pairs when both compute; that is source-transition identity, not identical neighborhoods. `source_idx` has length `n_increment_pairs`, not \(T\). Cross-fit is a level-1 `variant_id` with its own embargoed `support_id` (`embargo_semantics=index_steps`), not a level upgrade, and is not wired into diffusion `drift_source`. Multi-lag consistency is diagnostics; `ito_drift_level3` is not written and `ito_qualified` is never auto-written. `/mnps_3d_dot` remains the Savitzky–Golay sibling (`smoothed_velocity_savgol_level0` in the register only) and is not an alias of `realized_velocity_level0`. Coverage is fail-closed; invalid windows are NaN, not zero.
- **`/dynamical_families/one_step/v1/`**: `mndm.affine_one_step.v1`. Opt-in recording-level lag-1 affine map \(x_{t+\Delta}\approx\Phi(x_t-\bar x)+c\) (`estimator=recording_affine_discrete_map`). Common YAML default is off. PhysioNet I-CARE 2.1 dynamical-families overlays may set `enabled: true` as a coverage opt-in; they do not retune the 0.9 threshold. The same overlays may set `declared_lags: [1, 2]` so lag-2 identities write; common YAML `declared_lags` remains `[1]`. Identification uses two chronological blocked-holdout folds plus an index embargo (`embargo_semantics=index_steps`) and `mndm.one_step_fit_fidelity.v1`: identified iff the median of the two fold rel-MSE scores is **strictly less than** 0.9 versus a mean-next-state baseline. That gate is not translation qualification and not the I-CARE `jacobian_metrics.fit_fidelity_gate`. Level-2 `qualification_status` is `one_step_identified` or `one_step_not_identified`. Grain is `native=recording`, `repeated_measure=false` (broadcast window series are not \(T\) replicates). Nested identities: `affine_one_step_map_level2` (`phi_hat`, `affine_reference`, `affine_intercept`), `conditional_affine_mean_rate_level2` (`affine_mean_rate`), `innovation_covariance_level2` (affine residual `innovation_covariance`, not diffusion `a_hat`; C2 stays closed). Shared series include `rel_mse_baseline`. Euclidean functionals of a qualified \(\Phi\) (`operator_max_gain_rate_level2` \(=\log\sigma_{\max}(\Phi)/\Delta t\), `operator_volume_gain_rate_level2` \(=\log\lvert\det\Phi\rvert/\Delta t\), `operator_rotation_rate_level2` from polar \(\Phi=RP\)) write with `qualification_status=one_step_functional_of_qualified_map_not_independent_oos`. They are not spectral abscissa, not `finite_time_peak_gain_level4`, and not generator-proxy rotation. Rank-deficient volume is `insufficient_support` / `operator_phi_rank_deficient`, not epsilon-rescued. Polar reflection or a non-real \(\log R\) fails closed. The same three identities nest under `declared_lag_2/` when that lag is requested. Level-3 leaves (`spectral_abscissa_level3`, `numerical_abscissa_level3`, `divergence_level3`, `generator_rotation_norm_level3`, plus `generator_log_ok`) write only when the map is identified and `logm(Φ)/Δt` is real (`qualification_status=generator_proxy_from_qualified_one_step_not_ito`). Not `expm(J_hat dt)`, not `/jacobian/derived_metrics` abscissa, not `ito_drift_level3`, not `reactivity_gap_level3`, not `operator_gain_anisotropy_level2`, not a `drift_source` for \(A_{bD}\). Iterating \(\Phi\) is level 4 (`iterated_one_step_horizon_map_level4`) when the lag-1 map is identified and blocked holdout at horizon \(2\Delta t\) is strictly less than 0.9; that object is not the direct lag-2 map. Direct lag-2 identities (`affine_one_step_map_level2/declared_lag_2` and siblings) write at `/dynamical_families/one_step/v1/declared_lag_2/` when YAML `declared_lags` includes 2; they are a recording-level affine map on pairs \((x_t, x_{t+2\Delta})\), not \(\Phi_1^2\). Lag-2 generator proxies (`spectral_abscissa_level3/declared_lag_2` and siblings) write in the same nested group from \(\logm(\Phi_2)/\mathrm{nominal\_dt}\) when that map is identified (`nominal_dt` is the median lag-2 span, \(\approx 2\Delta t\); `qualification_status=generator_proxy_from_qualified_one_step_not_ito`). Not `ito_drift_level3`. Common YAML `declared_lags` remains `[1]`. YAML keys `ito_drift_level3`, `ito_qualified`, `jacobian_expm`, `jacobian_expm_as_fitted_phi`, `multi_step_rollout`, `one_step_iteration_as_level2`, `phi_one_composed_as_lag2`, `affine_two_step`, `two_step`, `abscissa_from_unidentified_operator`, `innovation_as_diffusion_a_hat`, `mnps_xdot_as_one_step_map`, `peak_gain_from_phi_powers`, `finite_time_peak_gain_level4`, `operator_gain_as_spectral_abscissa`, `epsilon_rescued_volume_gain`, `reactivity_gap_level3`, `abscissa_difference_as_reactivity_gap`, `jacobian_reactivity_gap_as_level3`, `operator_gain_anisotropy_level2`, and `generator_symmetric_anisotropy_level3` are refused. 3D subject-anchored only.
- **`/dynamical_families/amplification/v1/`**: `mndm.amplification.v1`. Opt-in window-level identity `neighbor_gain_q90_level1` (`estimator=same_pair_neighbor_gain_q90`). Common YAML default is off. Production PhysioNet I-CARE dynamical-families overlays do not enable this family. Named `*_amplification_pilot.yaml` overlays may enable it for bounded first-hour coverage only. The estimand is the **per-source** 0.90 quantile of same-pair Euclidean neighbor gains \((d_{ij}^{(1)}+\varepsilon)/(d_{ij}^{(0)}+\varepsilon)\) at lag 1. The written object is a window series, not a pooled pair-level \(Q_{0.90}\) over the recording. `summary/neighbor_gain_q90_median` is the median of those per-source q90 values. Neighbors are selected among lag-1 sources and followed one real step; they are not re-selected at the successor. \(\varepsilon\) is a documented distance floor, not a volume rescue. `qualification_status=same_pair_observed_gain_not_operator_max_gain`. Grain is `native=window`, `repeated_measure=true`. Nested same-pair identities `neighbor_separation_rate_level1`, `neighbor_gain_rate_q90_level1`, and `cloud_volume_change_rate_level1` write when the family is enabled; they are not YAML toggles. Separation is the per-source median of \(\log((d_1+\varepsilon)/(d_0+\varepsilon))/\mathrm{nominal\_dt}\). Gain-rate is \(\log G_{q90}/\mathrm{nominal\_dt}\) of the already written q90. Cloud volume is \((\log\det(C_1+\varepsilon I)-\log\det(C_0+\varepsilon I))/(2\,\mathrm{nominal\_dt})\) on the same neighbor cloud; \(\varepsilon\) is a documented logdet floor, not operator-volume rescue. Not `operator_max_gain_rate_level2`, not spectral abscissa, not peak gain. `history_predictive_gain_level1` is a separate family, not written here. YAML keys `resample_neighbors_at_target`, `neighbor_gain_as_operator_max_gain`, `neighbor_gain_as_spectral_abscissa`, `history_predictive_gain_level1`, `neighbor_separation_rate_level1`, `neighbor_gain_rate_q90_level1`, `cloud_volume_change_rate_level1`, `neighbor_separation_as_spectral_abscissa`, and `cloud_volume_as_operator_volume` are refused. `q` is frozen at 0.90. 3D subject-anchored only.
- **`/dynamical_families/history/v1/`**: `mndm.history.v1`. Opt-in recording-level identity `history_predictive_gain_level1` (`estimator=frozen_affine_m0_m1_oos`) with nested `history_conditioned_operator_level2`. Common YAML default is off. Production PhysioNet I-CARE overlays and named amplification pilots do not enable this family. The L1 estimand is out-of-sample \(H_{\mathrm{gain}}=\mathrm{MSE}(M_0)-\mathrm{MSE}(M_1)\) on the **same** lag-1 triples (source \(t\) whose predecessor \(t-1\) is also a lag-1 source). Frozen affine \(M_0\): \(x_t\to x_{t+1}\). Frozen affine \(M_1\): \((x_t,x_{t-1})\to x_{t+1}\). Two chronological blocked-holdout folds with `embargo_steps=4` (`embargo_semantics=index_steps`). Window series `history_predictive_gain` / `mse_m0` / `mse_m1` are holdout broadcasts, not \(T\) replicates. L1 `qualification_status=history_error_reduction_not_markov_restoration`. Nested L2 is identified only when M1 itself passes the frozen one-step OOS gate `mndm.one_step_fit_fidelity.v1` (median of two fold rel-MSE scores **strictly less than** 0.9 versus a mean-next-state baseline). Positive \(H_{\mathrm{gain}}\) is not identification. Identified M1 writes a recording-level \(3\times 6\) affine (`phi_hat`, `affine_reference`, `affine_intercept`); it is not lag-1 \(\Phi\) and not a 3×3 chart map. L2 `qualification_status` is `history_conditioned_operator_identified_not_markov_restoration` or `history_conditioned_operator_not_identified`. Grain is `native=recording`, `repeated_measure=false`. Not Markov restoration. Negative \(H_{\mathrm{gain}}\) remains `computed`; L1 stays independent if L2 is not identified. `n_blocks` is frozen at 2. `embargo_steps` is frozen at 4. The 0.9 threshold is frozen. Both OOS folds must succeed. Series `source_idx` / `transition_support_id` are the filtered triples (`lag1_transition_support_id` is the unfiltered lag-1 set). YAML keys `history_as_markov_restoration`, `history_conditioned_operator_level2`, `history_augmented_generator_level3`, `history_augmented_propagator_level4`, `mnps_xdot_as_history_model`, and `one_step_phi_as_history_m0` are refused; L2 is a nested identity, not a YAML toggle. Level-3/4 history rungs remain withheld: not \(\logm\) of the \(3\times 6\) M1 map, not iteration of M1, not `ito_drift_level3`. 3D subject-anchored only.
- **`/dynamical_families/turning/v1/`**: `mndm.turning.v1`. Opt-in window-level identities `turning_angle_level0` and `turning_rate_level0` (`estimator=successive_increment_turning`). Common YAML default is off. Production PhysioNet I-CARE overlays and amplification pilots do not enable this family. The estimand is the Euclidean angle between consecutive lag-1 displacements \(v_t=x_{t+1}-x_t\) and \(v_{t+1}=x_{t+2}-x_{t+1}\), and that angle divided by the observed first-step \(\Delta t\). If either \(\lVert v\rVert\) is below `min_displacement` (default \(10^{-12}\)), the value is **undefined (NaN), not zero**. `qualification_status=realized_turning_not_operator_rotation`. Grain is `native=window`, `repeated_measure=true`. Not `operator_rotation_rate_level2`, not `generator_rotation_norm_level3`. `cloud_volume_change_rate_level1` writes under amplification, not turning. YAML keys `operator_rotation_as_turning`, `generator_rotation_as_turning`, `mnps_xdot_as_turning`, `zero_fill_undefined_direction`, and `cloud_volume_change_rate_level1` are refused. 3D subject-anchored only.

These groups do not alter `/mnps_3d`, `/coords_9d`, or `/jacobian`. Common EEG/fMRI/ephys profiles do not enable them. Chart-drift, one-step, amplification, history, and turning are not TQ-gated and never write `measurement_validity=translation_qualified`. Each family v1 group carries the Round-2 validity certificate (`computation_status`, `measurement_validity`, `claim_status`) and the Round-3 nested `grain/` object as siblings; `provenance.validation_level` remains the method-validation tag.

---

### Group: `/coords_9d`

Created if stratified coordinates exist.

- **`/coords_9d/values`** *(float32, shape `[T,9]`)*: Stratified MNPS subcoords in canonical order.
- **`/coords_9d/names`** *(utf-8 strings, shape `[9]`)*: subcoord names (canonical order: `m_a,m_e,m_o,d_n,d_l,d_s,e_e,e_s,e_m`).
- **`/coords_9d` attrs**:
  - **`version`** = `"9d"`

Important note:

- `coords_9d` is a separate validity surface from `mnps_3d`
- non-finite retained 9D rows are reported via `geometry_contract` and `/qc/windows/coords_9d_valid`
- when the active 3D contract is derived from v2 coordinates, non-finite 9D rows also trigger row dropping on the shared MNPS time grid

---

### Group: `/blocks`

Created when `block_native.datasets.<id>.enabled: true` and at least one block was inferred for the subject.

Columnar table — one dataset per column, all arrays of length **B** (number of inferred blocks):

- **`/blocks/block_id`** *(int32)*: monotonically increasing block index within the subject recording.
- **`/blocks/stage_code`** *(int32)*: integer stage/condition code assigned to the block (from the configured `stage_map`; `−1` if unmapped).
- **`/blocks/start_sec`** *(float32)*: block start time in seconds (recording-relative).
- **`/blocks/end_sec`** *(float32)*: block end time in seconds.
- **`/blocks/duration_sec`** *(float32)*: block duration in seconds (`end_sec − start_sec`).
- **`/blocks/frequency_hz`** *(float32)*: inferred block parameter/frequency when available (NaN when not applicable).
- **`/blocks/source_event_idx`** *(int32)*: source events-table row index that seeded the block.
- **`/blocks/support_event_count`** *(int32)*: number of supporting bridge/phase events linked to the block.
- **`/blocks/derived_from`** *(utf-8 string)*: source kind (`stage_blocking`, `duration_events`, `task_phase`).
- **`/blocks/end_reason`** *(utf-8 string)*: reason used to terminate the block.
- **`/blocks/membership_mode`** *(utf-8 string)*: membership policy when provided by the source inference path.
- **`/blocks/bridge_tail_sec`**, **`/blocks/bridge_tail_cap_sec`** *(float32)*: bridge-marker tail policy copied from stage-blocking when available.
- **`/blocks/is_inferred`** *(int8, 0/1)*: `1` for inferred boundaries, `0` for explicit-duration event blocks.

Group attrs:
- **`_schema_version`** = `"block_native_v1"`

---

### Group: `/block_windows`

Created alongside `/blocks/` when block-native windows were generated. One dataset per column, all arrays of length **N** (total windows across all blocks for the subject):

- **`/block_windows/block_id`** *(int32)*: parent block index (foreign key into `/blocks/block_id`).
- **`/block_windows/window_id_within_block`** *(int32)*: 0-based window index within its parent block.
- **`/block_windows/stage_code`** *(int32)*: inherited from parent block.
- **`/block_windows/block_start_sec`** *(float32)*: parent block start (seconds).
- **`/block_windows/block_end_sec`** *(float32)*: parent block end (seconds).
- **`/block_windows/block_duration_sec`** *(float32)*: parent block duration (seconds).
- **`/block_windows/window_start_sec`** *(float32)*: window start time (seconds, recording-relative).
- **`/block_windows/window_end_sec`** *(float32)*: window end time (seconds).
- **`/block_windows/window_center_sec`** *(float32)*: window centre time (seconds).
- **`/block_windows/relative_time_in_block_sec`** *(float32)*: seconds elapsed from block start to window centre.
- **`/block_windows/distance_to_block_end_sec`** *(float32)*: seconds remaining until block end from window centre.
- **`/block_windows/relative_pos_0_1`** *(float32)*: fractional position within block in [0, 1] (0 = block start, 1 = block end).
- **`/block_windows/source_window_index`** *(int32)*: nearest original MNPS window index on `/time` for exact downstream joins (or `-1` when no robust match).
- **`/block_windows/partition_label`** *(utf-8 string)*: non-empty when `window_profile.kind: "partitioned"` or `"post_offset"`, empty otherwise.
- **`/block_windows/is_post_offset`** *(int8, 0/1)*: 1 when window belongs to a post-offset bin profile.

Group attrs:
- **`_schema_version`** = `"block_native_v1"`

Important distinction:

- `/block_windows/*` is the canonical HDF5 block-window geometry contract
- richer Parquet/CSV block-native sidecars may additionally join aligned MNPS,
  anchor-state, task-label, QC, and selected raw feature columns such as
  `ecg_hrv_*` and `qc_ok_ecg_hrv`

---

### Group: `/extensions`

Created if extensions exist. The structure is **free-form** and mirrors nested dicts:

- **`/extensions/<extension_name>/...`**: subgroups/datasets for extension payloads (e.g. `e_kappa`, `rfm`, `o_koh`, `tig`).
- Conventional EEG comparator summaries are written under:
  - **`/extensions/conventional_eeg/...`**
  - families are currently grouped as `relative`, `ratio`, `peak`, `complexity`,
    `connectivity`, and `coma`

Rule:
- dict → subgroup
- scalar/array → dataset (gzip-compressed only above the writer's size threshold; see "Compression / dtypes" below)

---

### Sleep-EAP Phase 2 sidecars and extensions

Available only for the versioned RichSleep Phase 2 configuration. These are
data/QC products, not inferential results.

- **`phase_continuous_v1` sidecar**: one row per regular phase sample, with
  `timestamp_sec`, `phi_cardiac`, `phi_resp`, per-modality validity flags, and
  optional REM-theta phase. Its sampling rate is recorded in the sidecar
  metadata and must be honoured by downstream hazard models.
- **`non_event_risk_v1` sidecar**: seeded, N2- and time-of-night-stratified
  non-event timestamps. It is distinct from `matched_control`, which is a
  30-s geometry control product rather than a point-process risk set.
- **`event_phase_v3` sidecar**: the catalog-filtered v2 event phase columns
  plus raw-EEG spindle strength (`sigma_power`, `sigma_power_z_n2`), YASA
  provenance, and SO-spindle coupling fields. `so_partner_missing=1` requires
  coupling metrics to be NaN; zero is never used to represent a missing partner.
- **`event_phase_n3_so_v1` sidecar**: one N3-gated slow-oscillation row per
  detected trough/up-state pair. It samples the SO carrier and cardiac/
  respiratory phases at both reference points. Missing autonomic phase is
  represented by NaN plus point-specific validity and QC flags.
- **`event_phase_rem_theta_v1` sidecar**: one row per scored 30-s REM epoch,
  referenced at the epoch midpoint. It samples REM-restricted theta and
  autonomic phase at that midpoint, without requiring a theta-burst detector.
- **`/extensions/phase_continuous_v1`** and
  **`/extensions/non_event_risk_v1`**: optional H5 embeddings of the same
  sidecars. They are enabled only by the Phase 2 overlay and are not aligned
  to the 30-s MNPS time grid.

The N3-SO and REM-theta tables are Parquet-only in this version and join on
`subject` plus their explicit reference times. Their schemas deliberately
exclude spindle-strength (`sigma_*`, `frequency_hz`) and SO-spindle-pairing
fields, so a carrier result cannot be mistaken for spindle coupling.

The primary `/coords_9d` and `/mnps_3d` surfaces remain body-signal-free and
retain their existing 30-s resolution. Phase 2 therefore supplies one
integrated geometry outcome per event, not a within-spindle geometry trace.

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

- Datasets are **not** unconditionally gzip-compressed. The writer (`core/src/core/io/h5_writer.py`, `_create_dataset`) applies **gzip** (`compression_opts=4`) with chunking/shuffle only when the array is large enough to benefit: `chunks`/`shuffle` are enabled once `arr.size >= 10_000` elements, and gzip is enabled only once `arr.nbytes >= 256_000` bytes. Most per-timepoint `[T]` arrays (e.g. `int8` labels/QC flags, small `float32` series) are therefore written **uncompressed**; large 2D/3D arrays (feature matrices, Jacobian stacks, coordinate matrices) typically cross the threshold and are gzip-compressed. True scalar datasets are never compressed (HDF5 does not support chunking for scalars).
- `time` is `float64`; most other numeric arrays are `float32` for disk/IO.
- `labels/*` is `int8`; `events/*` is `int64` or `float64`; `nn/indices` is `int32`.
- `/features_raw/metadata/robust_z_valid` (and the equivalent field on `/features_robust_z/metadata/*`, `/features_projection_z/metadata/*`) is written as **`int32`**, not `int8`, in current exports.

