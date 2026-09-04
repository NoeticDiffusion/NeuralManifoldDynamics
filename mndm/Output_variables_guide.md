### HDF5 output schema (MNDM 2.6 release line)

This file documents **all nodes (datasets + groups) and relevant HDF5 attributes** written by the MNDM summarization pipeline into each `*.h5`.

Release-version note:
- This document describes the **NeuralManifoldDynamics 2.6** measurement surface.
- Some embedded sub-schema identifiers intentionally still carry `v2.1` names
  (for example the explicit anchored-coordinate layer schema) because those
  subcontracts were introduced in the 2.1 release line and remain valid in 2.6.

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
- **`schema_version`** *(str)*: often `mnps_tensor_spec_v2_1` when the explicit anchored-coordinate sub-contract is present. In the 2.3 release line this identifier is still retained for backward-compatible coordinate-layer schema naming.
- **`mndm_version`** *(str)*: software/measurement-contract release line recorded in the export metadata; current documentation tracks `2.6`.
- **`primary_coordinate_layer`** *(str)*: usually `coords_3d_cohort_anchored` when a cohort/external anchor is configured, otherwise `coords_3d_subject_anchored`.
- **`primary_coordinate_contract`** *(str)*: `cohort_anchored` or `subject_anchored`.
- **`anchor_id`**, **`anchor_hash`** *(str|None)*: identity and hash of the feature-anchor artifact used for cohort-anchored coordinates.
- **`geometry_invalidity_policy`** *(str)*: always-on hard-invalidity contract version for canonical MNPS/MNJ exports, currently `standard_invalidity_v1`.
- **`geometry_contract_status`** *(str)*: status of the always-on geometry contract for this export, typically `ok` or `adjusted`.

Derived convenience attrs (best-effort; may be absent):
- **`meta_<field>`** *(str/int/float)*: flattens scalar fields from `participant_meta` (participants.tsv) into top-level attrs.
- **`group`** *(str)*: may be derived/normalized from `participant_meta` if not already set.
- **`condition`** *(str)*: may be derived from session-/meta-fields if not already set.

---

### Manifest (JSON in `h5.attrs["manifest"]`): Tier 0/1/2 measurement blocks

These are **analysis-agnostic descriptive blocks** embedded in the manifest JSON (and also written to `summary.json` / `qc_summary.json`).

- **`geometry_contract`** *(object)*: always-on mathematical invalidity contract for canonical geometry export.
  - **`policy_version`**: currently `standard_invalidity_v1`.
  - **`status`**: `ok` when no mathematically invalid geometry had to be removed, `adjusted` when invalid epochs or Jacobian windows were dropped or retained surfaces are degraded.
  - **`shared_time_grid`**: counts for `epochs_before_policy`, `epochs_retained`, `epochs_dropped`, `drop_fraction`, and `drop_reason_counts`.
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
- **`/labels/<name>`** *(shape `[T]`)*: optional aligned label series. Depending on the source, this may be integer-coded, binary, or UTF-8 string labels such as task-state labels.

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
- **`/jacobian/affine_reference`** *(float32, shape `[W,D]`, optional)*: local fit-neighborhood reference used by the affine derivative model.
- **`/jacobian/affine_intercept`** *(float32, shape `[W,D]`, optional)*: local derivative intercept paired with `affine_reference`; it does not alter `J_hat` semantics.
- **`/jacobian/diagnostics/hard_invalid_centers`** *(int32, optional)*: center indices of Jacobian windows removed by the standard invalidity policy.
- **`/jacobian/diagnostics/hard_invalid_windows`** *(scalar attr, optional)*: number of canonical windows removed after Jacobian estimation.
- **`/jacobian/diagnostics/hard_invalid_condition_number_threshold`** *(scalar attr, optional)*: hard condition-number threshold used by the standard policy.
- **`/jacobian/derived_metrics/v1/`** *(optional)*: `mndm.jacobian_metrics.v1` semantic local-dynamics metrics. Its `series/` datasets are per valid-Jacobian-window fields (`spectral_abscissa`, `numerical_abscissa`, `symmetric_rate_min`, `reactivity_gap`, `stable_reactive_flag`, magnitude/deformation/rotation diagnostics); `summary/` carries support counts and `stable_reactive_fraction`; `provenance/` carries zero tolerances and metric semantics. Sibling certificate fields: `computation_status` (`computed` if any finite metric windows, else `insufficient_support`), `measurement_validity` (`not_assessed` when computed, else `not_applicable`), `claim_status` (`no_biological_claim` on new writes). These are mathematical provenance on `J_hat`, not S3-licensed empirical NDT \(\alpha/\omega\).
  - Full `series/` fields are `spectral_abscissa`, `numerical_abscissa`,
    `symmetric_rate_min`, `symmetric_rate_max`, `reactivity_gap`,
    `stable_reactive_flag`, `dynamical_regime`, `spectral_radius`,
    `frobenius_norm`, `trace`, `rotation_norm`, and `henrici_departure`.
  - Theory \(\Omega\) appears only as the scalar `rotation_norm`, not as a matrix. Theory \(G_{\mathrm{peak}}\) is **not** in this group; see FTR `g_peak_over_horizons`.

Interpretation note:

- Jacobian windows are not clamped when mathematically unusable
- instead, invalid windows are removed from the canonical export and recorded in the diagnostics / `geometry_contract`

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
also include `eeg_conventional_connectivity_*` columns. When the coma pack is
enabled, the surface may also include `eeg_conventional_coma_*` columns.

For `ds003645` MEG shadow-mapping runs, `features_raw` may additionally expose
paired diagnostic and combined MEG columns:

### MEG measurement-safeguard surfaces

`/features_projection_z` is the transform-aware export surface: each row is
aligned exactly to `/time`, `/mnps_3d`, and `/features_raw`, while values have
the configured feature pipeline applied (for example `log10 → robust_z → clip`
for MEG power). `/row_source` contains additive row-aligned source lineage
(`raw_file`, source-format and modality flags); `/epoch_id` is an additive
per-row identity when available.

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
- new guarded exports add `robust_z_valid` (`int8`), `robust_z_invalid_reason`
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
- **`/provenance/geometry_contract/*`**: always-on mathematical invalidity contract for canonical geometry exports
- **`/provenance/anchoring/*`**: explicit coordinate contracts/layers available in this H5 plus primary contract/layer and optional anchor identity
- **`/provenance/normalization/*`**: concise normalization status/method/scope and sidecar references
- **`/provenance/event_stage_mapping/*`**: event/stage mapping versioning, source column/path, and codebook hash
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
- **`/qc/windows/qc_ok_eeg`**, **`/qc/windows/qc_ok_ecg`**, **`/qc/windows/qc_ok_eog`** *(int8, optional)*
- **`/qc/windows/coverage_ok`** *(int8, shape `[T]`, optional)*
- **`/qc/windows/mnps_3d_valid`** *(int8, shape `[T]`, optional)*: per-window finite validity for retained 3D MNPS rows.
- **`/qc/windows/coords_9d_valid`** *(int8, shape `[T]`, optional)*: per-window finite validity for retained stratified coordinates.
- **`/qc/windows/geometry_valid`** *(int8, shape `[T]`, optional)*: joint retained-window validity across available geometry surfaces.
- **`/qc/windows/stage_transition_flag`** *(int8, shape `[T]`, optional)*
- Group attrs:
  - **`_schema_version`** = `mndm.qc.windows.v1`

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
- **`/jacobian_9D/J_dot`** *(float32, shape `[W2-1,K,K]`)*: temporal difference.
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
- **`/stochastic_reachability/v1/{primary,stratified_9d}/`**: opt-in (`local_dynamics.stochastic_reachability.enabled`, default false) discrete reachability \(W\leftarrow\Phi W\Phi^\top+Q\) from Gate E Q and \(\Phi=\mathrm{expm}(J_{\mathrm{crossfit}}\Delta t)\). Not `/dynamical_families/spread`. Certificate when computed: `not_assessed` / `no_biological_claim`. Grain: `recording_horizon`. Analysis-repo I-CARE / coma reachability products (for example `tube_d_eff_median`) are **not** this schema.

### Validity certificate (v3 R2)

Every family and local-dynamics v1 group above may carry three sibling datasets:

- `computation_status`
- `measurement_validity`
- `claim_status`

These are additive keys on existing schema IDs (`mndm.jacobian_metrics.v1`,
`mndm.finite_time_response.v1`, `mndm.transition_residuals.v1`,
`mndm.diffusion_geometry.v1`, `mndm.committor.v1`,
`mndm.finite_amplitude_resilience.v1`, residual-covariance and reachability
v1). Schema IDs are unchanged.

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

Typical natives: Jacobian metrics and diffusion/destination `window`; FTR and
reachability `recording_horizon`; transition residuals `transition`; Q-proxy
`recording` (`repeated_measure=false`); FAR `event`. Grain is written even
when `computation_status` is not `computed`.

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

Schema IDs are unchanged. Family YAML `spread` remains `gate_closed` and is
not written here. Opt-in `mndm.stochastic_reachability.v1` lives at
`/stochastic_reachability/v1`, not under `/dynamical_families`.

- **`/dynamical_families/diffusion/v1/`**: `mndm.diffusion_geometry.v1`. Local increment-covariance diffusion geometry. `contract_status=standard` is the schema class, not an empirical license. Ingest uses C1 defaults (`drift=None`, `residualize_increments=False`): `a_hat` / `D_total` / `d_diff` / `c_diff` may be `computed`, but `A_bD` and `R_b_over_a` are NaN with `summary.A_bD_computation_status=not_testable`, `summary.R_b_over_a_computation_status=not_testable`, and `drift_alignment_failure_reason=independent_drift_not_supplied`. `summary.a_semantics=raw_increment_covariance`. Do not read those NaNs as zero alignment. Jacobian residuals, MNPS \(\dot x\), and Jacobian intercepts are not diffusion \(a\) or SDE drift. Library C1 may consume an externally qualified chart \(b\) without changing `a_hat` (Gate C1-A). Ingest does not estimate \(b\); nmd-analysis owns identification (SL-005). C2 residualization is not authorized. Legacy read path: `/orthogonal_dynamics/diffusion_geometry/v1`.
- **`/dynamical_families/destination/v1/`**: `mndm.committor.v1`. Production adapter is 1-D O2b (`local_law_dense_grid_o2b`) with explicit first-hit A/B + reaction coordinate. O2b does not serialize \(V_{1/2}\) or \(\lvert\nabla q\rvert\); the first-hit estimator serializes `q_A_to_B` only. Stage labels are not committor truth. Legacy read path: `/orthogonal_dynamics/committor/v1`.
- **`/dynamical_families/resilience/v1/`**: `mndm.finite_amplitude_resilience.v1`. Observed perturbation-outcome FAR summary. Legacy read path: `/orthogonal_dynamics/finite_amplitude_resilience/v1`.

These groups do not alter `/mnps_3d`, `/coords_9d`, or `/jacobian`. Common EEG/fMRI/ephys profiles do not enable them. Each family v1 group carries the Round-2 validity certificate (`computation_status`, `measurement_validity`, `claim_status`) and the Round-3 nested `grain/` object as siblings; `provenance.validation_level` remains the method-validation tag.

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
- scalar/array → dataset (gzip-compressed unless scalar)

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

- All non-scalar datasets are written with **gzip compression** (`compression_opts=4`).
- `time` is `float64`; most other numeric arrays are `float32` for disk/IO.
- `labels/*` is `int8`; `events/*` is `int64` or `float64`; `nn/indices` is `int32`.

