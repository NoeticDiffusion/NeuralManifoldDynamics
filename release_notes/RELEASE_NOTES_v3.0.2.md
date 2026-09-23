# NeuralManifoldDynamics v3.0.2 — fMRI measurement contract and NEMAR download

v3.0.2 is a patch on the v3.0.1 measurement contract. Canonical `mnps_3d = [m, d, e]`,
stratified `coords_9d`, and existing dynamical-family schema IDs are unchanged.
Common EEG profiles are unchanged. Common fMRI summarize now refuses an
underdetermined local Jacobian instead of writing a finite ridge matrix.

Package version: `mndm.__version__ = "3.0.2"`.

HDF5 `export_contract_version` for new fMRI files remains
`mndm.fmri_h5_contract.v1`. EEG/MEG remain `mndm.eeg_h5_contract.v1`.
Older fMRI files can still carry the historical EEG contract name; read
`modality` and the source class (`nifti_bold`) before treating that string
as an EEG recording.

## Highlights

**fMRI time, space, and missingness**

- TR order is BIDS sidecar `RepetitionTime`, then NIfTI `zooms[3]`, then
  `preprocess.fallback_tr`. JSON `null` is not a TR. A non-null invalid value,
  or JSON and zooms that disagree by more than 1 ms, is `NOT_TESTABLE`.
  Serialized `tr_source` is `bids_json`, `nifti_zooms`, or `config_fallback`.
- Same voxel shape is not the same atlas space. A mismatched affine or
  orientation at that shape is `NOT_TESTABLE` unless both
  `resample_atlas_to_bold` and `assume_same_space` are true. Resample uses
  nearest neighbour. Affine tolerance is `1e-3` mm.
- One session bandpass, in `fmri_continuous`. A bandpass listed under
  `preprocess.fmri` is recorded and not applied again.
- Nuisance status is one of `applied`, `skipped_missing_file`,
  `skipped_no_columns`, `truncated_length`, `failed_clean`, `disabled`.
  Enabled regression fail-closes unless status is `applied`. Length mismatch
  does not truncate BOLD. Confound NaNs are not filled with 0.
- Coverage and derivatives split on `t_start` gaps. Non-finite regional
  coordinates are not imputed. Regional `dt` is `fmri_step_sec` when that
  column exists, otherwise the median `t_start` step. `dt_source=default_1`
  is the standalone fallback, not a measured TR.
- `mnps.fs_out` is a configuration hint (`fs_out_role=configuration_hint`).
  Derivatives and `J_dot` use `dt_realized_sec`.

**fMRI Jacobian gate (common YAML)**

- `mnps.jacobian.require_determined_support: true` and
  `forbid_cross_gap: true`. A window is `not_testable` unless the unique
  neighborhood has at least `dim * (dim + 1)` samples in the same time
  segment. Missing time or file boundaries with that flag on are
  `gap_boundaries_unresolved`, not a silent cross-gap kNN fit.
- `mnps_9d.jacobian.enabled: false`. 3D stays enabled and capability class
  `limited`. An even `super_window` (common value 2) is realized as the next
  odd length; both requested and realized values are serialized.
- EEG `estimate_local_jacobians` calls that omit the new flags keep
  `min_samples = dim + 1` and the same `J_hat`.
- No HRF deconvolution, slice-timing, or spatial smoothing was added.
- `regional_mnps.enabled` stays false in common fMRI. The ds004796 overlay
  still enables regional Jacobians without `require_determined_support`.
  That overlay can still write an underdetermined regional ridge matrix.

**fMRI features and cache**

- ROI time series are cached as `mndm.roi_ts.v2`, keyed by atlas and BOLD
  identity, TR, nuisance inputs, and space flags. A second preprocess of the
  same file reloads that array and does not call `get_fdata` again.
  Production parcellation remains the boolean-mask label mean. A scatter-add
  path exists and matches it; it is not faster on the reference volume.
- Epoch AR(1)/AR(2), the dFC fallback, participation, and framewise-displacement
  epoch maxima were rewritten to match the previous numeric results on the
  tested synthetic cases. `cKDTree.query` uses `workers=1` under the feature
  thread pool.
- Stage-2 columns (sample entropy, permutation entropy, spectral entropy,
  smoothness, Hurst) stay opt-in. They are on in the ds007216 audit overlay
  and off in common fMRI. On one cached ds007216 run they added about 16 s
  to a 17.5 s feature pass. Columns were not dropped.
- Rejected aliases remain: `fmri_entropy_global` and `fmri_region_var_mean`
  copy `fmri_variance_global`; `fmri_lf_power` copies `fmri_signal_power`.
  Do not map weights onto those names. `fmri_spectral.py`, `fmri_dynamic.py`,
  and `fmri_phase.py` are unused by the feature path.

**NEMAR download**

- New package `nemar_ingest`. It downloads a public NEMAR BIDS release from
  the versioned anonymous data plane and `manifest.json`.
- Default backend is `manifest` (no Node CLI). `backend: auto` falls back to
  the manifest only when the `nemar` executable is absent. A CLI process
  failure is not swallowed.
- Annexed files are checked with SHA-256. Git-tree files use
  `checksum_algorithm: git` (SHA-1 of `blob <nbytes>\0` plus the bytes).
- A finished run writes `acquisition_receipt.json` with the pinned release,
  manifest URL, manifest SHA-256, and selected entries. With
  `continue_on_error` (the default), a per-file failure is recorded and the
  loop continues; the process still exits non-zero. Already verified files
  are skipped. A receipt from another release is rejected.
- `nemar_ingest/config/on005385.yaml` pins `on005385` `v1.0.0` and writes
  under `K:/ExternalReceivedDatasets/nemar`. The manifest is about 50 GB.
  Validation in this release is synthetic (`nemar_ingest/tests`, 16 passed).
  A live transfer was not part of the recorded test run.

## Validation and claim boundaries

| Surface | Evidence in v3.0.2 | Claim ceiling |
|---|---|---|
| MNPS `[m,d,e]` / 9D axes | Unchanged | Chart coordinates |
| fMRI TR, atlas, filter, nuisance, gaps | Synthetic NIfTI and unit tests | Provenance and fail-closed status |
| fMRI Jacobian gate | Synthetic T=8 dim=9 and underdetermined 3D; existing Jacobian tests with the flag off | `not_testable` vs finite ridge fit. 3D that passes stays `limited` |
| ROI-TS cache | One ds007216 run: second preprocess cache hit, bitwise ROI-TS match | Not a cohort hash |
| Stage-2 cost | One cached ds007216 run, two repeats | Cost, not validity of short-window SampEn or Hurst |
| NEMAR ingest | Synthetic manifest tests | Download and receipt. Not an MNPS measurement |
| ds004796 regional Jacobian | Overlay still omits the new flags | Not covered by the common-fMRI gate |

**Safe claims**

- New fMRI files record TR source, nuisance status, filter stage, and atlas-space
  status instead of inventing a TR, a second bandpass, or a matched space.
- Common-fMRI local Jacobians with fewer than `dim*(dim+1)` unique samples in
  one time segment are `not_testable`.
- `nemar_ingest` can select, verify, and receipt a pinned NEMAR release without
  changing MNPS.

**Not established in v3.0.2**

- A biophysical or HRF-deconvolved Jacobian.
- That a finite 3D fMRI `J_hat` which passes the sample-count gate is
  scientifically testable beyond class `limited`.
- Short-window sample entropy or Hurst as valid fMRI measurements. Finite
  Stage-2 columns are not validity.
- Cross-release bitwise equality of fMRI Jacobians. Common-fMRI summarize
  after this patch drops windows that earlier releases kept.
- A live `on005385` download in the recorded test run.
- The ds004796 regional Jacobian path.

## Upgrading

- EEG summarize that does not set `require_determined_support` or
  `forbid_cross_gap` keeps the previous Jacobian neighborhood.
- Datasets that import `config_ingest_common_fmri.yaml` (including ds007216
  and ds004796 for the global MNPS path) will refuse underdetermined 3D
  Jacobians and will not write a 9D `J_hat`. Compare those files by
  `mndm_version`, not by assuming `mndm.fmri_h5_contract.v1` means the
  pre-3.0.2 Jacobian.
- `fs_out: 0.5` on common fMRI is still a plotting hint. Read
  `dt_realized_sec`.
- Map projection weights to `fmri_variance_global` and `fmri_signal_power`,
  not to `fmri_entropy_global` or `fmri_lf_power`.
- NEMAR download is a separate package. It does not run `mndm` features or
  summarize.

```text
python -m nemar_ingest.cli --config nemar_ingest/config/on005385.yaml --dry-run
```

See [`mndm/CONFIG_GUIDE.md`](../mndm/CONFIG_GUIDE.md),
[`mndm/Output_variables_guide.md`](../mndm/Output_variables_guide.md),
[`nemar_ingest/README.md`](../nemar_ingest/README.md), and
[`RELEASE_NOTES_v3.0.1.md`](RELEASE_NOTES_v3.0.1.md).
