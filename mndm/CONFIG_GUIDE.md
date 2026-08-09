# NeuralManifoldDynamics — YAML Configuration Guide

A practical reference for neuroscientists adding a new EEG or fMRI dataset.

---

## Contents

1. [How configs work](#1-how-configs-work)
2. [Minimal working config (5 lines)](#2-minimal-working-config)
3. [Section-by-section reference](#3-section-by-section-reference)
4. [Common recipes](#4-common-recipes)
5. [Output anatomy](#5-output-anatomy)
6. [Troubleshooting checklist](#6-troubleshooting-checklist)

---

## 1. How configs work

Every dataset needs **one YAML overlay file** in `mndm/config/`.  
The overlay imports a shared common base and only overrides dataset-specific details.

```
config_ingest_common_eeg.yaml   ← canonical EEG defaults (don't edit this)
        ▲
        │ imports:
config_ingest_my_dataset.yaml   ← your file (only what differs)
```

The common base supplies all MNPS/projection/robustness defaults.  
Your overlay only needs: dataset ID, paths, and any deviations from those defaults.

```powershell
# Bootstrap from the generic template
Copy-Item mndm/config/config_template.yaml mndm/config/config_ingest_MY_DATASET.yaml
# Then edit MY_DATASET

# For a non-BIDS BioSemi BDF archive, use the dedicated adapter template.
Copy-Item mndm/config/bdf_config_ingest_template.yaml mndm/config/config_ingest_MY_BDF_DATASET.yaml
```

---

## 2. Minimal working config

This is the **smallest possible overlay** to process a BIDS EEG dataset.

```yaml
version: 2.0
modality: eeg

imports:
  - "./config_ingest_common_eeg.yaml"

datasets:
  - my_dataset                     # ← the OpenNeuro/BIDS dataset ID

source:
  name: "My Lab Study 2024"
  platform: "OpenNeuro"
  format: "BIDS"
  dataset_id: "my_dataset"
  doi: null                        # add if known

paths:
  received_dir: "E:/Science_Datasets/openneuro/received"
  processed_dir: "E:/Science_Datasets/openneuro/processed"
  dataset_received_dirs:
    my_dataset: "E:/Science_Datasets/openneuro/received/my_dataset"

epoching:
  length_s: 8.0
  step_s: 4.0
```

Run it with:

```powershell
python -m mndm.cli all --dataset my_dataset `
  --config mndm/config/config_ingest_my_dataset.yaml `
  --n-jobs 8
```

Outputs land in:
```
<processed_dir>/my_dataset/neuralmanifolddynamics_my_dataset_<timestamp>/
```

---

## 3. Section-by-section reference

### `version` / `modality`

```yaml
version: 2.0          # always 2.0
modality: eeg         # eeg | fmri | nwb | nwb_rodent
```

---

### `imports`

```yaml
imports:
  - "./config_ingest_common_eeg.yaml"   # EEG (OpenNeuro / BIDS)
  # - "./config_ingest_common_nwb.yaml"  # NWB ecephys via DANDI
  # - "./config_ingest_common_ephys.yaml" # NWB Units -> population-rate MNPS
```

Imports are merged in order. Your overlay keys win over the imported ones.

---

### NWB extracellular ephys: Units and LFP

Use `config_ingest_common_ephys.yaml` when the NWB `Units` table contains
spike-sorted units. It bins each unit's `spike_times` into a population-rate
matrix before calculating `ephys_*` window features. The rate path is selected
explicitly:

```yaml
preprocess:
  nwb:
    units:
      enabled: true
      rate_bin_sec: 0.05
      smoothing_sigma_sec: 0.05
      quality_policy: "all"  # all | good | acceptable where a recognized quality field exists
```

For continuous LFP/ElectricalSeries, retain or import
`config_ingest_common_nwb.yaml` and set `units.enabled: false`. The existing
NWB loader then selects an `ElectricalSeries` using `series_path` or
`prefer_series_keywords` and processes it through the EEG feature path:

```yaml
preprocess:
  nwb:
    units: {enabled: false}
    prefer_series_keywords: ["lfp", "electrical"]
    channel_type: "seeg"
```

The two paths are intentionally distinct: binned spikes produce population
features (`ephys_*`), whereas LFP produces spectral EEG-style features
(`eeg_*`). Do not enable Units for an LFP-only file, and do not assume every
spike-sorted NWB file contains a continuous LFP series.

For a dual-modality file, do not rely on generic keywords when surface EEG and
multiple probe LFP streams coexist. Probe the file first, then set the exact
path in a dedicated overlay. The validated DANDI 000458 LFP smoke uses:

```yaml
preprocess:
  crop: {start_sec: 0, stop_sec: 59}
  nwb:
    datasets:
      dandi_000458:
        units: {enabled: false}
        series_path: "acquisition/LFPprobeB/ElectricalSeriesprobeB"
        prefer_series_keywords: ["lfp"]
        channel_type: "seeg"
        max_channels: 32
```

The crop and channel limit are smoke-test memory bounds, not a recommended
scientific analysis subset. Full-probe analyses require an explicit channel
selection and a documented memory budget.

### Full-session Neuropixels LFP selection

For full-session probe LFP, do not use `max_channels`: it selects leading
channels and is not a scientific sampling rule. First create a versioned
`lfp_channel_selection.json` by streaming one explicitly named
`ElectricalSeries` through `neuropixel_ingest.run_lfp_channel_qc`. The artifact
records every series-channel-to-electrode mapping, validity metadata, QC metric,
rejection reason, and selected contact.

The full-session probe overlays for DANDI 000458 reference the artifact through
`preprocess.nwb.channel_selection.artifact_path`. This loads only selected
non-leading indices and preserves selection provenance in the per-file QC JSON.
The policy is depth × horizontal balanced: two QC-ranked contacts per horizontal
column, then an explicit within-depth fallback. Probe B and F must remain
separate through feature extraction and MNPS. Their comparison is restricted to
the depth bins shared by both probes.

The QC reader uses single-probe, 60-second chunks. Feature extraction still
loads selected channels into the existing EEG-style path; retain single-worker
execution and profile memory before changing selection size or sampling rate.
The acquisition's native reference is preserved. Within-probe median
rereference is a diagnostic sensitivity run, not a replacement for it.

### State and stimulation-aware DANDI 000458 LFP

For `sub-551399`, use the continuous blocks derived from
`/intervals/trials.behavioral_epoch` (`awake`, `isoflurane`, `recovery`).
Do not use the generic `nwb_intervals` state order: its pharmacological
`epochs.tags` overlap point-like trial stimulation events and do not provide a
clean primary state factor.

`neuropixel_ingest.lfp_state.write_annotated_lfp_features` adds the following
to the selected-channel feature table: `lfp_behavioral_state`,
`lfp_stim_adjacent`, `lfp_stim_contains_onset`,
`lfp_nearest_stim_distance_sec`, `lfp_nearest_estim_current`, and
`lfp_running_speed_mean`. The primary filter is
`lfp_interstim_primary`: a state-assigned epoch whose midpoint is more than
one second from a stimulation onset. This preserves the 4-second feature
window while avoiding a falsely point-free interpretation of the whole epoch.

Use the dedicated `*_awake.yaml`, `*_isoflurane.yaml`, and `*_recovery.yaml`
probe overlays only with their matching annotated `features.parquet` inputs.
They summarize B and F separately; cross-probe results remain descriptive and
restricted to shared depth strata.

### Focused LFP contact/reference sensitivity

The primary 000458 LFP result uses eight contacts per depth and the native
acquisition reference. The focused sensitivity matrix adds:

- four contacts per depth with the native reference;
- eight contacts per depth with `reref: median_within_probe`.

`median_within_probe` subtracts the samplewise median across the artifact-
selected contacts after loading and before feature extraction. It is not the
median depth-ensemble aggregation used later by the feature extractor. The
preprocess QC JSON records the rereference mode, selected channel count, and
SHA-256 of the selection artifact. Do not substitute mean `average` reference
for this diagnostic.

The unrun four-contact median-reference cell remains outside the focused
design. Compare feature-level stability by state first; independent MNPS
normalization can rotate or reflect coordinates, so raw cross-run m/d/e
coordinate correlations are descriptive only.

---

### `datasets`

```yaml
datasets:
  - ds004511        # list of dataset IDs to process
  - ds003490        # can be more than one (runs sequentially)
```

You can also pass `--dataset ds004511` on the CLI to override this.

---

### `source`

Human-readable provenance. Copied verbatim into every `run_manifest.json`.

```yaml
source:
  name: "Study of Sleep Cognition"
  platform: "OpenNeuro"
  format: "BIDS"
  dataset_id: "ds003490"
  dataset_name: "Full title from OpenNeuro"
  doi: "10.18112/openneuro.ds003490.v1.1.0"
  url: "https://openneuro.org/datasets/ds003490"
  acknowledgement: "DOI citation text here"
```

---

### `paths`

```yaml
paths:
  received_dir: "E:/Science_Datasets/openneuro/received"
  processed_dir: "E:/Science_Datasets/openneuro/processed"

  # Fallback roots if the primary received_dir is unavailable for some datasets
  received_dir_fallbacks:
    - "G:/Science_Datasets_longtime_storage"

  # Per-dataset explicit roots (preferred over received_dir for most cases)
  dataset_received_dirs:
    ds003490: "E:/Science_Datasets/openneuro/received/ds003490"
```

The pipeline resolves: `dataset_received_dirs[id]` → `received_dir/id` → fallbacks.

---

### `preprocess`

Controls EEG signal conditioning before feature extraction.

```yaml
preprocess:
  sfreq: 250                    # target sample rate after resampling
  sfreq_candidates: [250, 256]  # integer-ratio downsample targets tried in order
  notch_hz: 50                  # power-line notch (use 60 for US datasets)
  eeg_bandpass: [1, 45]         # bandpass filter (applied before epoching)
  reref: "average"              # "average" (CAR) | "none" | "REST"

  artifacts:
    method: "none"              # "none" | "ica" | "eog_reg" | "autoreject"
```

**Dataset-specific overrides** (when one dataset needs different settings):

```yaml
preprocess:
  datasets:
    ds003490:
      notch_hz: 60              # US recording
      sfreq: 256
```

---

#### `preprocess.bdf_adapter` — non-BIDS BioSemi BDF archives

Use the opt-in BDF adapter when raw files do not follow BIDS naming or when a
BioSemi archive contains auxiliary channels or repeated/flat channel banks.
Start from `mndm/config/bdf_config_ingest_template.yaml`; it keeps all
source-specific policy in YAML while the adapter code remains generic.

```yaml
preprocess:
  bdf_adapter:
    enabled: true
    mapping_path: "metadata/subject_session_map.csv"
    mapping_key_column: "old_id"
    mapping_subject_session_column: "subject_id"
    mapping_subject_column: "subject"
    mapping_session_column: "session"
    subject_pad: 2
    session_pad: 2
    # Fallback when a raw stem has no mapping row:
    filename_regex: "^TD(?P<subject>\\d+)v(?P<session>\\d+)\\.bdf$"
    task: "continuous_eeg"

    # Select source channels before resampling/filtering. This prevents Status,
    # EXG, and inactive duplicate banks from entering scalp-EEG features.
    channel_selection: {mode: "first_n", n_channels: 32}
    canonical_channel_names: ["Fp1", "AF3", "...", "Cz"]  # exactly 32 names
    montage: "biosemi32"

    # Optional recording-specific QC policy.
    bad_channels_path: "metadata/bad_channels.csv"
    bad_channels_key_column: "subject_id"
    bad_channels_column: "bad_channels"
    interpolate_bads: true
    # Optional MNE annotation CSVs, resolved from the mapping-table identity.
    # BAD_ intervals become NaNs in the source time grid; every overlapping
    # EEG epoch is rejected before features are computed.
    bad_segments_dir: "metadata/bad_segments"
    bad_segments_key_column: "subject_id"
    bad_segments_filename_template: "{subject_id}_annotations.csv"
    mask_bad_segments: true
    recording_crops:
      RECORDING_STEM: {start_sec: 0, stop_sec: 300}
```

`bad_segments_*` is appropriate only for source-reviewed artifact annotations
(such as MNE `BAD_` intervals). It is deliberately not an event-label importer:
do not encode inferred behavioral states such as rest or reaching through this
mechanism. Missing annotation files remain auditable in each recording's
preprocessing metadata and leave that recording unmasked.

The mapping CSV is the source of truth for BDF `subject` and `session`
entities during both indexing and summary grouping. A mapping row must match a
raw stem exactly, without the `.bdf` suffix. The adapter supports
`channel_selection.mode: "first_n"` and `"names"`.

Do not treat the BDF `Status` channel as experimental events unless the source
protocol and an explicit event audit verify its codes. The adapter intentionally
does not infer tasks or conditions from Status.

For subject- and session-level scientific metadata, configure privacy-safe
tables under `metadata_extraction.datasets.<id>.participants.path` and
`.sessions.path`. The session table must contain `participant_id` and
`session_id`; it is joined to the corresponding summary/HDF5 output.

---

#### `preprocess.artifacts` — ICA artifact removal

ICA removes eye-movement and cardiac components before feature extraction.
Activate by setting `method: "ica"` in `preprocess.artifacts`.

```yaml
preprocess:
  artifacts:
    method: "ica"

    # Number of ICA components to fit. Integer = exact count; 0.999 = explain
    # 99.9% of variance (recommended after Maxwell filtering / SSS on MEG).
    ica_n_components: 20

    # ICA algorithm. "fastica" is the standard choice for EEG.
    # "infomax" and "picard" are also supported via MNE.
    ica_method: "fastica"

    ica_random_state: 42
    ica_fit_highpass_hz: 1.0    # high-pass before ICA fit (does not affect exported data)
    ica_max_components_to_remove: 5   # safety cap

    # EOG component detection thresholds (z-score)
    ica_eog_threshold: 3.0
    ica_ecg_threshold: 3.0

    # --- EOG proxy channels ---
    # For datasets WITHOUT dedicated EOG channels (e.g. BrainVision recordings
    # without VEOG/HEOG), use frontopolar channels as an eye-movement proxy.
    # The pipeline temporarily retypes these as EOG for ICA component detection,
    # then restores original types.
    # Leave empty ([]) when dedicated EOG channels are present.
    eog_proxy_channels: ["Fp1", "Fp2"]

    # --- ECG component detection ---
    # Null = auto-detect the first channel with MNE type "ecg".
    # Set explicitly if the channel has an unusual name.
    ecg_channel: null
```

**Dedicated EOG channels** (e.g. Neuromag FIF / MEG recordings):

## MEG measurement safeguards

`meg_ensembles` optionally freezes channel membership for named helmet sectors.
These names describe sensor measurement geometry, not cortical regions.

```yaml
meg_ensembles:
  sensor_family: mag
  min_channels: 12
  groups:
    left_anterior: ["MEG0111", "MEG0121"]
```

Enable `sensor_topography_qc` only for an explicit report-only QC run. It is
disabled by default; cross-modal testing additionally requires a frozen
`sensor_topography_contract` and null-gated paired inputs. No setting may use
these sectors to alter `/mnps_3d` or harmonize MEG signs to EEG.

```yaml
preprocess:
  artifacts:
    method: "ica"
    ica_n_components: 0.999        # rank-based for post-SSS MEG
    eog_proxy_channels: []         # leave empty; dedicated EOG channels are used
    ecg_channel: "ECG063"          # Neuromag standard
    ica_eog_threshold: 3.0
    ica_ecg_threshold: 3.0
```

**Verifying ICA ran**: look for this line in the pipeline log:

```
INFO: Applied ICA (n_components=20); excluded 2 components (eog_proxy=['Fp1', 'Fp2'], ecg=['ECG'])
```

**Important**: after changing `preprocess.artifacts.method`, always re-run with
`--force-features` to discard cached intermediate JSON files:

```powershell
python -m mndm.cli features --dataset my_dataset --force-features
python -m mndm.cli summarize --dataset my_dataset
```

---

#### `preprocess.bad_channels` — bad-channel detection

Automatic detection of flat, high-variance, or decorrelated EEG channels
before average re-referencing. Detected channels are dropped (or interpolated
if `interpolate: true`) prior to CAR.

The detection heuristics read from `robustness.bad_channels`:

```yaml
robustness:
  bad_channels:
    var_low_factor: 1.0e-4    # flat channel threshold (fraction of median variance)
    var_high_factor: 25.0     # high-variance threshold (× median variance)
    corr_thresh: 0.2          # minimum correlation with global mean
    max_bad_fraction: 0.3     # abort if more than this fraction would be dropped
    min_good_channels: 8      # minimum channels that must remain
```

This heuristic bad-channel check runs automatically for all EEG datasets.
At present, `preprocess.bad_channels.method: ransac` is not implemented in
the MNDM preprocessing path; use the `robustness.bad_channels` thresholds
above, or provide source-reviewed channel lists through
`preprocess.bdf_adapter.bad_channels_path` for non-BIDS BDF data.

**Note**: The number of dropped bad channels per subject is logged in
`qc_summary.json` under `artifacts.bad_eeg_channels` and
`artifacts.n_bad_eeg_channels`.

---

### `epoching`

Sliding window parameters and optional event-based stage labeling.

```yaml
epoching:
  length_s: 8.0      # window length in seconds
  step_s: 4.0        # step / stride (overlap = (length-step)/length = 50%)
```

**Stage labels from BIDS events** (for task or clinical datasets):

```yaml
epoching:
  datasets:
    my_dataset:
      sampling:
        stage_columns: ["value"]           # which events.tsv column carries the label
        prefer_events_stage_in_summary: true
```

**Stage-blocking** (when events are sparse markers, not continuous labels — e.g. photic stimulation):

```yaml
epoching:
  datasets:
    my_dataset:
      sampling:
        stage_blocking:
          enabled: true
          stage_event_regex: "(?i)^PHOTO\\s*(\\d+)\\s*Hz$"  # block-start events
          bridge_marker_labels: ["Photo/HV mark"]            # in-block densifiers
          min_block_sec: 2.0
          max_block_sec: 20.0
          window_membership:
            mode: "midpoint_in_interval"    # or "fully_contained" for stricter epochs
```

---

### `features`

Feature extraction parameters. Defaults from the common base are usually fine.

```yaml
features:
  eeg_psd: {method: "multitaper", bandwidth: null}    # PSD method
  eeg_bands:
    delta: [1,  4]
    theta: [4,  8]
    alpha: [8,  12]    # note: 8–12 Hz (see CONFIG_GUIDE §3 for rationale)
    beta:  [13, 30]
    gamma: [30, 45]
  ratios:
    alpha_theta: ["alpha", "theta"]
    beta_alpha:  ["beta",  "alpha"]
  permutation_entropy:
    order: 5
    delay: 1
    normalize: true
```

**Enable ECG / HRV features** (for multimodal datasets with an ECG channel):

```yaml
features:
  ecg:
    hrv:
      enabled: true
      superwindow_s: 60.0      # HRV superwindow (longer than EEG epoch)
      min_nn_intervals: 20
```

**ECG polarity auto-detection** (enabled by default, no config needed):

MNDM automatically detects whether ECG QRS deflections are predominantly
positive or negative (by comparing P99 vs P01 of the bandpass-filtered signal)
and inverts the signal before peak detection when needed.  Every epoch record
now includes:

| Column | Type | Description |
|---|---|---|
| `ecg_polarity_inverted` | bool | `True` when the signal was inverted before peak detection |
| `ecg_peak_detector` | str | Name of detector used (`neurokit2`, `scipy_polarity`, `scipy_abs`) |

To override the automatic polarity correction, set `peak_detector: scipy_abs`
(the legacy absolute-value detector which ignores polarity).

**HRV superwindow contamination reporting** (automatic when HRV is enabled):

When `features.ecg.hrv.enabled: true` and the recording has a companion BIDS
`*_events.tsv`, MNDM automatically computes per-label overlap fractions for
every HRV superwindow.  New columns:

| Column | Description |
|---|---|
| `ecg_hrv_dominant_stage_label` | `trial_type` label covering the largest fraction of the HRV window |
| `ecg_hrv_dominant_stage_frac` | Fraction of the HRV window covered by the dominant label (0–1) |
| `ecg_hrv_n_stage_labels` | Number of distinct labels present in the HRV window (>1 % overlap) |
| `ecg_hrv_contains_excluded_label` | `True` when any excluded label (see below) is present |

To gate HRV windows that overlap retrieval / speech / motor artefacts:

```yaml
features:
  ecg:
    hrv:
      enabled: true
      superwindow_s: 60.0
      exclude_labels:
        - Digits_Retrieval   # any trial_type value(s) to flag
        - Speech
```

Windows with `ecg_hrv_contains_excluded_label: true` are still exported; the
flag lets downstream scripts exclude them without re-running the pipeline.

**Enable HRV nonlinear complexity** (requires `antropy` and `nolds`):

```yaml
features:
  ecg:
    hrv:
      enabled: true
      superwindow_s: 60.0
      min_nn_intervals: 20
      complexity:
        enabled: true            # opt-in; adds ecg_hrv_sampen + ecg_hrv_dfa_alpha1
        sampen_order: 2          # embedding dimension m for Sample Entropy
        sampen_tolerance_mult: 0.2   # tolerance r = mult × std(nn)
        min_nn_for_sampen: 50    # minimum NN intervals required for SampEn
        dfa_short_nvals_lo: 4    # short-range DFA lag window lower bound
        dfa_short_nvals_hi: 12   # short-range DFA lag window upper bound (exclusive)
        min_nn_for_dfa: 16       # minimum NN intervals required for DFA α₁
```

Both metrics return `NaN` gracefully when fewer than the minimum required samples
are available, or when the optional library is not installed.

**Enable EDA / GSR features** (for multimodal datasets with an electrodermal
activity channel):

```yaml
features:
  eda:
    enabled: true               # default: true whenever an eda channel exists
    target_sfreq_hz: 50.0       # internal downsample rate for decomposition
    decomposition_method: "neurokit"   # passed to neurokit2.eda_process()
```

MNDM identifies the EDA channel by MNE's native `"gsr"` channel type — type
your channel accordingly in `preprocess.channel_typing` or
`preprocess.datasets.<id>.physio_tsv_inject` (see below), rather than
`"misc"`, so it survives the pre-resample channel pruning step:

```yaml
preprocess:
  channel_typing:
    enabled: true
    datasets:
      <dataset_id>:
        rules:
          - {regex: "^EDA$", type: "gsr"}
```

For datasets where EDA lives in a companion BIDS `*_physio.tsv.gz` file
(common for BioPac/AcqKnowledge exports), use `physio_tsv_inject` (see
`preprocess.datasets.<id>.physio_tsv_inject` in ``preprocess.py``'s
docstring) with `type: "gsr"`:

```yaml
preprocess:
  datasets:
    <dataset_id>:
      physio_tsv_inject:
        enabled: true
        channels:
          - {column: "EDA", name: "EDA", type: "gsr", unit: "uS"}
```

For backward compatibility, a channel still typed `"misc"` and named with
`eda`/`gsr` (case-insensitive) is also picked up as a fallback, but the
native `"gsr"` type is recommended since it is what keeps the channel from
being dropped during pre-resample channel pruning.

The full recording is decomposed once (tonic/phasic separation via
NeuroKit2, or a scipy median-filter fallback when NeuroKit2 is not
installed), then summarised per epoch:

| Column | Description |
|---|---|
| `eda_tonic_scl` | Mean tonic skin-conductance level over the epoch (µS, or native unit) |
| `eda_tonic_slope` | Linear slope of the tonic level within the epoch |
| `eda_phasic_scr_rate` | Skin-conductance-response (SCR) events per minute |
| `eda_phasic_scr_amp` | Mean SCR amplitude for events within the epoch |
| `eda_phasic_scr_count` | Number of SCR events within the epoch |
| `eda_phasic_auc` | Mean absolute phasic residual (area-under-curve proxy) |
| `eda_arousal_index` | `scr_rate + |tonic_slope|` (unnormalised, higher = more aroused) |
| `qc_ok_eda` | `False` for the whole file when the channel is flat/saturated |

---

### `robustness` / `coverage`

Minimum data requirements. Recordings shorter than these are skipped.

```yaml
robustness:
  coverage:
    min_seconds: 60      # minimum usable seconds after QC
    min_epochs: 20       # minimum usable 8 s windows

  # Per-dataset overrides (e.g. ICU data is often shorter)
  datasets:
    my_dataset:
      min_seconds: 30
      min_epochs: 10
```

**Enable reviewer-facing QA baselines** (recommended for new datasets):

```yaml
robustness:
  review_qc:
    baseline_comparisons:
      enabled: true       # exports mean/std per feature vs shuffled null
    null_sanity_tests:
      enabled: true       # checks trajectory length vs white-noise surrogate
    mnps_mnj_sanity:
      enabled: true       # checks Jacobian conditioning and anisotropy
```

---

### `mnps_9d` / `mnps_projection` / `mnps_3d`

These define the measurement contract itself.  
**Do not change these without understanding the versioning implications.**

The canonical pipeline is:

```
features  ──W_9D──►  coords_9d  ──P (v1_mapping)──►  mnps_3d
```

The defaults in `config_ingest_common_eeg.yaml` implement the v2.0 contract.  
To check what version is active: look for `mnps_9d.definition_version: "2.0"`.

For MNDM 2.N anchored outputs:

```yaml
mnps_projection:
  export_contracts:
    subject_anchored: true
    cohort_anchored: true
  anchor_auto_fit:
    enabled: true
    anchor_id: "my_dataset_cohort_auto_v2_1"
    anchor_source: "my_dataset_all_subjects_features_table"
    scale_method: "iqr"   # iqr | mad | qn
    min_subjects: 3
```

`export_contracts.cohort_anchored: true` alone is not enough. Cohort layers are
emitted only when an anchor source is active (`mnps_projection.anchor.path`,
`mnps_projection.anchor_auto_fit.enabled: true`, or CLI `--fit-anchor`).

#### Strict robust-z export guardrail

`/features_robust_z` is the un-clipped feature surface consumed by embodied
`anchor_state`; it is not the coordinate projection surface. New runs reject
columns with too little finite support or a MAD-derived scale at/below the
configured floor:

```yaml
mnps_projection:
  feature_export:
    degenerate_scale_policy: "nan"  # default: emit NaN plus per-feature status
    degenerate_scale_eps: 1.0e-9
    min_finite_count: 3
```

Set `degenerate_scale_policy: "eps_floor"` only to reproduce a legacy
strict-robust-z export. It may reproduce prior extreme values; it does not
change `features_projection_z`, MNPS coordinates, or frozen H5 files.

---

### `conventional_eeg` (optional)

Adds classic qEEG comparator columns **beside** the MNPS pipeline.  
Useful for validation against established biomarkers.

```yaml
conventional_eeg:
  enabled: true
  packs: ["tier1", "complexity"]   # "tier1" = band power + ratios + alpha peak
                                    # "complexity" = SpEn, PE, Hjorth
                                    # "connectivity" = PLV/coherence between channel pairs
                                    # "coma" = suppression ratio, ADR, burst-suppression proxy
```

Outputs appear as `eeg_conventional_*` columns in the feature table  
and as `/extensions/conventional_eeg/*` in the HDF5.

---

### `phase_anchor` (optional — EAP Mål B/C)

Per-epoch cardiac and respiratory phase extraction. Disabled by default; opt in
per dataset. Results land in `features.parquet` alongside all other features and
are automatically carried into HDF5 by `mndm.cli summarize`.

```yaml
phase_anchor:
  enabled: true                     # disabled by default in common_eeg.yaml
  ecg_bipolar: false                # true → compute row[0] − row[1] for 2-channel ECG (ANPHY)
  frontal_eeg_channels: ["F3", "F4"]  # channel names for HEP; empty → global EEG mean
  hep_window_lo_s: 0.200            # HEP onset relative to R-peak (s)
  hep_window_hi_s: 0.600            # HEP offset relative to R-peak (s)
  resp_bandpass_lo_hz: 0.10         # respiratory bandpass (Hz)
  resp_bandpass_hi_hz: 0.50
  chunk_minutes: 5                  # chunk size for NeuroKit2 on whole-night ECG
  min_rpeaks_epoch: 5
  min_rpeaks_hep: 3
```

Dataset-specific channel names:

| Dataset | `ecg_bipolar` | `frontal_eeg_channels` | Notes |
|---------|--------------|----------------------|-------|
| RichSleep | `false` | `["F3-A2", "F4-A1"]` | ECG + Chest → full output |
| ANPHY | `true` | `["Fp1","Fp2","F3","F4","Fz"]` | ECG1−ECG2 bipolar; no resp |
| BOAS | `false` | `["F3", "F4"]` | PSG_THOR resp; no ECG |

Missing modalities always produce NaN columns, never an error.

Output columns added to the feature table: `phi_cardiac_mean`, `phi_resp_mean`,
`rr_interval_ms`, `hr_bpm`, `resp_rate_bpm`, `inhale_fraction`, `hep_amplitude`,
`n_rpeaks_in_epoch`, `pa_cardiac_quality`, `pa_resp_quality`.

### `anchor_state` guardrails (optional — EAP)

`anchor_state` is a physiology-aware composite, not an EOG-only arousal
surrogate. The default `sympathetic_index` requires raw ECG, PPG, or pupil
support on each row; EOG remains available for ocular indices. Invalid,
unsupported, degenerate, and gross-scale components become `NaN` before the
composite is computed.

```yaml
anchor_state:
  enabled: true
  v2:
    enabled: true                  # additive respiration/ocular/cardioresp family
  guards:
    abs_max: 10000.0               # reject finite robust-z values beyond this bound
  validation:
    enabled: true                  # write anchor_state_validation to run manifest
    blocking: false                # set true only in smoke/CI runs
    max_over_iqr: 1000.0
```

The additive `/anchor_quality` surface contains `<component>_eligible`,
`<component>_valid`, and `anchor_valid_fraction` alongside the legacy quality
columns. Its attrs carry `quality_surface = "v2"` and the guard-policy version.

---

### `normalization` (optional — multi-site only)

ComBat-based batch harmonization for multi-site datasets.  
Do not enable for single-site/single-scanner data.

```yaml
normalization:
  enabled: true
  method: "combat"
  scope: "post_features"           # applied before MNPS projection
  batch_key: "hospital"            # column in participants.tsv with site labels
  covariates: ["group", "age"]
```

---

### `regional_mnps` (optional)

Per-channel-group (EEG) or per-network (fMRI) trajectories and Jacobians.  
Requires `robustness.ensembles` to define which channels belong to which group.

```yaml
robustness:
  ensembles:
    enabled: true
    groups:
      frontal:          ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"]
      central:          ["C3", "C4", "Cz"]
      parietal_occipital: ["P3", "P4", "Pz", "O1", "O2", "Oz"]
      temporal:         ["T7", "T8", "TP9", "TP10"]

regional_mnps:
  enabled: true
  stratified:
    enabled: true      # adds 9D trajectories per region
  block_jacobians:
    enabled: true      # adds block-Jacobian summaries per region (EEG only)
```

---

### `event_locked` (optional — derived layer)

Aligns MNPS trajectories to annotated point events (sleep spindles, stimulation onsets, etc.).  
This is a **derived analysis layer**; it does not modify the canonical HDF5 outputs.

**Classic spindle/annotation event-locking** (`kind: csv`):

```yaml
event_locked:
  datasets:
    my_dataset:
      enabled: true
      event_source:
        kind: "csv"
        source_path: "path/to/events.tsv"
      event_types: ["sleep_spindle"]
      stage_filter: ["N2"]
      reference: "peak"
      bins:
        pre_near_ms:  [-4.0, 0.0]
        post_near_ms: [0.0,  4.0]
      controls:
        n_controls_per_event: 3
        exclusion_margin_sec: 30.0
        seed: 42
      export:
        write_parquet: true
        write_csv: true
```

**Direct BIDS events.tsv event-locking** (`kind: bids_events`):

Reads event onsets directly from the companion BIDS `*_events.tsv` file.
**Does NOT require a `derived:task_state_label` column.**  Suitable for any
BIDS cognitive-task dataset.  Event windows may freely cross unlabeled
recording periods (default behaviour; no `stage_filter` applied).

```yaml
event_locked:
  datasets:
    my_dataset:
      enabled: true
      event_source:
        kind: bids_events
        trial_type_column: trial_type   # default
        onset_column: onset             # default
        duration_column: duration       # default
        exclude_types:                  # optional blacklist
          - Boundary
          - Baseline_2s
          - n/a
      event_types:                      # whitelist of trial_type values to keep
        - "StimOn"
        - "TargetPresented"
        - "ResponseWindow"
      reference: onset
      stage_filter: []                  # empty = accept all windows
      bins:
        baseline: [-4.0, 0.0]
        early:    [0.0,  2.0]
        late:     [2.0,  6.0]
      export:
        write_parquet: true
        write_csv: true
```

Each run logs `BIDS event-lock (<filename>): N/M events kept (excluded_type=X, excluded_onset=Y)`.
If `N=0`, check that `event_types` matches the actual `trial_type` values in the file.

**`event_source.source_path` vs. dataset-level `csv_source_glob`** — these
look similar but are resolved very differently, and mixing them up silently
points every subject/run at the same unrendered path:

* `event_source.source_path` (nested under `event_source:`) is read
  **verbatim** by `resolve_event_table_for_event_locked()` — it does **not**
  substitute `{events_core}`/`{events_stem}`/`{events_dir}`. Use it only for
  a single, literal, dataset-wide annotation file shared by every run.
* Dataset-level `csv_source_glob` / `csv_source_globs` (siblings of
  `event_source:`, not nested inside it) **are** rendered per run via
  `_resolve_event_locked_csv_sources()`, substituting `{events_core}` (the
  run's `*_events.tsv` stem minus the trailing `_events`), `{events_stem}`,
  and `{events_dir}`. Use this whenever each run has its own annotation CSV
  — including absolute, non-wildcard, `{events_core}`-templated paths
  pointing outside the raw BIDS tree (e.g. a separately maintained sidecar
  directory). This is how `event_locked.datasets.ds004587` below resolves
  one recovered-trial CSV per run.

**Trial-level event-locking from an externally recovered per-run CSV**
(ds004587 Illusion Game, `kind: csv` with a `{events_core}`-templated
`csv_source_glob` pointing outside the raw dataset):

```yaml
event_locked:
  datasets:
    ds004587:
      enabled: true
      profile: "ds004587_ig_trial_v1"
      event_source:
        kind: "csv"
      csv_source_glob: "J:/processed/openneuro/ds004587_ig_trial_sync/{events_core}_ig_trials_v1.csv"
      event_types: ["ig_trial"]
      stage_filter: []          # task data has no sleep-stage axis
      reference: "onset"
      bins:
        at_trial: [-2.0, 2.0]   # +/-2s: nearest-window match at 8s/4s-step MNPS cadence
      controls:
        n_controls_per_event: 0  # no meaningful non-trial control condition here
      export:
        write_parquet: true
        write_csv: true
```

ds004587's raw EEG `events.tsv` carries only a single "recording start"
marker (no hardware trial triggers), so per-trial `onset_sec` is instead
recovered offline from the injected `LUX` photosensor channel via
landmark-consensus clock-offset estimation
(`mndm.pipeline.ds004587_lux_sync`, run by
`project/scripts/28_ds004587_lux_trial_sync.py` *before* `summarize`) and
written as one CSV per run at the `csv_source_glob` path. Trials from a run
that failed the sync quality gate, or that fall outside the run's
landmark-bracketed interval, carry `onset_sec = NaN` in that CSV and are
therefore excluded by the alignment step (`n_events_excluded_non_finite`),
never extrapolated. Behavioral fields outside the standard `EventTable`
schema (`illusion_strength`, `type`, `correct`, `rt`, `block_number`,
`trial_number`, `qc_ok_event_sync`, `within_sync_bracket`) are not dropped —
`load_event_table_from_csv` folds any unrecognized CSV columns into a single
JSON string in the exported `event_metadata_json` column (see
`Output_variables_guide.md`).

Because MNPS windows are 8s wide with a 4s step
(`config_ingest_common_eeg.yaml`) while IG trials occur roughly every 1-3s,
several trials commonly share the same nearest window; `bins: at_trial:
[-2, 2]` intentionally replaces the RichSleep pre/post-event bin scheme
with a single narrow "nearest window" bin, and downstream analysis (see
`illusionGame_EAP/src/09_trial_level_event_locked_eap.py`) must collapse
each trial (`event_id`) to its single nearest window (`min(abs(rel_time_sec))`)
before running trial-level statistics.

---

### `block_native` (optional — derived layer)

Generates analysis windows relative to inferred **temporal blocks**  
(photic stimulation blocks, task phases, eyes-open/closed segments).  
Better than event-locked when the scientific question concerns position *within* a block.

```yaml
block_native:
  datasets:
    my_dataset:
      enabled: true
      source:
        kind: "stage_blocking"   # reuses stage_blocking from epoching
        label_column: "value"
      window_profile:
        kind: "sliding"
        window_length_sec: 4.0
        step_sec: 2.0
        emit_relative_position: true
        min_block_sec: 4.0
      export:
        write_parquet: true
        write_csv: true
```

---

## 4. Common recipes

### Recipe A — Simple resting-state EEG (most common case)

```yaml
version: 2.0
modality: eeg
imports:
  - "./config_ingest_common_eeg.yaml"
datasets:
  - ds003490
source:
  name: "Resting-state EEG"
  dataset_id: "ds003490"
  doi: "10.18112/openneuro.ds003490.v1.1.0"
paths:
  received_dir: "E:/datasets/received"
  processed_dir: "E:/datasets/processed"
  dataset_received_dirs:
    ds003490: "E:/datasets/received/ds003490"
epoching:
  length_s: 8.0
  step_s: 4.0
robustness:
  review_qc:
    baseline_comparisons: {enabled: true}
    null_sanity_tests:    {enabled: true}
```

---

### Recipe B — Sleep EEG with stage labels

```yaml
version: 2.0
modality: eeg
imports: ["./config_ingest_common_eeg.yaml"]
datasets: [ds005555]
source:
  name: "Sleep PSG Study"
  dataset_id: "ds005555"
paths:
  dataset_received_dirs:
    ds005555: "E:/datasets/received/ds005555"
  processed_dir: "E:/datasets/processed"
epoching:
  length_s: 8.0
  step_s: 4.0
  datasets:
    ds005555:
      sampling:
        stage_columns: ["stage_hum"]
        prefer_events_stage_in_summary: true
robustness:
  coverage:
    datasets:
      ds005555:
        min_seconds: 120
        min_epochs: 30
```

---

### Recipe C — Multimodal EEG+ECG (HRV features)

```yaml
version: 2.0
modality: eeg
imports: ["./config_ingest_common_eeg.yaml"]
datasets: [ds003838]
source:
  name: "EEG+ECG Cognitive Task"
  dataset_id: "ds003838"
paths:
  dataset_received_dirs:
    ds003838: "E:/datasets/received/ds003838"
  processed_dir: "E:/datasets/processed"
epoching:
  length_s: 8.0
  step_s: 4.0
features:
  ecg:
    hrv:
      enabled: true
      superwindow_s: 60.0
      min_nn_intervals: 20
      min_coverage_fraction: 0.5
      complexity:
        enabled: true            # adds ecg_hrv_sampen + ecg_hrv_dfa_alpha1
conventional_eeg:
  enabled: true
  packs: ["tier1", "complexity"]
```

---

### Recipe E — ICA-cleaned cognitive-task EEG (no dedicated EOG channels)

For BrainVision or EasyCap recordings without separate VEOG/HEOG channels,
use Fp1/Fp2 as EOG proxies:

```yaml
version: 2.0
modality: eeg
imports: ["./config_ingest_common_eeg.yaml"]
datasets: [ds006848]
source:
  name: "Verbal Working Memory EEG"
  dataset_id: "ds006848"
paths:
  dataset_received_dirs:
    ds006848: "E:/datasets/received/ds006848"
  processed_dir: "E:/datasets/processed"
epoching:
  length_s: 8.0
  step_s: 4.0
preprocess:
  artifacts:
    method: "ica"
    ica_n_components: 20
    ica_method: "fastica"
    ica_random_state: 42
    ica_eog_threshold: 3.0
    ica_ecg_threshold: 3.0
    ica_max_components_to_remove: 5
    eog_proxy_channels: ["Fp1", "Fp2"]
    ecg_channel: "ECG"
robustness:
  bad_channels:
    enabled: true
    var_low_factor: 1.0e-4
    var_high_factor: 25.0
    corr_thresh: 0.2
```

After changing preprocessing, always force a full re-extraction:

```powershell
python -m mndm.cli features --dataset ds006848 `
  --config mndm/config/config_ingest_ds006848.yaml `
  --force-features --n-jobs 4
python -m mndm.cli summarize --dataset ds006848 `
  --config mndm/config/config_ingest_ds006848.yaml
```

---

### Recipe F — Multimodal EEG + Phase Anchor (EAP Mål B/C)

For datasets with ECG and/or respiratory belts where you want per-epoch cardiac
phase, respiratory phase, and HEP amplitude alongside the standard MNPS features:

```yaml
version: 2.0
modality: eeg
imports: ["./config_ingest_common_eeg.yaml"]
datasets: [my_multimodal_dataset]
source:
  name: "Sleep PSG with ECG and Respiration"
  dataset_id: "my_multimodal_dataset"
paths:
  dataset_received_dirs:
    my_multimodal_dataset: "K:/ExternalReceivedDatasets/my_dataset"
  processed_dir: "K:/ExternalReceivedDatasets/mndm_rvc_processed"
epoching:
  length_s: 30.0
  step_s: 30.0

# Phase anchor — Mål B/C: per-epoch cardiac and respiratory phase
phase_anchor:
  enabled: true
  ecg_bipolar: false              # set true for ECG1−ECG2 bipolar montage
  frontal_eeg_channels: ["F3", "F4"]  # adjust to your recording's channel names
  hep_window_lo_s: 0.200
  hep_window_hi_s: 0.600
  chunk_minutes: 5                # chunked NeuroKit2 for whole-night ECG

features:
  ecg:
    hrv:
      enabled: true               # standard HRV features alongside phase_anchor
      superwindow_s: 60.0
```

After running `mndm.cli features`, the output `features.parquet` will contain all
standard EEG features plus the `phi_cardiac_mean`, `phi_resp_mean`,
`hep_amplitude`, etc. columns.  No separate join step is needed — `mndm.cli
summarize` carries them into the HDF5 event-locked tables automatically.

---

### Recipe D — Multi-site clinical EEG (ComBat harmonization)

```yaml
version: 2.0
modality: eeg
imports: ["./config_ingest_common_eeg.yaml"]
datasets: [my_multisite_dataset]
paths:
  dataset_received_dirs:
    my_multisite_dataset: "E:/datasets/received/my_multisite_dataset"
  processed_dir: "E:/datasets/processed"
epoching:
  length_s: 8.0
  step_s: 4.0
normalization:
  enabled: true
  method: "combat"
  scope: "post_features"
  batch_key: "site"              # column in participants.tsv
  covariates: ["group", "age"]
  combat:
    winsorize_quantiles: [0.005, 0.995]
robustness:
  coverage:
    min_seconds: 30              # ICU data is often shorter
    min_epochs: 10
conventional_eeg:
  enabled: true
  packs: ["tier1", "complexity", "coma"]
```

---

## 5. Output anatomy

Every run produces a directory named:

```
neuralmanifolddynamics_<dataset>_<timestamp>/
├── command_used.txt           ← exact CLI call to reproduce this run
├── run_manifest.json          ← capabilities, provenance, H5 field guide
├── features_snapshot.json     ← feature table statistics and coverage
├── run_errors.json            ← any subjects/runs that failed (if any)
├── <sub-001>/
│   ├── sub-001_task-rest.h5   ← primary output (HDF5)
│   ├── summary.json           ← MNPS + Jacobian scalar summaries
│   ├── qc_summary.json        ← QC distributions (if review_qc enabled)
│   └── qc_reliability.json    ← reproducibility hashes
└── ...
```

### Reading the HDF5 in Python

```python
import h5py, numpy as np

with h5py.File("sub-001_task-rest.h5", "r") as f:

    # Canonical 3D trajectory [T, 3]  — axis order [m, d, e]
    mnps = f["/mnps_3d"][:]                     # shape (T, 3)
    t    = f["/time"][:]                         # shape (T,)  seconds
    
    # Stratified 9D coordinates [T, 9]
    c9d       = f["/coords_9d/values"][:]        # shape (T, 9)
    c9d_names = f["/coords_9d/names"][:]         # ['m_a','m_e',...]
    
    # Jacobian estimates [W, 3, 3]
    J     = f["/jacobian/J_hat"][:]              # shape (W, 3, 3)
    j_ctr = f["/jacobian/centers"][:]            # shape (W,) — indices into /time
    
    # Raw features [T, K]
    feat       = f["/features_raw/values"][:]    # shape (T, K)
    feat_names = list(f["/features_raw/names"][:])
    
    # Stage / condition labels
    stage = f["/labels/stage"][:]               # shape (T,)  integer codes

# Quick trajectory plot
import matplotlib.pyplot as plt
fig = plt.figure()
ax  = fig.add_subplot(111, projection="3d")
ax.plot(mnps[:,0], mnps[:,1], mnps[:,2], lw=0.7)
ax.set_xlabel("m"); ax.set_ylabel("d"); ax.set_zlabel("e")
plt.show()
```

### Key output files explained

| File | What it contains |
|------|-----------------|
| `command_used.txt` | Exact CLI command + git rev + timestamp. Copy-paste to reproduce. |
| `run_manifest.json` | H5 capability flags, subject index, config digest, `field_guide` with all H5 path explanations. |
| `features_snapshot.json` | Per-feature statistics, coverage fractions, normalization provenance. |
| `summary.json` | Per-subject MNPS scalar summaries: `dist_summary`, `tau_summary`, `tier2_jacobian`, `tier2_emmi`. |
| `qc_summary.json` | Baseline comparisons, null-sanity checks, Jacobian conditioning (when `review_qc` enabled). |

---

## 6. Troubleshooting checklist

**"No datasets found"**  
→ Check `paths.dataset_received_dirs.<id>` points to a directory that exists.  
→ Run `python -m mndm.cli prerequisite-check --dataset <id> --config <yaml>` first.

**"min_epochs not met — skipping"**  
→ The recording is shorter than `robustness.coverage.min_epochs × epoching.length_s`.  
→ Reduce `min_epochs` / `min_seconds` under `robustness.coverage.datasets.<id>`.

**Jacobians missing / all NaN**  
→ Too few usable windows after QC. Check `qc_summary.json.geometry_contract`.  
→ Try lowering `robustness.coverage.min_seconds` to retain more epochs.

**feature column `eeg_hjorth_mobility` missing in features.csv**  
→ This feature is part of the 9D contract (needed for `d_l`).  
→ It is extracted automatically from EEG when `modality: eeg` — check that your preprocessing isn't dropping all channels.

**"config_ingest_common_eeg.yaml not found"**  
→ The `imports:` path is relative to the config file's own directory.  
→ Both files must be in `mndm/config/` (or adjust the import path accordingly).

**H5 exists but `coords_9d` is absent**  
→ Check `mnps_9d.enabled: true` in your effective config (the common base sets this by default).  
→ It can also be absent if `mnps_3d.mode: "direct_features"` overrides the from_v2 path.

**Different results between runs**  
→ Check `qc_reliability.json` — the pipeline should be deterministic for the same input.  
→ Verify git rev in `command_used.txt` matches across runs.

**ICA did not seem to run**  
→ Confirm `preprocess.artifacts.method: "ica"` is under `preprocess.artifacts`, not `preprocess.ica` (wrong path).  
→ Look for `Applied ICA` in the pipeline log. If absent, ICA silently fell back — check log warnings for MNE errors.  
→ Always re-run `mndm features --force-features` after enabling ICA; cached intermediate JSONs are not re-cleaned automatically.

**No EOG/ECG components excluded by ICA**  
→ Check that `eog_proxy_channels` contains valid channel names present in the recording.  
→ Confirm `ecg_channel` matches the channel name in the raw file (case-sensitive).  
→ Lower `ica_eog_threshold` / `ica_ecg_threshold` (default 3.0) if components are missed.

**MNE RuntimeWarning: n_components too high**  
→ The recording's effective data rank is lower than `ica_n_components`.  
→ Reduce `ica_n_components` to a smaller integer or switch to a variance fraction such as `0.995`.

**"YAML file has changed since original run — how do I know what was used?"**  
→ The pipeline copies the active YAML into the run directory as `<config_name>.yaml`.  
→ `run_manifest.json → config.yaml_source.copied_filename` records the copy path.  
→ `run_manifest.json → config.digest_sha256` is the SHA-256 of the full resolved config.

---

## Further reading

| Resource | Location |
|----------|----------|
| Full output schema | `mndm/Output_variables_guide.md` |
| CLI cheat sheet | `mndm/Command_cheat_sheet.md` |
| Common base config (EEG) | `mndm/config/config_ingest_common_eeg.yaml` |
| Full template | `mndm/config/config_template.yaml` |
| Non-BIDS BDF adapter template | `mndm/config/bdf_config_ingest_template.yaml` |
| Quickstart notebook | `quickstart.ipynb` (repo root) |
| Article | `project/articles/NeuralManifoldDynamics/` |
| Docs | https://neuralmanifolddynamics.readthedocs.io |

---

## RichSleep Sleep-EAP Phase 2

Use `config_ingest_richsleep_rvc_trv_phase2.yaml` as an overlay over the frozen
RichSleep RVC/TRV configuration. It redirects `processed_dir` and introduces
only additive, versioned Phase 2 products:

- `sleep_eap_phase2`: contracts and detector parameters for raw-EEG sigma
  strength, N2/N3 slow oscillations, SO-spindle coupling, and REM theta.
- `phase_continuous`: opt-in H5 embedding of
  `phase_continuous_v1` sidecars under `/extensions/phase_continuous_v1`.
- `non_event_risk`: opt-in H5 embedding of seeded, stratified
  `non_event_risk_v1` sidecars under `/extensions/non_event_risk_v1`.
- The Phase 2 `event_locked` override selects
  `*_spindles_yasa_v3_*.csv`, so event-locked rows carry the source
  spindle-strength and SO-spindle fields when available. The base RichSleep
  v1 profile remains unchanged.
- Script 20 additionally writes Parquet-only `event_phase_n3_so_v1` and
  `event_phase_rem_theta_v1` sidecars. The former is N3-SO trough/up-state
  referenced; the latter contains one row at the midpoint of each scored
  30-s REM epoch. These are separate carrier contracts, not spindle products,
  and are not H5 embedded in this release.

First produce sidecars with:

```text
python project/scripts/16_yasa_spindle_detection.py --dataset RichSleep --contract v3
python project/scripts/20_sleep_eap_phase2_extract.py --subject xn001
python project/scripts/21_deliver_event_phase_v3.py --phase-v2-dir <phase_resolved_v2> --spindle-root <xndata> --output-dir <phase_resolved_v3>
```

Then run the normal features/summarize commands with the Phase 2 overlay to
embed already validated sidecars. Do not use the overlay to overwrite existing
v1/v2 run roots. `matched_control` remains a geometry control and must not be
used as the hazard-model risk set. Missing cardiac or respiratory phase in the
new N3/REM sidecars is represented by NaN plus validity/QC flags, never a
zero-phase sentinel.
