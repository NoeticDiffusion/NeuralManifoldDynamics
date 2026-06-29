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
```

Imports are merged in order. Your overlay keys win over the imported ones.

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
    method: "none"              # "none" | "autoreject" (requires autoreject package)
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
| Quickstart notebook | `quickstart.ipynb` (repo root) |
| Article | `project/articles/NeuralManifoldDynamics/` |
| Docs | https://neuralmanifolddynamics.readthedocs.io |
