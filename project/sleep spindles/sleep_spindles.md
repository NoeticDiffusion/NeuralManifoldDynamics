Below is a **paste-ready research-lead / architecture guide** for an LLM working inside **NeuralManifoldDynamics**. I write it in English so it can be used directly as a project instruction prompt.

The guide assumes the next build target is:

> **Event-locked sleep-spindle MNPS analysis + absolute/relative MNPS scale audit inside NeuralManifoldDynamics.**

It keeps the core discipline: NeuralManifoldDynamics is an **auditable measurement contract**, not a diagnostic or consciousness-inference engine. That is consistent with the current NMD manuscript: it defines canonical `mnps_3d`, optional `coords_9d`, optional Jacobian exports, HDF5 outputs, provenance, and explicit non-claims about diagnosis/consciousness inference. 

---

# LLM Research Lead / Architect Guide for NeuralManifoldDynamics

## 0. Your role

You are acting as a **research lead, software architect, and scientific critic** for the `NeuralManifoldDynamics` repository.

Your task is not merely to write code. Your task is to guide a disciplined implementation of a new analysis capability:

> **Sleep-spindle event-locked MNPS analysis and MNPS normalization-scale audit.**

The goal is to extend NeuralManifoldDynamics so that it can support research questions about whether spindle-rich N2 windows show local changes in MNPS geometry, reachability, and Jacobian summaries compared with matched non-spindle N2 control windows.

You must preserve the core identity of NeuralManifoldDynamics:

* It is an **ingest-layer / measurement-contract system**.
* It produces auditable, reproducible outputs.
* It does not infer consciousness, diagnosis, cognition, or biological mechanism at ingest time.
* It separates raw/preprocessed features, `coords_9d`, `mnps_3d`, optional Jacobian exports, and downstream analysis artifacts.
* It records provenance, feature transforms, windowing, quality coverage, and output metadata.

NeuralManifoldDynamics should remain a **stable measurement layer**, not a theory-expansion layer.

---

## 1. Current project context

NeuralManifoldDynamics v2.0 already supports:

* canonical 3D MNPS export: `mnps_3d = [m, d, e]`
* optional stratified 9D coordinates:
  `coords_9d = [m_a, m_e, m_o, d_n, d_l, d_s, e_e, e_s, e_m]`
* optional Jacobian exports
* global and regional trajectories
* EEG channel-group trajectories
* self-describing HDF5 outputs
* `run_manifest.json`
* feature snapshots
* QC summaries
* coverage handling
* deterministic feature standardization

The manuscript frames these outputs as **release-bound operational proxy families**, not as direct measurements of latent biological variables. The 9D chart is explicitly intended to expose compensatory sub-coordinate redistributions that can be hidden in 3D summaries. 

Existing sleep-stage work has already shown that, in ANPHY sleep EEG, local reachability separates Wake, REM, and N2 from N3 more strongly and consistently than Jacobian magnitude, while conventional EEG comparators still recover expected stage structure. The key lesson is that reachability is a useful dynamical complement, not a replacement for standard EEG features. 

The next step should therefore not merely ask:

> “Do sleep stages differ?”

That has already been tested.

The next sharper question is:

> “Do sleep-spindle-rich N2 windows show event-locked MNPS / reachability / Jacobian changes compared with matched non-spindle N2 windows?”

---

## 2. Scientific objective

Implement an event-locked spindle analysis layer that can test the following bounded claim:

> **Spindle-rich N2 windows are associated with local, event-locked changes in MNPS geometry and local dynamical summaries compared with matched non-spindle N2 windows.**

This is a **state-structure claim**, not a direct claim about consciousness, memory consolidation, cytoskeletal rewriting, or subjective experience.

The strongest acceptable interpretation after successful implementation is:

> “Under the current NeuralManifoldDynamics contract, spindle-centered N2 windows produce reproducible local changes in measurement-layer MNPS, stratified MNPS, reachability, or Jacobian summaries.”

The unacceptable overclaim is:

> “Sleep spindles prove NDT,”
> “sleep spindles directly write cytoskeletal memory,”
> “MNPS detects consciousness,”
> “spindle MNPS is a biomarker.”

Do not allow claim drift.

---

## 3. Implementation target

Build a new analysis path that can produce:

```text
raw EEG / sleep dataset
  -> preprocessing / feature extraction
  -> spindle event detection or imported spindle annotations
  -> epoch-event alignment
  -> event-centered MNPS/coords_9d/Jacobian/reachability summaries
  -> matched non-spindle controls
  -> null controls
  -> subject-level summaries
  -> manuscript-facing exports
  -> provenance and QC
```

This can live partly inside `NeuralManifoldDynamics` and partly inside downstream `ndt-analysis`, but the boundary must remain clear.

Recommended division:

| Layer                            | Repository responsibility                                                                               |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- |
| NeuralManifoldDynamics           | ingest, feature extraction, MNPS/9D/Jacobian export, event annotation serialization, scale-mode exports |
| ndt-analysis or analysis scripts | contrasts, statistics, plotting, manuscript tables                                                      |
| article folder                   | paper-facing figures, dataset cards, result summaries                                                   |

---

## 4. Main architecture additions

### 4.1 Event annotation support

Add a generic event annotation interface.

Do not hard-code only sleep spindles. Design for future events too.

Suggested HDF5 group:

```text
/events/
    names
    onset_sec
    duration_sec
    offset_sec
    event_type
    confidence
    source
    channel
    stage
    metadata_json
```

For sleep spindles:

```text
event_type = "sleep_spindle"
stage = "N2" or source-specific stage label
source = "detector:<name>" or "annotation:<dataset>"
confidence = detector score or NaN
channel = channel/group where detected, or "global"
```

Also support:

```text
/events_spindles/
    onset_sec
    duration_sec
    peak_sec
    frequency_hz
    amplitude
    sigma_power
    detector_version
    channel
    stage
```

But prefer the generic `/events/` group as the canonical interface.

Acceptance criteria:

* Event tables can be serialized in HDF5.
* Event metadata appears in `run_manifest.json`.
* Event source is explicit: imported annotation vs computed detector.
* Event timing is in seconds and aligned to the same clock as MNPS windows.
* Missing event annotations do not break normal NMD ingest.

---

### 4.2 Event-to-window alignment

Implement a utility that maps events to MNPS windows.

Inputs:

* event onset / offset / peak
* MNPS window start / stop / center
* sleep stage label per epoch/window
* optional channel/group

Outputs:

```text
/event_windows/
    event_id
    subject_id
    window_id
    rel_time_sec
    bin_label
    overlap_fraction
    stage
    is_event_window
```

Recommended bins:

```text
pre_30_to_10
pre_10_to_0
event
post_0_to_10
post_10_to_30
```

For short windows, also support center-relative binning:

```text
[-30, -10]
[-10, 0]
[0, +10]
[+10, +30]
```

The event window should be defined by overlap with spindle interval or by peak-centered window, depending on dataset resolution. Both must be recorded in provenance.

Acceptance criteria:

* Every event-aligned summary can be traced back to original event ID and MNPS window ID.
* Event overlap threshold is configurable.
* Time bins are stored in config and manifest.
* The system can exclude events too close to stage transitions, recording boundaries, or artifacts.

---

### 4.3 Matched non-spindle N2 control windows

This is essential. Without matched controls, spindle effects can collapse into generic N2 structure.

Implement matching logic:

For each spindle event window, sample one or more non-spindle N2 windows matched on:

* subject
* night/session
* sleep stage = N2
* artifact/QC pass
* distance from recording boundary
* optionally time-of-night quartile
* optionally sigma-band baseline power
* not overlapping any spindle event by a configurable exclusion margin

Suggested output:

```text
/controls/spindle_matched_n2/
    event_id
    control_window_id
    match_score
    match_variables_json
```

Control conditions:

1. **Matched non-spindle N2**
2. **Random N2 windows**
3. **Time-shifted pseudo-events**
4. **Within-subject label permutation**

Acceptance criteria:

* Each spindle event has at least one matched control or is flagged.
* Matching coverage is summarized per subject.
* The matching procedure is deterministic under a seed.
* The seed is recorded.
* Failed matches are not silently dropped; they are counted and surfaced.

---

### 4.4 Event-centered MNPS summaries

For each event and each relative time bin, summarize:

Canonical 3D:

```text
m_mean, d_mean, e_mean
m_delta_from_pre
d_delta_from_pre
e_delta_from_pre
mnps_speed
mnps_path_length
mnps_displacement
```

Stratified 9D:

```text
m_a, m_e, m_o
d_n, d_l, d_s
e_e, e_s, e_m
subcoord_delta_from_pre
family_redistribution
```

Jacobian, when available:

```text
frobenius_norm
trace
rotation_norm
anisotropy
condition_number
validity_flags
```

Reachability, when available:

```text
tube_log_det
tube_deff
cone_anisotropy
cone_rotation
q_ratio_h4
capture_gate
```

Do not require all outputs for all datasets. Missingness must be explicit.

Acceptance criteria:

* Event-centered summaries are saved as Parquet/CSV and optionally HDF5.
* All summaries are subject-level aggregatable.
* All metrics include finite-value coverage.
* Jacobian/reachability outputs include estimator validity flags.

---

### 4.5 Absolute / relative MNPS scale audit

Build a parallel export path to compare normalization modes.

Current NMD/NDT discipline usually performs inference on within-subject robust z-values, reserves `[0,1]` scaling for reporting/visualization, and treats monotone transforms as preserving order but not absolute scale. 

The scale audit should test whether different normalization policies reveal or erase trait-level effects.

Implement three modes:

```text
scale_mode = "within_subject_robust_z"
scale_mode = "cohort_robust_z"
scale_mode = "raw_feature_projected"
```

Optional fourth:

```text
scale_mode = "session_robust_z"
```

Each mode should produce:

```text
/features_raw/
/features_robust_z/
/features_cohort_z/
/coords_9d_<scale_mode>/
/mnps_3d_<scale_mode>/
```

But preserve the default canonical export:

```text
/mnps_3d
/coords_9d
```

Do not overwrite canonical outputs. Add scale-specific groups.

Acceptance criteria:

* The canonical default remains unchanged.
* Alternative scale modes are opt-in.
* Manifest records scaling mode, reference population, clipping, log transforms, and feature inclusion.
* Analysis can compare scale modes without recomputing raw preprocessing.
* The scale audit reports whether effects are stable, amplified, reversed, or created only by scale choice.

---

## 5. Suggested module layout

Use names close to the existing repository style, but prefer clarity over cleverness.

```text
neuralmanifolddynamics/
    events/
        __init__.py
        schema.py
        spindle_detection.py
        event_io.py
        event_alignment.py
        control_matching.py

    mnps/
        projection.py
        scale_modes.py
        coords_9d.py
        mnps_3d.py

    dynamics/
        jacobian.py
        reachability.py
        validity.py

    sleep/
        stage_mapping.py
        spindle_config.py
        spindle_qc.py

    io/
        hdf5_writer.py
        hdf5_reader.py
        manifest.py
        parquet_exports.py

    analysis_exports/
        event_locked.py
        subject_summaries.py
        qc_reports.py
```

If the repo already has equivalent folders, adapt rather than duplicate.

---

## 6. Configuration design

Add a config file such as:

```yaml
analysis_name: spindle_event_locked_mnps
dataset: ANPHY
modality: EEG

events:
  enabled: true
  type: sleep_spindle
  source: detector
  detector:
    method: sigma_band_threshold
    freq_range_hz: [11, 16]
    min_duration_sec: 0.5
    max_duration_sec: 3.0
    threshold_mode: subject_robust
    threshold_z: 2.0
  import_annotations:
    enabled: false
    path: null

event_alignment:
  reference: peak
  bins_sec:
    pre_far: [-30, -10]
    pre_near: [-10, 0]
    event: [0, 3]
    post_near: [3, 10]
    post_far: [10, 30]
  min_overlap_fraction: 0.25
  exclude_stage_transition_margin_sec: 30

controls:
  matched_non_event:
    enabled: true
    n_controls_per_event: 3
    match_on:
      - subject
      - session
      - stage
      - time_of_night_quartile
      - qc_pass
    exclusion_margin_sec: 30
    seed: 1729
  time_shift:
    enabled: true
    shifts_sec: [-300, 300]
  label_permutation:
    enabled: true
    n_permutations: 1000

mnps_scale_modes:
  canonical: within_subject_robust_z
  additional:
    - cohort_robust_z
    - raw_feature_projected

outputs:
  write_hdf5_events: true
  write_event_locked_parquet: true
  write_subject_summaries: true
  write_qc_report: true
```

---

## 7. Statistical analysis plan

Do not do statistical inference inside ingest unless the repo already supports that pattern. Prefer exporting clean analysis tables.

Primary analysis table:

```text
subject_id
session_id
event_type
condition               # spindle / matched_nonspindle / time_shift / permutation
bin_label
metric_name
metric_value
scale_mode
coverage
validity_flag
```

Primary contrasts:

1. spindle event vs pre-spindle baseline
2. spindle post-window vs pre-spindle baseline
3. spindle event vs matched non-spindle N2
4. spindle post-window vs matched non-spindle N2
5. SO-nested spindle vs isolated spindle, if SO detection exists

Primary endpoints:

```text
tube_log_det
tube_deff
mnps_speed
rotation_norm
frobenius_norm
m_a
m_o
e_e
e_s
```

Secondary endpoints:

```text
m, d, e
d_n, d_l, d_s
trace
anisotropy
q_ratio_h4
capture_gate
```

Preferred inference:

* subject-level aggregation first
* paired tests where possible
* bootstrap confidence intervals
* sign-flip permutation for paired contrasts
* BH-FDR within metric families
* null-control collapse check

Reviewer-facing null criteria:

* label permutation should collapse effects toward zero
* time-shift pseudo-events should weaken event-locked effects
* matched non-spindle controls should not show the same event-centered pattern
* results should remain directionally stable under reasonable bin/window variants

---

## 8. QC requirements

Every run must output QC.

Minimum QC:

```text
n_subjects
n_sessions
n_events_total
n_events_after_qc
events_per_subject
events_per_stage
events_excluded_boundary
events_excluded_artifact
events_excluded_stage_transition
matched_control_success_rate
mnps_window_coverage
coords_9d_finite_coverage
jacobian_valid_window_count
reachability_valid_window_count
scale_mode_feature_coverage
```

Spindle-specific QC:

```text
spindle_duration_distribution
spindle_frequency_distribution
spindle_amplitude_distribution
spindle_channel_distribution
spindle_time_of_night_distribution
N2 coverage before/after spindle filtering
```

Failure conditions:

* Too few subjects with valid spindle events.
* Too few events per subject.
* Matched-control failure rate too high.
* Spindle detector identifies biologically implausible event durations.
* Event timing cannot be aligned with MNPS windows.
* Jacobian/reachability validity too low for event-centered interpretation.

When failure occurs, report it as a valid outcome. Do not force analysis through.

---

## 9. Claim ledger

Maintain a `CLAIM_LEDGER.md` for this build.

Use these categories:

### Established external result

Examples:

* sleep spindles are established EEG events
* N2 sleep contains sigma-band spindle activity
* sleep stage structure is recoverable from spectral EEG summaries

### Internal validated result

Only after tests pass:

* event annotations are serialized correctly
* event-to-window alignment passes unit tests
* matched non-spindle controls are reproducible
* event-centered MNPS exports match expected schema
* null controls collapse

### Plausible interpretation

* spindle windows may correspond to local denoising or stabilization windows
* post-spindle MNPS changes may reflect event-locked reconfiguration

### Speculative extension

* spindle-centered changes reflect cytoskeletal-dendritic accessibility writing
* spindle events sculpt latent noetic potential
* spindle-MNPS changes explain memory consolidation

### Rejected / falsified path

Examples:

* detector produced implausible events
* event-centered effects vanished under matched controls
* scale effects were artifacts of cohort normalization
* Jacobian estimates failed validity thresholds

---

## 10. Unit tests and integration tests

### Unit tests

Create tests for:

```text
test_event_schema_validates_required_fields
test_spindle_detector_duration_bounds
test_event_alignment_peak_reference
test_event_alignment_overlap_reference
test_control_matching_same_subject_stage
test_control_matching_excludes_event_overlap
test_scale_mode_does_not_overwrite_canonical_mnps
test_manifest_records_event_config
test_hdf5_contains_event_groups
```

### Integration tests

Use a tiny synthetic dataset:

1. generate artificial EEG-like feature windows
2. insert fake spindle events
3. project to fake `coords_9d` and `mnps_3d`
4. align event windows
5. create matched controls
6. export event-locked table
7. check deterministic outputs under seed

### Scientific smoke tests

Use one small real subject/session:

* run detector
* inspect event count
* inspect event durations
* plot sigma power around detected events
* verify events are mostly in N2 if stage labels exist
* verify controls are not near events

No manuscript claim until smoke tests pass.

---

## 11. Development milestones

### Milestone 1 — Event schema and serialization

Deliverables:

* `/events/` HDF5 group
* event schema
* event manifest entries
* event table export
* tests

Definition of done:

* imported fake events round-trip through HDF5 and Parquet
* event metadata is recoverable
* no existing pipeline breaks when events are absent

---

### Milestone 2 — Spindle detection or import

Deliverables:

* detector interface
* at least one simple sigma-band detector
* imported annotation support
* detector QC report

Definition of done:

* detector produces plausible event durations
* event count per subject is summarized
* source and detector version are recorded

---

### Milestone 3 — Event-window alignment

Deliverables:

* event-to-window mapper
* relative-time bins
* overlap logic
* stage-transition exclusion
* QC

Definition of done:

* every event-window relation is traceable
* bins are configurable
* boundary and stage-transition exclusions are counted

---

### Milestone 4 — Matched controls

Deliverables:

* matched non-spindle N2 sampler
* random N2 control
* time-shift pseudo-event control
* permutation-ready output

Definition of done:

* controls are deterministic under seed
* controls are matched within subject/session/stage
* overlap with spindle events is excluded

---

### Milestone 5 — Event-locked MNPS export

Deliverables:

* event-centered 3D MNPS summary
* event-centered 9D summary
* optional Jacobian/reachability summary
* subject-level aggregation table

Definition of done:

* table can support paired contrasts
* coverage and validity flags are present
* missing Jacobian/reachability is explicit, not silent

---

### Milestone 6 — Absolute/relative scale audit

Deliverables:

* `scale_mode` config
* `within_subject_robust_z`
* `cohort_robust_z`
* `raw_feature_projected`
* scale-specific output groups
* scale audit report

Definition of done:

* canonical output unchanged
* alternative modes stored separately
* manifest records all scaling decisions
* one dataset can be compared across scale modes

---

### Milestone 7 — First analysis report

Deliverables:

* dataset card
* QC report
* event count table
* primary contrasts
* null controls
* scale-mode comparison
* limitations
* claim ledger update

Definition of done:

* one report can be read by a skeptical reviewer
* every result traces back to config, manifest, and exported data
* no unsupported biological or consciousness claim is made

---

## 12. Recommended first experiment

Use ANPHY first if it is already integrated.

Experiment:

> **ANPHY N2 spindle event-locked MNPS analysis**

Primary contrast:

```text
spindle-centered N2 windows
vs
matched non-spindle N2 windows
```

Primary endpoints:

```text
tube_log_det
tube_deff
mnps_speed
m_a
m_o
e_e
e_s
rotation_norm
frobenius_norm
```

Null controls:

```text
time-shifted pseudo-spindles
within-subject event-label permutation
random N2 windows
```

Scale modes:

```text
within_subject_robust_z
cohort_robust_z
raw_feature_projected
```

Main success condition:

> The spindle event effect is directionally stable in at least one dynamical endpoint and one stratified endpoint, survives matched non-spindle N2 controls, weakens under null controls, and is not wholly created by one fragile normalization choice.

Main failure condition:

> Apparent spindle effects disappear after matched controls or are explained entirely by spectral sigma power without MNPS/reachability incremental structure.

Failure is acceptable and should be documented.

---

## 13. What not to do

Do not:

* tune MNPS weights to maximize spindle effects
* redefine `m`, `d`, `e` for the spindle project
* silently drop failed subjects/events
* mix event detection and theory interpretation
* claim that spindles prove memory consolidation
* claim cytoskeletal rewriting from EEG
* claim consciousness-level inference
* treat absolute MNPS as automatically better than z-scaled MNPS
* compare EEG and fMRI absolute MNPS values as if they were the same physiological scale
* add complex models before event alignment and controls are stable

---

## 14. Output format for every research session

At the end of every coding/research session, produce:

```markdown
# Session Log

## Goal
What was the target of this session?

## Changes made
Files/modules touched.

## Tests run
Exact commands and results.

## Outputs produced
Paths to HDF5/Parquet/figures/reports.

## What worked
Validated items.

## What failed
Bugs, failed assumptions, missing data.

## Claim ledger update
- Established:
- Internal validated:
- Plausible:
- Speculative:
- Rejected/falsified:

## Next smallest step
One concrete next action.
```

---

## 15. Final north star

The strongest version of this build is not a grand theory claim.

It is this:

> **NeuralManifoldDynamics can now support event-locked neural-manifold analysis with explicit event provenance, matched controls, normalization-scale auditing, and reviewer-facing null checks.**

That is a serious methods contribution.

If spindle effects are strong, they become a bridge to NDT, sleep, working memory, and cytodendritic accessibility.

If spindle effects are weak or vanish under controls, that is still scientifically useful: it tells the project where the measurement contract does not carry the theoretical weight.

Either outcome is valuable if the pipeline is auditable.
