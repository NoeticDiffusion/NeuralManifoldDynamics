# Claim Ledger — Sleep Spindle / Event-Locked MNPS Track

**Last updated**: 2026-05-08 (session 010)  
**Track**: Sleep spindle event-locked analysis, ds005555 sub-1 to sub-5  
**Maintainer**: implementer + architect review  

This ledger tracks the epistemic status of every material claim in the sleep spindle track. Each entry is dated and categorized. Entries are appended; they are never silently overwritten.

---

## Category definitions

| Category | Meaning |
|----------|---------|
| **EVR — Established external result** | Supported by peer-reviewed literature or authoritative documentation |
| **IVR — Internal validated result** | Supported by code, tests, reproducible scripts, or committed experiment outputs |
| **PLI — Plausible interpretation** | Reasonable reading of existing evidence, not yet directly proven |
| **SPE — Speculative extension** | Creative hypothesis, not yet tested |
| **REJ — Rejected / falsified** | Tested and found not to hold under current conditions |
| **DQF — Detector QC failure** | Technical result showing measurement instrument is not yet fit for purpose |

---

## Pipeline and infrastructure claims

### [IVR-001] — 2026-05-08
**Claim**: The event-locked MNPS export pipeline produces finite, provenance-complete spindle/control rows with matched N2 controls.  
**Evidence**: `smoke_real_h5_event_locked.py` passed end-to-end on ds005555 sub-1. 8124 rows, 54 columns, 8124/8124 finite MNPS values, 864/864 controls matched.  
**Scope**: ds005555 sub-1, PSG_F3 channel, YASA 0.7.0 detector-derived events, profile `sleep_spindle_event_locked_v1`.  
**Limitations**: Single subject, single channel. Detector reliability not established (see DQF-001).

---

### [IVR-002] — 2026-05-08
**Claim**: The `/time` axis in event-locked HDF5 outputs equals `(window_start + window_end) / 2`, and `dt` equals `median(diff(window_start))`, for runs using feature-epoch bounds.  
**Evidence**: `mndm/tests/test_time_axis_regression.py`, 20/20 tests passed. Pre-fix value was `time[0] = 4.0 s`; post-fix = `3.0 s`. Pre-fix `dt = 4.0 s`; post-fix = `2.0 s`. Coverage pre-fix was 2× actual recording span.  
**Scope**: Any `summary.py` run where `sub_frame` contains `t_start` and `t_end` columns.  
**Limitations**: Tests replicate the derivation logic extracted from `summary.py`; they do not run the full pipeline on real data.

---

### [IVR-003] — 2026-05-08
**Claim**: Export provenance is complete and contains `profile_name`, `window_length_s`, `window_step_s`, `alignment_reference`, `control_seed`, `event_source_path`, `annotation_source_hash`, `n_events_input`, `n_events_aligned`, `n_events_excluded_transition`, `match_success_rate`.  
**Evidence**: `smoke_real_h5_event_locked.py` asserts all fields non-None; provenance section confirms `annotation_source_hash = 81b3369b26d90fd4...` (64-char SHA-256 of YASA CSV).  
**Scope**: Profile-driven runs via `event_locked_profile_from_config` + `export_config_from_profile`.  
**Limitations**: The hash identifies the specific YASA output file; it does not validate the detector's biological accuracy.

---

### [IVR-004] — 2026-05-08
**Claim**: The 30-second sleep staging epochs in ds005555 `events.tsv` map correctly onto 6 s/2 s feature windows as interval labels.  
**Evidence**: `smoke_6s_config_driven.py` confirmed N2 window counts consistent with expected N2 sleep proportion.  
**Scope**: ds005555 sub-1 PSG run.

---

### [IVR-005] — 2026-05-08
**Claim**: Stage-transition exclusions (46 of 288 events, 16%) are within the acceptable range (< 20%) for the `stage_transition_margin_sec` setting in the current alignment config.  
**Evidence**: `qc_spindle_detections.py` output; smoke test alignment QC.  
**Scope**: ds005555 sub-1, `sleep_spindle_event_locked_v1` profile.

---

## Detector claims

### [DQF-001 — REVISED 2026-05-08] — Metric error corrected; baseline was within range
**Original claim**: YASA defaults produce implausibly high rates (86.8/h, 73.6/h).  
**Correction**: The /h metric was compared against an incorrect 5–15/h reference. The correct reference metric is **spindles/minute per channel** (Purcell et al. 2017, N=11 630: 1.88/min N2; PLOS One range: 0.70–4.80/min).  
**Converted values**:
- Default YASA, `freq_sp=(11,16)`: 288 events → **1.45/min N2** — within range ✓  
- Default YASA, `freq_sp=(12,15)`: 184 events → **0.92/min N2** — within range ✓  
- "Calibrated" (rms=3.0 + remove_outliers): 54 events → **0.27/min N2** — BELOW range ✗  

**Multi-subject check (sub-1 to sub-5, DEFAULT freq_sp=(12,15))**:  
Mean N2 rate = **1.03/min** (range 0.55–1.73/min); **4/5 subjects within reference range**.  

**Revised status**: DQF-001 is **retracted**. The default YASA detection (freq_sp=(12,15) or (11,16)) is plausible for this PSG dataset on PSG_F3. The previous "calibrated" parameter set (IVR-006) was over-corrected (too strict) and is now superseded.

**Remaining concern**: N3 spindle detection returns 0 in 4/5 subjects with default params. N3 spindles are expected (~1.0–1.8/min per Purcell). This may reflect PSG_F3 sensitivity limits in deep sleep, or a staging/detection mismatch. Flagged for investigation but does not block N2-based analysis.

---

### [DQF-002 — REVISED 2026-05-08]
**Original claim**: 39.3% F3/C3 temporal overlap indicates many false positives.  
**Correction**: With the revised /min metric showing both channels in plausible range, the 39.3% overlap is better explained by PSG_F3 and PSG_C3 being non-homologous sites (frontal vs central) rather than by false positives. Single-channel detections on frontal vs central electrodes showing partial co-occurrence is expected.  
**Revised status**: Not a QC failure; remains a note about channel-pair non-homology.

---

## MNPS measurement claims

### [PLI-001] — 2026-05-08 — **(Pending calibrated detector)**
**Plausible interpretation**: Detector-derived event rows show lower mean m, d, e than matched N2 controls (spindle event: m=0.265, d=0.231, e=−0.087; control: m=0.340, d=0.279, e=+0.024; diff e = −0.111).  
**Evidence**: `descriptive_event_vs_control.py` on the 8124-row event-locked Parquet.  
**Status**: Descriptive observation only. **Cannot be attributed to spindles while DQF-001 is unresolved.** The difference may reflect detector-selection bias (YASA tends to flag lower-entropy segments), N2 sub-stage effects, or other confounds.  
**Required for promotion to IVR**: (1) Calibrated detector with plausible rate; (2) Replication in ≥1 additional subject; (3) Formal statistical test with appropriate paired structure.

---

### [REJ-001 — REVISED 2026-05-08]
**Original claim rejected**: "Default YASA F3/C3 detections are unsuitable."  
**Correction**: With the /min metric recalculation, default YASA on PSG_F3 produces 0.55–1.73/min N2 across 5 subjects (mean 1.03/min), which is within the reference range. The original rejection was based on an incorrect /h comparison.  
**Revised status**: Default YASA with freq_sp=(12,15) is **plausible for exploratory use** on this dataset. The "overpermissive" label is retracted. The export with the original annotations (SHA-256 = 81b3369b26d90fd4) is now considered a **plausible baseline** rather than a DQF artifact.

---

## External result anchors

### [EVR-001 — REVISED 2026-05-08]
**Original claim (retracted)**: Typical healthy adult spindle rates are 5–15/h.  
**Corrected claim**: The correct primary metric is **spindles/minute per channel** per sleep stage:
- N2: ~1.88/min mean (Purcell et al. 2017, N=11 630, C3/C4); range 0.70–4.80/min (PLOS One validation study)
- N3: ~1.45/min (Purcell 2017)
- Warning zones: < 0.3–0.5/min (under-detection) or > 4–5/min (over-detection)

**Note on channel**: Purcell et al. use C3/C4 (central). Frontal channels (F3) may show different absolute rates. The 0.7–4.8/min range remains a reasonable approximate reference.  
**Sources**: Purcell et al. (2017) Nature Communications; PLOS One validation study; AASM sleep scoring manual.

### [EVR-002]
**Claim**: YASA 0.7.0 `spindles_detect` uses relative power, correlation, and RMS thresholds tunable via the `thresh` parameter. Default thresholds are documented as relatively permissive starting points.  
**Sources**: YASA docs https://raphaelvallat.com/yasa/build/html/generated/yasa.spindles_detect.html

---

## Detector calibration results

### [IVR-006] — 2026-05-08
**Claim**: YASA with `remove_outliers=True`, `freq_sp=(12,15)`, `rel_pow=0.30`, `corr=0.70`, `rms=3.0`, `min_distance=700 ms` produces 54 spindles at **16.3/h N2** on ds005555 sub-1 PSG_F3 — within the plausible 5–20/h range.  
**Evidence**: `calibrate_yasa_final.py` + `detect_spindles_yasa_calibrated.py`. Output file: `sub-1_task-Sleep_acq-psg_spindles_yasa_calibrated.csv`.  
**Source tag**: `detector:yasa-0.7.0/calibrated/freq12-15Hz_relp0.30_corr0.70_rms3.0_md700ms_rmoutliers`  
**Key metric**: Duration 0.69 ± 0.12 s, Frequency 12.94 Hz, Amplitude 40.2 µV.  
**Resolves**: DQF-001 for this parameter set. The calibrated detector is **suitable as a starting point for exploratory analysis**, subject to the limitations below.  
**Limitations**: (1) `remove_outliers=True` uses YASA's internal outlier criterion (feature-space), which is data-driven but not externally validated for this PSG equipment. (2) Single subject. (3) Not compared to expert annotation. (4) N=54 events is sufficient for exploratory analysis but underpowered for robust estimates.

### [DQF-001 — PARTIALLY RESOLVED] — 2026-05-08 update
**Status update**: DQF-001 is resolved for the calibrated parameter set (IVR-006). The **overpermissive baseline (288 events, 87/h) remains rejected** (REJ-001). The calibrated CSV replaces it for any future event-locked MNPS runs.

### [REJ-002] — 2026-05-08
**Rejected approach**: F3∩C3 consensus filtering (0.3–0.5 s tolerance) with either default or calibrated parameters produces event counts that are too low for the available N2 data (3–25/h with N=8–83 events at 0.3 s tolerance).  
**Reason**: C3 detections are substantially fewer than F3 (47 vs 70 with calibrated params) and the intersection is too sparse. The PSG channels are not truly homologous electrode sites; F3 and C3 show different sensitivity on this recording.  
**Action**: Consensus filtering is not recommended for this dataset and channel pair. Use calibrated single-channel (IVR-006) instead.

## Next hypotheses (SPE)

### [SPE-001 — superseded by IVR-006]
~~Stricter YASA thresholds may produce rates in the 5–20/h range.~~  
**Resolved**: `remove_outliers=True` + stricter thresholds achieves 16.3/h (IVR-006).

### [SPE-002 — partially tested, not recommended]
~~F3∩C3 consensus filtering may yield a higher-precision event set.~~  
**Tested**: Consensus over-filters to below-range N for this channel pair (REJ-002).

### [SPE-003]
After calibrated detector (IVR-006) is used, the `event` bin MNPS values may show a different pattern from the `pre_far` baseline than from matched controls.  
**Status**: Testable now that calibration is complete. Next step: run event-locked export with calibrated CSV.

### [SPE-004] — SUPERSEDED 2026-05-08
A second subject from ds005555 with similar N2 duration may show a similar calibrated rate.  
**Status**: Superseded by IVR-007 (multi-subject QC pass).

---

## Multi-subject batch results — 2026-05-08 (session 010)

### [IVR-007] — 2026-05-08
**Claim**: All 5 subjects (sub-1 to sub-5) pass the QC gate for the event-locked MNPS pipeline using canonical YASA 0.7.0 parameters on PSG_F3 N2.  
**Evidence**: `batch_event_locked.py` — 5/5 PASS; N2 spindle rates 0.55–1.73/min; all 5 bins populated; control match rate 1.00; finite MNPS = 100%; transition exclusion 4–12%.  
**Scope**: ds005555, PSG_F3, YASA 0.7.0 canonical (freq_sp=(12,15), defaults, N2 only), protocol_n2_event_locked_v1.  
**Limitations**: PSG_F3 (frontal), not central. Detector-derived events, not ground truth. N3 not analysed.

### [IVR-008] — 2026-05-08
**Claim**: Descriptive MNPS event vs matched-control per bin was computed across 5 subjects (48 054 rows total).  
**Evidence**: `descriptive_multi_subject.py` — per-subject and pooled results in `multi_subject_descriptive.csv`.  
**Scope**: Same as IVR-007.  
**Limitations**: Exploratory only. No inferential statistics. High between-subject variability.

### [PLI-001] — 2026-05-08
**Claim**: Direction consistency for the event bin (t=0 to +3 s) vs matched controls is 60% (3/5) for each MNPS dimension (m, d, e), which is insufficient to claim a stable MNPS spindle signature on PSG_F3.  
**Evidence**: `descriptive_multi_subject.py` direction agreement table. Δm: 3/5 positive; Δd: 3/5 positive; Δe: 3/5 positive.  
**Scope**: Same as IVR-007.  
**Interpretation**: Either true null, small effect hidden by inter-subject variability, or PSG_F3 is not the optimal channel. Central-channel robustness check (PSG_C3) is the logical next step.

---

## Central-channel robustness — 2026-05-08 (session 011)

### [IVR-010] — 2026-05-08
**Claim**: PSG_C3 event-locked MNPS shows 4/5 direction agreement (event > control) in all three MNPS dimensions (m, d, e) for the event bin (t=0 to +3 s) vs matched N2 controls.  
**Evidence**: `batch_event_locked.py --channel PSG_C3` + `compare_channels.py`. C3 QC: 4/5 PASS, 1 WARN (sub-3: 0.26/min, marginally below 0.3/min gate). Delta table: `channel_comparison_deltas.csv`.  
**Scope**: ds005555, PSG_C3, YASA 0.7.0 canonical, N2 only, protocol_n2_event_locked_v1.  
**Limitations**: sub-3 flagged (rate 0.26/min). Sub-5 shows large d/e values (Δd=−1.20, Δe=+1.05), suggesting high within-subject variability or outlier spindle events on C3. Detector-derived events, not ground truth.

### [IVR-011] — 2026-05-08
**Claim**: Cross-channel (F3 vs C3) direction agreement is 4/5 for MNPS-m and 4/5 for MNPS-e; only 2/5 for MNPS-d.  
**Evidence**: `compare_channels.py` cross-channel table. F3/C3 agree on sign for m in 4/5 subjects, e in 4/5 subjects; d only 2/5 (sub-1 and sub-5 disagree substantially).  
**Scope**: Same as IVR-010.  
**Limitations**: Single recording session per subject; no repeat reliability check. Sub-5 d-axis disagreement is large in magnitude.

### [PLI-002] — 2026-05-08
**Claim**: The weak/unstable F3 result (3/5 agreement) is at least partly a frontal-channel limitation: PSG_C3 (central) shows stronger directional consistency (4/5) and cross-channel F3/C3 agreement on m and e suggests these dimensions may carry some channel-robust MNPS spindle signal.  
**Evidence**: C3 meets the pre-specified 4/5 promotion criterion; F3 does not. Cross-channel agreement 4/5 for m, 4/5 for e.  
**Interpretation**: Plausible, not yet validated. Remaining confounds: (1) MNPS projection uses the same H5 (same MNPS space) for both channels — the channel difference is only in spindle detection, not in MNPS coordinates. (2) Sub-5 variability on C3-d/e is unexplained. (3) No claim of effect size, clinical relevance, or NDT interpretation.

---

## Full ds005555 C3 batch — 2026-05-08 (session 013)

### [IVR-012] — 2026-05-08  *(F3 sensitivity confirmed — provisional flag lifted)*
**Claim**: In ds005555, using YASA 0.7.0 detector-derived N2 spindle events on PSG_C3 with canonical `freq_sp=(12,15)` settings, baseline-corrected MNPS-d (operational MNPS-d coordinate; event bin − pre_far bin) was positive in **115/117 QC-passing subject runs** (98%), with median Δd_bc = +0.216. This is an internal validated measurement-layer result, pending F3 sensitivity confirmation and downstream statistical analysis.  
**Denominator**: 128 subjects run, 117 QC-PASS (11 WARN, all rate-related), 0 SKIP. Denominator audit passed (128 unique parquets, 117 event-bin rows matched to 117 QC-PASS entries). Full-load run: `neuralmanifolddynamics_ds005555_20260508_164627`.  
**Evidence**: `_baseline_corrected_all.py` (deduplicated). `baseline_corrected_all_c3.csv`. `batch_event_locked_qc_psg_c3.json`.  
**Scope**: ds005555, PSG_C3, YASA 0.7.0 canonical, N2 only, protocol_n2_event_locked_v2.  
**Limitations**: (1) Detector-derived events, not expert-annotated ground truth. (2) MNPS-d is an operational coordinate in the robust-z projection, not a direct physiological measure. (3) Baseline correction (event bin − pre_far, t=−30 to −10 s) removes level shifts and time-of-night gradients, but also absorbs any sustained MNPS change already present before t=−10 s. (4) No inferential statistics (pending analysis repo). (5) F3 and C3 share the same MNPS trajectory; channel agreement is detector consistency, not independent MNPS replication.

### [IVR-013] — 2026-05-08  *(F3 sensitivity confirmed — provisional flag lifted)*
**Claim**: Baseline-corrected MNPS-m is positive in 97/117 QC-passing subjects (83%); baseline-corrected MNPS-e is positive in 92/117 (79%). Both exceed the 4/5 direction-consistency threshold at 128-subject scale.  
**Evidence**: Same as IVR-012. Median Δbc: m=+0.042, e=+0.024.  
**Scope**: Same as IVR-012.  
**Limitations**: Smaller and more variable effect sizes compared to MNPS-d. m and e overlap with natural N2 MNPS fluctuations more than d does.

### [IVR-014] — 2026-05-08
**Claim**: Full ds005555 H5 dataset (128 subjects, 6s/2s MNDM, `/mnps_3d`, `/coords_9d`, `/labels/stage`) produced at run `neuralmanifolddynamics_ds005555_20260508_164627`, 1.5 GB total.  
**Evidence**: 128/128 H5 files confirmed. Run completed with return code 0. `run_manifest.json` present.  
**Scope**: ds005555, `config_ingest_ds005555_sleep_spindles.yaml`.

### [DQF-003] — 2026-05-08
**Claim**: Raw event-vs-control Δd for sub-5 C3 (= −1.20) is a time-of-night confound, not an event-locked MNPS signal. Baseline-corrected Δd_bc for sub-5 C3 = +0.06.  
**Evidence**: `audit_sub5_c3.py` — bimodal spindle timing (early cluster at ~2000 s), strong MNPS-d time gradient (r=0.55), control matching failure in Q1 (14 vs 1976 rows). Baseline correction resolves the anomaly.  
**Action**: Baseline-corrected delta adopted as primary readout for protocol v2.

### [IVR-015] — 2026-05-08
**Claim**: PSG_F3 detector-derived N2 spindle events (YASA 0.7.0, same canonical params) reproduce the MNPS-d positive direction in 110/111 QC-passing subjects (99%), median Δd_bc = +0.264. Cross-channel sign agreement (all 128 subjects): MNPS-d 126/128 (98%), MNPS-m 115/128 (90%), MNPS-e 106/128 (83%).  
**Denominator**: 128 subjects, 111 PASS, 17 WARN. Denominator audit passed.  
**Evidence**: `_baseline_corrected_all.py --channel PSG_F3`, `_channel_agreement.py`. `baseline_corrected_all_psg_f3.csv`.  
**Scope**: ds005555, PSG_F3, same protocol as C3 (v2). F3 has more WARN subjects (17 vs 11 C3), reflecting lower frontal spindle density in some subjects.  
**Limitations**: F3 and C3 share the same MNPS trajectory (same H5 file). Cross-channel agreement is a detector consistency check, not an independent MNPS replication. No inferential statistics.  
**Implication**: IVR-012/013 provisional flags can be lifted for the primary MNPS-d finding. F3 sensitivity condition met.

### [IVR-009] — 2026-05-08
**Claim**: `mndm --subject N` zero-pads to `sub-00N`, which is incompatible with ds005555's non-zero-padded BIDS subject IDs (`sub-1` ... `sub-5`). Running `summarize` without `--subject` filter processes all subjects correctly.  
**Evidence**: Direct observation; "No epochs for subject sub-002" warning on `--subject 2`. All 5 H5 files created correctly when running without filter.  
**Scope**: ds005555 only (non-zero-padded IDs).  
**Action**: Noted as usability issue; no fix required for current analysis.
