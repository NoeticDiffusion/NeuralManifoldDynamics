# NeuralManifoldDynamics — Wishlist for Embodied Anchoring Research

Based on the ds003838 dataset (EEG 64ch + ECG + PPG + pupillometry + behavioral TSV)
and the current analysis gaps identified in the EAP empirical article.

---

## 1. Cardiac-cycle-aligned MNPS windows (`event_locked_rpeaks`)

**What:** For each detected R-peak, output a ±2 s event-locked MNPS window aligned to
the heartbeat, giving one row per heartbeat per subject. Columns: `pre_500ms`,
`post_500ms`, `post_1000ms`, `post_2000ms` bins — or a continuous 10-sample trace.

**Why:** EAP's core entrainment equation (Eq. 4) predicts that manifold variance is
modulated at the cardiac cycle (~1 Hz). All three null coupling results (within-subject
Spearman, between-subject Spearman, pre-trial recall) operate at timescales ≥8 s. This
would be the first direct test of cycle-level predictions. The dataset has ECG for 62 of
86 subjects, giving adequate power.

**Feasibility:** ds003838 provides synchronized ECG; R-peak detection is already part of
the HRV v0.1 pipeline in `anchor_surface.py`. The MNDM H5 file has a continuous `/time`
axis from which MNPS values can be interpolated to any timestamp.

---

## 2. Digit-onset event-locked MNPS (`event_locked_stimuli`)

**What:** For each individual digit presentation within a sequence (e.g., digit 1 of 13,
digit 2 of 13 …), align the MNPS + anchor state to that onset. Output `pre_1s`,
`post_early_1s`, `post_late_2s` bins, plus position-in-sequence as a covariate.

**Why:** The current event-locked analysis uses segment-level onsets (listen / mem5 onset).
Individual digit onsets would reveal:
- Serial position effects on manifold state during encoding (are early digits encoded under
  different MNPS conditions than late digits?)
- Whether per-digit anchor state predicts that digit's contribution to recall accuracy
  (partial-score `NCorrect` allows digit-level scoring if the recall TSV is parsed)
- ERP-linked dynamics: P300 latency/amplitude as a function of anchor state at onset

**Feasibility:** Stimulus timing is available in the task design sidecar. Requires parsing
the 2 s SOA structure from the task design to extract per-digit onsets. Medium effort.

---

## 3. Respiratory phase alignment (`respiration_phase` derived from ECG)

**What:** Estimate instantaneous respiratory phase from ECG-derived respiration (EDR —
respiratory sinus arrhythmia envelope extraction from RR intervals), and output per-window
respiratory phase bin (inspiration / expiration / transition) alongside MNPS coordinates.

**Why:** Respiration is a slow oscillator (~0.2–0.3 Hz) that modulates both vagal tone and
EEG amplitude envelopes. EAP predicts that respiratory phase is a secondary anchor layer
below the cardiac cycle. Current anchor surface does not separate cardiac vs. respiratory
contributions. Respiratory phase gating of MNPS would test whether the anchor surface
captures breath-level variation distinct from HRV.

**Feasibility:** No explicit respiratory belt in ds003838, but RSA-based EDR is feasible
from the ECG. Moderate effort; adds one derived column `resp_phase` and two derived
statistics `resp_rate` and `resp_amplitude`.

---

## 4. Pupil-linked MNPS windows (`pupil_event_locked`)

**What:** For each spontaneous large pupil dilation event (pupil diameter > subject-mean
+ 1.5 SD, sustained > 200 ms), output a ±4 s MNPS window. Also: continuous per-window
pupil summary (`pupil_mean`, `pupil_slope`, `pupil_peak_latency`) as block-native columns.

**Why:** Pupillometry is a noradrenergic / LC-NE index. EAP's anchor surface includes a
`pupil_amplitude` component in HRV v0.1, but it is folded into the composite. Separating
spontaneous dilation events (bottom-up arousal) from task-locked constriction (cognitive
load / attentional narrowing) would distinguish two EAP-relevant pathways. 64 of 86
subjects have pupillometry.

**Feasibility:** ds003838 provides raw pupil data in the pupil folder. Requires dilation
event detection and epoch extraction. Low–medium effort for block-native columns; medium
effort for event-locked epochs.

---

## 5. Pre-recall MNPS state (`recall_onset_event_locked`)

**What:** Align MNPS + anchor state to the recall onset (when the subject begins to speak
the digits back), outputting `pre_4s`, `pre_1s`, `post_2s`, `post_4s` bins. Also include
`NCorrect` and `partialScore` as covariates at the event level.

**Why:** The pre-trial analysis tested the pre-*encoding* anchor state. The untested
question is: what manifold state at *recall onset* predicts accuracy? Recall is a retrieval
operation; EAP predicts that a high-anchor recall onset would allow tighter retrieval
trajectories and thus better accuracy. The recall onset is clearly time-stamped by the
behavioral design (silence period ends after the last digit presentation + retention
interval).

**Feasibility:** Recall onset timing requires careful reconstruction from the behavioral
design (retention interval duration is defined in the task protocol). Medium effort.

---

## 6. Sequence-level manifold curvature and path length (`trajectory_summary`)

**What:** For each sequence (one listen or mem trial), compute the path length and
curvature of the MNPS trajectory (m, d, e) across windows within the trial. Output:
`path_length`, `mean_curvature`, `max_curvature`, `endpoint_distance_from_start`,
`trajectory_efficiency` (endpoint distance / path length).

**Why:** EAP predicts that high-anchor states produce straighter, more efficient manifold
trajectories during encoding (the system follows the gradient of the EAP landscape more
directly). Low-anchor states would produce more wandering trajectories. This metric is not
captured by any existing block-native summary.

**Feasibility:** Requires concatenating per-window MNPS values across a trial. Path length
and curvature are straightforward geometric operations. Low effort once block-native data
is available.

---

## 7. Network-to-network Jacobian coupling time series (`jacobian_offdiag_timeseries`)

**What:** For each block-native window, output the Frobenius norm of the off-diagonal
blocks of the 4-network × 4-network Jacobian: frontal↔temporal, frontal↔parietal,
central↔temporal, etc. This extends the current 4-network diagonal summary.

**Why:** Current R5 shows that within-network Jacobian norm decreases uniformly during
task. The between-network coupling (off-diagonal) may show a different pattern: some
pairs may *increase* coupling during task (task-relevant integration) while others
decrease (irrelevant disengagement). EAP predicts that anchor state modulates the
*specificity* of inter-network coupling, not just its overall magnitude.

**Feasibility:** The regional Jacobian computation in the existing pipeline already
partitions by network; outputting off-diagonal blocks is a minor extension. Low effort.

---

## 8. Autonomic complexity index (`sample_entropy_hrv`)

**What:** Compute sample entropy and/or DFA (detrended fluctuation analysis) α exponent
from the RR interval series per block (minimum 30 beats, ~30 s), and add as block-native
columns `hrv_sampen` and `hrv_dfa_alpha`.

**Why:** Heart rate variability complexity (entropy, scaling exponents) is a distinct
dimension from HRV power (vagal/sympathetic index). Multiscale entropy of HRV has been
linked to cognitive reserve, attentional flexibility, and emotional regulation —
all EAP-relevant constructs. Current HRV v0.1 uses SDNN, RMSSD, LF/HF power; complexity
measures would add a fractal/nonlinear anchor dimension.

**Feasibility:** Requires ≥30 RR intervals per window, limiting resolution to ~30-second
blocks. Compatible with the existing `anchor_surface.py` pipeline. Medium effort; several
Python libraries implement these metrics (`antropy`, `nolds`).

---

## 9. Eye-tracking gaze stability and fixation density (`gaze_anchor`)

**What:** For subjects with eye-tracking during eyes-open rest or listen segments, compute
per-window gaze dispersion (standard deviation of gaze position), fixation rate, and
blink rate. Add as block-native columns `gaze_dispersion`, `fixation_rate`, `blink_rate`.

**Why:** In the listen condition, gaze stability is a plausible somatic anchor: stable gaze
is associated with focused parasympathetic engagement. Blink rate is inversely correlated
with dopamine tone and attentional arousal. These metrics would extend the anchor surface
beyond cardiovascular signals to oculomotor and dopaminergic components.

**Feasibility:** 64 of 86 subjects have pupillometry/eye-tracking. ds003838's pupil data
includes gaze position and blink markers. Low effort to compute per-window summaries.

---

## 10. Multimodal anchor phase alignment (`cardiac_pupil_coherence`)

**What:** For each block-native window, compute coherence between cardiac RR intervals and
pupil diameter fluctuations in the 0.05–0.15 Hz band (low-frequency, `~10 s` cycles).
Output `rr_pupil_coherence` and `rr_pupil_phase_lag` as block-native columns.

**Why:** If anchor state is a genuine interoceptive signal rather than a superposition of
independent modalities, cardiac and pupil signals should co-oscillate with coherent phase
at the timescale where both are sensitive to autonomic state. High cardiac-pupil coherence
would indicate a "true" embodied anchor window; low coherence would indicate modal
discordance. This would directly address whether the `anchor_quality` field captures
true multimodal coherence or merely availability.

**Feasibility:** Requires both ECG and pupillometry in the same subject (~60 of 86).
Coherence in a 30 s window at 0.05–0.15 Hz is at the edge of spectral resolution; would
require at least 60 s windows or Welch averaging across trials. Medium–high effort.

---

## Summary table

| # | Field | Effort | EAP relevance | Timescale | New signal |
|---|-------|--------|--------------|-----------|------------|
| 1 | R-peak event-locked MNPS | Low | **Critical** — cycle-level EAP | ~1 Hz | ECG |
| 2 | Per-digit stimulus event-locked | Med | **High** — serial position | ~2 s SOA | EEG timing |
| 3 | Respiratory phase (EDR) | Med | **Medium** — slow anchor layer | ~0.25 Hz | ECG |
| 4 | Pupil dilation events + block columns | Med | **High** — LC-NE pathway | ~1–4 s | Pupil |
| 5 | Recall-onset event-locked | Med | **High** — retrieval anchor | ~4 s pre-recall | Behavioral |
| 6 | Trajectory curvature per trial | Low | **Medium** — path efficiency | trial-level | MNPS |
| 7 | Inter-network Jacobian off-diagonal | Low | **Medium** — specificity | window-level | MNDM |
| 8 | HRV complexity (SampEn, DFA) | Med | **Medium** — nonlinear anchor | ~30 s | ECG |
| 9 | Gaze stability / blink rate | Low | **Low-medium** — oculomotor anchor | window-level | Eye-track |
| 10 | Cardiac-pupil coherence | High | **Conceptual** — anchor quality | ~30 s | ECG + Pupil |

**Top priority for EAP validation:** Items 1 (R-peak aligned MNPS) and 5 (recall-onset
event-locked) directly test the two strongest untested EAP predictions at timescales not
yet covered.
