Yes — the right target is **not high EEG–MEG correlation**, but **convergent measurement behavior**.

I would define the validation goal like this:

> **MEG should reproduce the same NMD claim surface as EEG when the task, timing, labels, and coordinate contract are shared — even if the raw trajectories and absolute coordinates differ.**

That fits NMD’s own contract: fixed export naming, deterministic preprocessing, coverage handling, 3D/9D coordinates, optional Jacobians, and manifest-level provenance are the core claims; the chart labels are operational, not biologically unique identities.  The 9D layer is especially important here because it is designed to expose redistributions that can be hidden in coarse 3D composites. 

Below is the test package I would give to the repo-LLM / implementer.

---

# MEG–EEG NMD Validation Package

## Success criterion

MEG does **not** need to correlate highly with EEG window-by-window.

MEG should pass if it shows:

1. same data integrity,
2. same export contract,
3. same event-label alignment,
4. same broad task-effect direction where physiologically plausible,
5. same subcoordinate family involvement,
6. stronger-than-null cross-modal agreement,
7. robustness across window sizes and MEG sensor types.

The main validation objects should be:

* **feature-level agreement**
* **event-response vector agreement**
* **9D subcoordinate sign agreement**
* **family-level m/d/e agreement**
* **null separation**
* **window-size sensitivity**
* **magnetometer vs gradiometer stability**
* **MEG-specific bug audits**

---

# A. Sequential gate tests

These should run first. If one fails, stop and fix before moving on.

## A0 · Raw file and modality audit

**Purpose:** verify that ds003645 files actually contain usable EEG, magnetometer, and gradiometer channels.

For each subject/run, export:

```text
subject
run
sfreq
n_eeg
n_mag
n_grad
n_eog
n_ecg
duration_sec
n_events
event_file_found
raw_file_found
```

**Pass:**

* EEG, MAG, and GRAD present in FIF.
* Event TSV exists for every run.
* Run duration and event counts are within expected range.
* No silent modality drop.

**Fail interpretation:**

* If EEG missing from FIF, use `.set` for EEG but record source divergence.
* If MAG/GRAD missing, mark MEG branch incomplete, not failed globally.

---

## A1 · Timebase and event-grid audit

**Purpose:** ensure EEG, MEG, events, and NMD windows live on the same temporal grid.

For each run:

```text
min(window_start)
max(window_end)
event_onset_min
event_onset_max
n_windows
n_labeled_windows
n_face
n_scrambled
n_mixed
n_no_stim
```

**Pass:**

* Event onsets fall inside raw recording duration.
* Window labels reproduce expected face:scrambled imbalance.
* Mixed/no-stim rate is explainable.
* Window joins do not drop one modality.

This test directly protects against your current suspected bug: MEG values present in H5 but missing after DataFrame construction.

---

## A2 · EEG FIF vs EEG .set consistency

You already have this and it passed.

**Current result:** bandpower r≈0.88 across delta/theta/alpha/beta/gamma.

Keep it as a permanent regression test.

**Pass threshold:**

```text
mean_r_by_band >= 0.80
min_run_band_r >= 0.65
```

**Interpretation:**

This validates EEG ingestion and `.fdt` repair. It does **not** validate MEG yet.

---

## A3 · MEG feature completeness and lineage audit

**Purpose:** fix the missing MEG band contrast CSV issue.

For every MEG feature expected in the mapping:

```text
meg_delta
meg_theta
meg_alpha
meg_beta
meg_gamma
meg_beta_alpha
meg_alpha_theta
meg_hjorth_mobility
meg_hjorth_complexity
meg_permutation_entropy
meg_highfreq_power_30_45
```

Check each stage:

```text
raw feature table
all_epochs_features.csv
labeled_manifold_epochs.csv
face/scrambled subset
band_contrast input frame
band_contrast output CSV
```

**Pass:**

* Non-NaN count is stable across stages except for intentional QC drops.
* Column naming is consistent.
* MAG/GRAD/combined feature names are not accidentally filtered out.

**Most likely bug class:** column mismatch or modality filter, not true NaNs.

---

## A4 · NMD HDF5 contract test

**Purpose:** verify that MEG produces the same measurement object as EEG.

Required paths:

```text
/mnps_3d
/mnps_3d_dot
/coords_9d/values
/coords_9d/names
/jacobian/J_hat
/jacobian_9D/J_hat
/window_start
/window_end
/features_raw
/features_robust_z
```

The NMD contract explicitly treats 3D MNPS, 9D coordinates, and Jacobian exports as separate measurement objects; MEG should inherit that structure rather than create a parallel ad hoc output. 

**Pass:**

* EEG and MEG export the same required paths.
* Axis order is explicit.
* Feature anchors/provenance are written.
* Coverage fields are present.

---

# B. MEG-internal validation tests

These can run in parallel after A-tests pass.

## B1 · MAG vs GRAD feature agreement

**Purpose:** determine whether MEG features are stable across sensor families.

Compute MAG and GRAD features separately:

```text
meg_mag_alpha
meg_grad_alpha
meg_alpha_combined
...
```

Metrics:

```text
corr(mag_feature, grad_feature)
sign agreement face-scrambled
event-response cosine
feature variance ratio
```

**Pass:**

Not necessarily high correlation, but:

```text
same face/scrambled direction in >= 60–70% of core features
combined feature not dominated by a broken sensor branch
MAG and GRAD both beat temporal null for gamma or task-sensitive features
```

**Fail interpretation:**

* MAG and GRAD disagree globally → use separate MEG branches in 9D.
* One branch noisy → exclude or down-weight in combined MEG.

---

## B2 · MEG unit/scale invariance test

**Purpose:** ensure magnetometers and gradiometers are never mixed before robust scaling.

Run two variants:

```text
Variant 1: combine raw MAG+GRAD then robust-z
Variant 2: robust-z MAG and GRAD separately, then combine
```

**Expected:** Variant 2 should be more stable.

**Pass:**

* Variant 2 reduces feature dominance by one sensor type.
* Face/scrambled direction is preserved.
* 9D coverage improves or remains stable.

---

## B3 · MEG spectral sanity test

For each run and sensor branch:

```text
delta/theta/alpha/beta/gamma distributions
bandpower covariance matrix
first PC variance explained
artifact-sensitive high-frequency outliers
```

**Pass:**

* No feature is constant.
* Gamma/high-frequency is not entirely driven by a few extreme windows.
* Broadband dominance is quantified, not ignored.

This matters because your pilot already suggests MEG features may be more broadband-correlated than EEG.

---

# C. Cross-modal “same thing” tests

These are the core tests. They should not demand high raw correlation.

## C1 · Event-response vector agreement

This should replace Procrustes as the main cross-modal validation.

For each subject/run/modality:

[
\Delta x^{q}_{face-scrambled}
=============================

## \bar{x}^{q}_{face}

\bar{x}^{q}_{scrambled}
]

where (q \in {EEG, MEG}).

Compute:

```text
cosine(ΔEEG_3D, ΔMEG_3D)
cosine(ΔEEG_9D, ΔMEG_9D)
cosine by family: m-family, d-family, e-family
sign agreement by subcoordinate
```

**Pass:**

```text
mean cosine observed > temporal/event-label null
sign agreement > 50% null expectation
e-family/gamma agreement strongest if face task is mainly gamma-sensitive
```

**Why this is better:** it asks whether EEG and MEG respond to the task in the same **direction**, not whether their whole trajectories overlap.

---

## C2 · Subcoordinate family agreement

Compute face–scrambled effects separately:

```text
m_a, m_e, m_o
d_n, d_l, d_s
e_e, e_s, e_m
```

Then summarize family-level signs:

```text
m_family_sign = sign(mean Δ over m_a/m_e/m_o)
d_family_sign = sign(mean Δ over d_n/d_l/d_s)
e_family_sign = sign(mean Δ over e_e/e_s/e_m)
```

**Pass:**

* EEG and MEG agree at family level in at least one interpretable family.
* For face task, `e_m` / gamma-family agreement is allowed to be the primary positive result.
* 3D can remain weak if 9D shows interpretable redistribution.

This is aligned with the NMD rationale: 3D can mask subcoordinate redistributions, while 9D is explicitly intended to expose them. 

---

## C3 · Temporal co-variation with lag tolerance

For each feature/subcoordinate, compute EEG–MEG correlations across lags:

```text
lag = -2s, -1s, 0s, +1s, +2s
```

Metrics:

```text
max_corr_within_lag_window
best_lag
null_corr_distribution
p_vs_temporal_shift_null
```

**Pass:**

* Gamma / `e_m` beats temporal null.
* Best lag is plausible and not random across runs.
* Correlations do not need to be high; they need to be above null.

---

## C4 · Rank-order condition agreement

For each run, rank conditions or event bins by each metric:

```text
face
scrambled
mixed
no_stim
```

Ask whether EEG and MEG rank them similarly.

Metrics:

```text
Spearman rank agreement
Kendall tau
sign agreement of pairwise contrasts
```

**Pass:**

* EEG and MEG agree that face > scrambled for gamma/`e_m`, if that survives bug fix.
* Family-level rank agreement beats label-shuffle null.

This is useful when absolute scales differ strongly.

---

# D. Window-size multiverse

This is essential.

Run the same tests at multiple temporal scales:

```text
8s / 4s step      current slow-state baseline
4s / 1s step      intermediate
2s / 0.5s step    event-sensitive spectral surface
1s / 0.25s step   optional, only if features remain stable
event-locked      -0.2 to 0.8s or 0 to 1.0s
```

For each scale, compute:

```text
face-scrambled centroid distance
9D Δ vector
EEG–MEG event-response cosine
gamma / e_m effect
temporal-null p
coverage
Jacobian validity rate
```

**Expected pattern:**

* 8s: weak face/scrambled separation, stable slow geometry.
* 2–4s: stronger event-response agreement.
* 1s/event-locked: stronger face/gamma signal, but Jacobian may become unstable.
* MNJ should only be interpreted at scales with enough windows and support.

This is consistent with the broader NMD discipline that windowing, feature mapping, and normalization are part of the logged measurement contract, not incidental preprocessing. 

---

# E. Null and falsification tests

These should run automatically for every major comparison.

## E1 · Event-label shuffle

Shuffle face/scrambled labels within subject/run.

Expected:

```text
observed Δ distance > shuffled Δ distance
observed EEG–MEG Δ cosine > shuffled cosine
```

## E2 · Temporal circular shift

Circularly shift MEG windows relative to EEG within run.

Expected:

```text
observed EEG–MEG feature agreement > shifted agreement
```

This is already partly validated by your gamma-null result.

## E3 · Wrong-run pairing

Pair EEG run 1 with MEG run 2 from the same subject, or with same run from another subject.

Expected:

```text
true-pair agreement > wrong-run/wrong-subject agreement
```

This is one of the strongest tests for “same thing” without requiring high correlation.

## E4 · Modality-label permutation

Randomly swap modality labels after feature extraction.

Expected:

```text
true EEG–MEG mapping beats permuted mapping
```

## E5 · Feature-family ablation

Drop one family at a time:

```text
without low-frequency
without alpha/beta ratios
without gamma/high-frequency
without entropy/Hjorth
```

Expected:

* If face signal depends on `e_m`, gamma ablation should reduce face/scrambled agreement.
* If low-frequency dominates global trajectory but not task response, removing it may improve event sensitivity.

---

# F. MEG mapping improvement tests

These are extension tests designed to improve the MEG path.

## F1 · Three MEG mappings

Run three MEG variants in parallel:

```text
MEG-shadow-v1:
  eeg_* -> meg_* direct feature substitution

MEG-sensor-split-v1:
  separate mag and grad 9D, then combine at coordinate level

MEG-family-calibrated-v1:
  same 9D families, but MEG-specific feature weights learned only from unsupervised stability criteria
```

Do **not** optimize against face/scrambled labels yet.

Compare:

```text
coverage
null separation
event-response cosine with EEG
face/scrambled effect size
MAG/GRAD agreement
Jacobian validity rate
```

**Recommended winner criterion:**

Choose the mapping that maximizes:

```text
null_separation_score
+ event_response_agreement
+ coverage
- instability_penalty
```

not the one that maximizes face classification.

---

## F2 · Sensor-region MEG groups

Create rough MEG regional groups analogous to EEG channel groups:

```text
left_temporal
right_temporal
frontal
central
parietal_occipital
```

Then compute regional MEG MNPS.

**Pass:**

* Temporal groups should show stronger face-related effects than irrelevant groups.
* Regional MEG effects should not be globally identical everywhere.

This test is important because a global MEG average may erase the spatial structure of face processing.

---

## F3 · Gamma proxy audit

Because current `e_m` may fallback to `meg_highfreq_power_30_45`, audit it carefully:

```text
e_m_source
meg_highfreq_power_30_45
meg_gamma
artifact flags
EOG correlation
muscle/noise proxy correlation
face-scrambled Δ
temporal null p
```

**Pass:**

* `e_m` face effect survives after excluding high-artifact windows.
* `e_m` is not just EOG/muscle contamination.
* MEG and EEG show same direction in gamma-like proxy.

The NMD limitation text already flags `e_m` fallback as useful but weaker for strict comparability, so this must be explicit in validation. 

---

# G. MNJ and reachability tests

Do these only after coordinate tests pass.

## G1 · Jacobian validity audit

For each modality/window scale:

```text
n_valid_jacobians
condition_number_median
spectral_radius_median
frobenius_norm_median
rotation_norm_median
anisotropy_valid_rate
```

**Pass:**

* Enough valid Jacobians per run.
* Condition numbers not pathological.
* Results stable under k-neighborhood sensitivity.

If 2s/event windows do not support MNJ, do not force it.

---

## G2 · MNJ direction agreement

For face vs scrambled, compare:

```text
rotation_norm
spectral_radius
frobenius_norm
trace
anisotropy
```

Use direction/sign agreement rather than raw magnitude.

**Pass:**

* EEG and MEG agree in direction for at least one MNJ family above null.
* If MNJ fails but 9D response vectors pass, the MEG ingest is still useful.

---

## G3 · Reachability extension

Once MEG MNJ is stable, compute short-horizon reachability:

```text
tube_log_det
tube_d_eff
tube_anisotropy
Q_ratio
capture_gate if available
```

**Pass:**

* MEG reachability is finite and stable.
* Face/scrambled or event-bin differences beat temporal null.
* EEG and MEG agree directionally in one reachability endpoint.

Reachability should be treated as local dynamical availability, not occupancy or direct cognition. That boundary is already part of your NMD/NDT method discipline. 

---

# H. Recommended execution graph

Run this as a staged pipeline:

```text
Stage 0: Integrity
  A0, A1, A2, A3, A4

Stage 1: MEG-internal
  B1, B2, B3

Stage 2: Cross-modal core
  C1, C2, C3, C4

Stage 3: Window multiverse
  D at 8s, 4s, 2s, 1s/event-locked

Stage 4: Nulls
  E1, E2, E3, E4, E5

Stage 5: MEG mapping improvement
  F1, F2, F3

Stage 6: Dynamics
  G1, G2, G3
```

Parallelizable:

```text
B1/B2/B3
C1/C2/C3/C4
all window sizes in D
all nulls in E
MEG mapping variants in F1
```

Sequential dependencies:

```text
A-tests before everything
D before deciding final event-sensitive mapping
F before scaling to all subjects
G only after stable coordinate layer
```

---

# Concrete pass/fail dashboard

I would make one summary table per run and one aggregate table.

```text
test_id
subject
run
modality_pair
window_sec
step_sec
metric
observed
null_mean
null_sd
p_value
effect_size
pass
failure_reason
```

Then compute an aggregate “MEG readiness score”:

```text
MEG readiness = 
  0.20 * contract_pass_rate
+ 0.20 * feature_completeness
+ 0.20 * null_separation
+ 0.15 * event_response_agreement
+ 0.10 * mag_grad_stability
+ 0.10 * window_robustness
+ 0.05 * jacobian_validity
```

Interpretation:

```text
>= 0.80  ready to scale beyond 5 subjects
0.60–0.79 usable but needs targeted fixes
0.40–0.59 ingest works, mapping unstable
< 0.40   do not scale yet
```

---

# What I would prioritize immediately

For the next implementation session, I would run only this subset:

1. **A3: MEG feature completeness / missing band contrast bug**
2. **B1: MAG vs GRAD feature agreement**
3. **C1: EEG–MEG event-response vector agreement**
4. **D: 8s vs 4s vs 2s window comparison**
5. **E2/E3: temporal shift and wrong-run nulls**
6. **F3: `e_m` / gamma proxy audit**

That will tell you whether MEG needs only a small export fix, or whether the mapping itself needs redesign.

My main recommendation: make **event-response vector agreement in 9D** the central test. Not Procrustes, not raw correlation, not 3D centroid distance. That is the cleanest way to express your goal: **MEG and EEG do not have to be the same measurement, but they should point to the same task-sensitive structure under the NMD contract.**
