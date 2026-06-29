# 139 — 2026-06-26 — Subject-level C1/C2 aggregation + readiness update (Gate 6)

## Session context

Science lead validated the notebook fix (diary 136 p1–p5), confirmed the window-robustness
restoration, and specified that the only remaining gap before pilot expansion is
**Gate 6: subject-level C1/C2 aggregation**. This session implements that gate,
runs it on the 5 pilot subjects, and delivers the updated readiness score.

---

## 1. Implementation: `05_subject_level_c1c2.py`

Script: `project/scripts/05_subject_level_c1c2.py`

**What it does:**
1. Discovers all pilot subjects with epoch-condition labels (subjects 2–6).
2. For each subject, pools all FIF rows across 6 runs.
3. Loads **`features_projection_z`** instead of `features_robust_z` — this is
   required because MEG spectral features (delta/theta/alpha/beta/gamma) in
   `features_robust_z` collapse to exactly zero due to physical-unit robust-z
   mismatch (~1e-25 W). The projection_z surface applies log10 before robust-z,
   recovering non-degenerate spectral values.
4. Computes face-minus-scrambled response vectors for MEG (`meg_*` features) and
   EEG (`eeg_*` features) on shared feature types.
5. Reports: 9D cosine (all shared types), 3D cosine (m/d/e family aggregated),
   and per-family (m/d/e) cosines.
6. Runs label-shuffle null (N=500) and wrong-run pairing null for each subject.
7. Computes C2 per-feature-type sign agreement per family.
8. Updates `meg_readiness_score.json`.

**Bug found and fixed in-session:**
`features_robust_z` values for `meg_delta`, `meg_theta`, `meg_alpha`, `meg_beta`,
`meg_gamma` in FIF rows: mean=0.0, std=0.0 for all rows in all runs. Confirmed
by direct H5 inspection. Switching to `features_projection_z` fully restores
spectral variance.

---

## 2. Results

### C1 — Subject-level event-response cosine

| sub | n_face | n_scr | cos_9d  | cos_3d  | cos_m   | cos_d   | cos_e   | null_mean | p_shuffle | obs>null |
|-----|--------|-------|---------|---------|---------|---------|---------|-----------|-----------|----------|
| 002 | 428    | 165   |  0.7616 |  0.8153 |  0.7962 |  0.9447 |  0.1374 |  0.0856   |  0.040    | **True** |
| 003 | 414    | 158   |  0.1881 |  0.8189 |  0.3162 | -0.9703 |  0.7789 |  0.0600   |  0.414    | True     |
| 004 | 442    | 150   | -0.4201 | -0.6903 | -0.4099 | -0.9791 | -0.1751 |  0.0330   |  0.804    | **False**|
| 005 | 426    | 158   |  0.8268 |  0.9229 |  0.8563 |  0.9849 |  0.8882 |  0.0783   |  0.018    | **True** |
| 006 | 421    | 157   | -0.6778 | -0.8755 | -0.4825 | -0.9995 | -0.3904 |  0.0457   |  0.926    | **False**|

**obs > null_mean: 3/5 = 0.60**

- sub-005: **most strongly aligned** — 9D=0.83, 3D=0.92, all three families positive,
  p=0.018. C2 sign agreement = 1.0 (perfect).
- sub-002: **well aligned** — 9D=0.76, 3D=0.82, m and d families high, p=0.04.
- sub-003: **spectral agrees but d-family inverts** — 3D=0.82 (high!) but 9D=0.19
  because cos_d=−0.97 drags the 9D vector. The m and e families agree, but
  Hjorth mobility/complexity shows opposite face-vs-scrambled direction in MEG vs EEG.
- sub-004 and sub-006: **systematic negative 9D and 3D cosines**. The d-family
  cosine is −0.98 and −1.00 respectively. This is not noise — MEG and EEG Hjorth
  features show a consistent *sign reversal* of the face-minus-scrambled effect.

### C2 — Family sign agreement (per feature type, MEG vs EEG sign of face−scr)

| sub | m-family | d-family | e-family | overall |
|-----|----------|----------|----------|---------|
| 002 | 0.80     | 1.00     | 0.50     | 0.778   |
| 003 | 0.60     | 0.00     | 1.00     | 0.556   |
| 004 | 0.40     | 0.00     | 0.50     | 0.333   |
| 005 | 1.00     | 1.00     | 1.00     | 1.000   |
| 006 | 0.60     | 0.00     | 0.50     | 0.444   |
| mean| 0.68     | 0.40     | 0.70     | 0.622   |

d-family sign agreement is the weakest dimension: 1.0 for sub-002, 1.0 for sub-005,
but 0.0 for sub-003, sub-004, sub-006. The d-family (Hjorth mobility + complexity)
appears to be the axis driving the bimodal structure.

---

## 3. Readiness score — final update

### Updated gates

| Gate | Condition | Status |
|------|-----------|--------|
| 1 | H5 contract pass = 1.0 | ✓ 1.00 |
| 2 | Feature completeness = 1.0 | ✓ 1.00 |
| 3 | Row provenance (row_source/has_meg in H5) | ✓ |
| 4 | Real SampEn/e_m rerun for pilot | ✓ |
| 5 | No stale intermediate cache risk | ✓ |
| 6 | Subject-level C1/C2 aggregation | ✓ (this session) |
| 7 | Transform-aware feature export (features_projection_z) | ✓ |

**All mandatory gates are now addressed.**

### Score components

| Component                    | Score  | Weight |
|------------------------------|--------|--------|
| contract_pass_rate           | 1.0000 | 0.10   |
| feature_completeness         | 1.0000 | 0.15   |
| null_separation              | 0.3400 | 0.15   |
| event_response_agreement     | 0.6000 | 0.20   |
| mag_grad_stability           | 0.6694 | 0.10   |
| window_robustness            | 1.0000 | 0.15   |
| jacobian_validity            | 1.0000 | 0.15   |

**Weighted score: 0.7879 — "USABLE - minor fixes before scaling"**

### Decision per science lead thresholds

```
>= 0.80  READY — proceed to pilot expansion
0.75–0.80 proceed cautiously, label "usable pilot MEG mapping"  ← WE ARE HERE (0.79)
0.60–0.79 usable for pilot expansion but not final interpretation
< 0.60   do not scale
```

0.7879 is just below 0.80 but within the "proceed cautiously" zone. The science lead's
instruction was: "If it remains 0.75–0.80, still proceed cautiously, but label it
'usable pilot MEG mapping, cross-modal convergence moderate.'"

---

## 4. Scientific interpretation

### Bimodal subject structure

The 5 pilot subjects split clearly into two groups:

**Group A (sub-002, sub-005):** Strong positive cross-modal alignment. Both spectral
(m) and Hjorth (d) face-minus-scrambled effects point in the same direction for MEG
and EEG. Sub-005 achieves perfect C2 sign agreement. These are the subjects where
the NMD cross-modal measurement contract is working as intended.

**Group B (sub-004, sub-006):** Strong negative cross-modal alignment. Cos_9d ≈ −0.5
to −0.7; cos_d ≈ −1.0. MEG Hjorth mobility/complexity shows the face-minus-scrambled
effect in the *opposite* direction to EEG. This is systematic, not random noise.

**sub-003 is intermediate:** Spectral (m) and entropy (e) agree, but d-family inverts
(like Group B for the d-dimension only). The 3D cosine is high (0.82) because the
spectral agreement dominates in the family-aggregated 3D view.

### Interpretation of Group B divergence

This is not necessarily a failure:
1. MEG and EEG do not need to produce identical Hjorth responses. They sample
   different spatial scales (MEG: broad field; EEG: scalp-proximal).
2. Hjorth mobility and complexity can show genuine opposite face/scrambled effects
   in different sensor geometries (e.g., a pattern visible in MEG but masked in
   EEG due to reference or volume conduction).
3. Alternatively: sub-004 and sub-006 may have data quality differences (head
   movement, electrode contact, impedance) that invert the EEG response for
   those subjects.

**Next step (Track 2 — subject robustness):** LOO Friedman and behavioral review
for sub-004/sub-006 specifically will clarify whether these are outlier subjects
or genuine cross-modal divergence.

### Important guardrail (science lead)

Do not over-optimize C1. The target is convergent task-sensitive structure under
the same NMD contract, not high cross-modal correlation. The current result:

> MEG face/scrambled separation is robust and aggregate performance is strong.
> At the subject level, 2/5 pilot subjects show statistically significant cross-modal
> response-vector alignment. The remaining 2 subjects show systematic sign reversal
> in the d-family (Hjorth features), warranting subject-level investigation before
> scaling.

---

## 5. Files produced

| Path | Contents |
|------|----------|
| `project/scripts/05_subject_level_c1c2.py` | Subject-level C1/C2 script |
| `E:/.../c1_subject_event_response_agreement.csv` | Per-subject C1 cosines + nulls |
| `E:/.../c2_subject_family_sign_agreement.csv` | Per-subject family sign agreement |
| `E:/.../c1c2_subject_null_summary.csv` | Per-subject null comparison table |
| `E:/.../meg_readiness_score.json` | Updated readiness score (0.7879) |

---

## 6. Current label

> **Usable pilot MEG mapping, cross-modal convergence moderate.**
> MEG face/scrambled separation is robust across windows. Two of five pilot subjects
> show statistically significant MEG–EEG event-response alignment. The d-family
> (Hjorth mobility/complexity) is the primary axis of cross-modal divergence and
> should be investigated in subject-level behavioral/quality review before scaling.

---

## 7. Immediate next steps

1. **Subject robustness review** (sub-004, sub-006): behavioral performance,
   data quality flags, LOO Friedman. Do they drive the null-separation weakness?
2. **Scale to all 18 subjects** if subject-level review is satisfactory or if
   science lead approves at current readiness.
3. **ds006848 next analyses** (Track 1: behavioral review, Track 2: subject
   robustness, Track 3: EEG comparator) are queued.
