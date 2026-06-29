# 136 — 2026-06-26: P1–P5 Validation Complete; Readiness 0.77

## Session context

Continuing from diary 135 (P2–P4 engineering implementation).
This session ran the full validation notebook after refreshed H5s and fixed a notebook
`window_robustness` regression that temporarily pulled the score down.

---

## What was done

### Notebook fix: stream-output JSON validation error

The executed notebook (`ds003645_meg_validation_package.ipynb`) had 17 stream output cells
missing the required `name` field. Fixed by patching the notebook JSON directly
(`"name": "stdout"` added to all affected outputs).

### nbconvert path fix

`nbconvert` was concatenating a relative `--output` path on top of the working directory,
producing doubled paths. Fixed by using absolute paths for both input and output notebooks.

### notebook run: exit 0

The validation notebook executed successfully (exit 0, ~56 seconds) and produced
`ds003645_meg_validation_package_executed.ipynb`.

### window_robustness regression (identified and fixed)

**Root cause**: Cell 29 of the validation notebook hardcoded `scores['window_robustness'] = 0.0`
and wrote this to `meg_readiness_score.json`, overwriting the previously validated 1.0
(all three window sizes 8s/4s/2s showed face/scrambled centroid distance above null,
from `d_window_robustness.json`).

**Fix applied**:
- Cell 29 now loads `d_window_robustness.json` if available instead of hardcoding 0.0:
  ```python
  try:
      _d_wr = json.load(open(SAVE_DIR / 'd_window_robustness.json'))
      scores['window_robustness'] = float(_d_wr.get('window_robustness', 0.0))
  except (FileNotFoundError, KeyError):
      scores['window_robustness'] = 0.0  # pending D-tests
  ```
- `meg_readiness_score.json` patched to restore `window_robustness = 1.0` and recompute
  with final weights.

---

## Final readiness score: 0.7679

| Component                    | Score  | Weight |
|------------------------------|--------|--------|
| contract_pass_rate           | 1.0000 | 0.10   |
| feature_completeness         | 1.0000 | 0.15   |
| null_separation              | 0.3400 | 0.15   |
| event_response_agreement     | 0.5000 | 0.20   |
| mag_grad_stability           | 0.6694 | 0.10   |
| window_robustness            | 1.0000 | 0.15   |
| jacobian_validity            | 1.0000 | 0.15   |

**Weighted score: 0.7679 — "USABLE - minor fixes before scaling"**

### Science lead threshold bands (from diary 135)
```
>= 0.80  production-ready for full ds003645 scientific run
0.70–0.79 usable for pilot expansion, not final interpretation  ← WE ARE HERE
0.60–0.69 engineering-valid but scientifically provisional
< 0.60   do not scale
```

---

## P1–P5 status summary

| Priority | Description | Status |
|----------|-------------|--------|
| P1 | Real meg_sample_entropy + pilot rerun | COMPLETE |
| P2 | Row provenance (row_source/has_meg/raw_file) in H5 | COMPLETE |
| P3 | features_projection_z export surface | COMPLETE |
| P4 | --force-features CLI + cache metadata in JSONs | COMPLETE |
| P5 | Full validation notebook on refreshed H5s | COMPLETE |

---

## C1 at 4s (D2 section, from Cell 31)

```
sub  run  obs_cosine  null_mean  p_vs_null  n_face  n_scr
  2    1    0.2163    0.0319       0.272      280    116
  2    2    0.0672    0.0245       0.480      272    110
  2    3    0.1800   -0.0199       0.330      302    104
  2    4    0.5355    0.0252       0.100      286    108
  2    5   -0.3739    0.0228       0.830      296    104
```

Fraction obs > null_mean: **0.50** (3 of 5 runs for sub-002).

Run 4 shows the strongest alignment (obs=0.54, p=0.10 one-tailed). Run 5 is a clear
outlier (negative cosine). This is consistent with the science lead's note that
"per-run C1 is too noisy" — subject-level aggregation is the correct next target.

---

## D-test window robustness (from d_window_robustness.json)

| Window | Centroid dist | p-value |
|--------|---------------|---------|
| 8s     | 0.2529        | 0.000   |
| 4s     | 0.1837        | 0.000   |
| 2s     | 0.0870        | 0.014   |

Window robustness = **1.0** (all three window sizes above null).

---

## H5 structure (verified)

From `verify_h5_structure.py` run on the refreshed pilot H5:
```
[PASS] row_source: set=121 fif=121
       schema: mndm.row_source.v1
[PASS] features_projection_z: meg_delta std=1.4433 (expected >0.5)
       export_transform: projection_z
```

Both P2 (row provenance) and P3 (projection_z) are confirmed in the H5 contract.

---

## Mandatory gate status (science lead criteria)

1. H5 contract pass = 1.0 ✓
2. Feature completeness = 1.0 ✓
3. Row provenance implemented (row_source/has_meg/raw_file in H5) ✓
4. Real SampEn/e_m rerun completed for pilot ✓ (from P1)
5. No stale intermediate cache risk ✓ (--force-features + cache meta)
6. 4s C1/C2 subject-level analysis completed ✗ (run-level only, one subject)
7. Transform-aware feature export fixed ✓ (features_projection_z)

**Gates 1–5 and 7 are green. Gate 6 is the remaining gap.**

---

## Next steps

Immediate:
1. **Subject-level C1/C2 aggregation** — aggregate across runs per subject, then
   across subjects, to get a stable event-response cosine signal (Gate 6).
2. **Scale to all 18 subjects** — only after readiness >= 0.80 or all gates pass.
3. **Freeze MEG config** — only after all-subject rerun and final validation.

Optional refinements:
- MAG/GRAD stability gap (0.67): investigate which features drive instability.
- null_separation (0.34): review E2 temporal-shift null setup for edge cases.
- e_m independence audit from P1 audit table.
