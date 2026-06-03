# ComBat pilot follow-up explanation

Date: 2026-05-27

## Question

Why did the part1 ComBat pilot reduce site signal without improving the downstream CPC1-vs-CPC5 pilot?

## New artifacts

- Script:
  - `articles/CLINICAL/comatose_ndt/src/combat_harmonization_followup.py`
- Shareable markdown:
  - `articles/CLINICAL/comatose_ndt/NeuroCombat_harmonization/047_20260527_icare_part1_combat_followup_analysis.md`
- Follow-up CSV outputs:
  - `articles/CLINICAL/comatose_ndt/NeuroCombat_harmonization/results/combat_pilot_20260527/combat_followup_*.csv`

## Analyses added

1. Hospital x CPC imbalance in the paired CPC1/CPC5 cohort.
2. Family-wise site dependence changes using pilot `eta^2` values.
3. Family-wise within-site outcome signal after hospital-wise z-scoring.
4. Family-wise paired subject perturbation magnitudes.
5. Top feature-level site reductions and top feature-level outcome gains/losses.

## Main conclusions

- Hospital composition is not neutral in the paired CPC cohort:
  - `chi2 = 11.840`
  - `p = 0.0186`
  - `Cramer's V = 0.292`
- This supports the explanation that some baseline predictive signal was site-correlated proxy signal.
- ComBat clearly de-batched Hjorth complexity and entropy-style features.
- Beta-family features remained problematic: several became more site-linked after ComBat and some of the within-site gains concentrated there.
- Delta-family features were a clear loser in the follow-up:
  - site dependence decreased,
  - but within-site CPC separation also dropped sharply.

## Working interpretation

The behavior is no longer especially paradoxical:

- ComBat removed real hospital-linked nuisance structure,
- but the observed cohort also contains hospital-linked prognostic imbalance,
- so the transform can reduce hospital predictability and CPC predictability at the same time.

Under the current configuration, this looks more like a successful de-batching transform than a successful prognostic enhancement transform.
