# Part 1 ComBat run comparison

Date: 2026-06-06

## Question

Compare the new part 1 `0-12h` ComBat summarize run against the older ComBat run, focusing on whether duplicate `coords_9d` failures and 9D falsification / low-rank issues became better or worse.

Compared runs:

- Old: `G:/Science_Datasets_longtime_storage/processed/physionet_part1_0_12h_regional_combat/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260601_131927`
- New: `E:/Science_Datasets/physionet/processed/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260605_202236`

## Findings

### 1. Duplicate `coords_9d` behavior is operationally much better

- Old run: `380` hard duplicate failures in `run_errors.json`, affecting `52` subjects.
- New run: `0` hard duplicate failures, but `380` runs now export with duplicate diagnostics in `geometry_contract.coords_9d`.
- The duplicate pair signature is effectively unchanged:
  - `m_e <- m_a`: `355`
  - `d_n <- m_a`: `122`
  - `e_m <- m_a`: `5`
  - `d_n <- m_e`: `5`

Interpretation: the duplicate-tolerance fix worked as intended. It removed crash behavior, but it did not remove the underlying duplicate geometry pattern.

### 2. Output coverage improved substantially

- Old run manifest counts: `3104` H5 / summary / QC outputs.
- New run manifest counts: `3484` H5 / summary / QC outputs.
- Net gain: `+380` successful outputs (`+12.2%`).
- Missing run directories without `summary.json` dropped from `397` to `17`.

The `380` recovered outputs line up exactly with the old duplicate-failure count, strongly suggesting that duplicate tolerance explains almost the entire coverage improvement.

### 3. 9D falsification did not improve

Log-level comparison:

- Old run: `173` `CRITICAL WARNING: 9D MNPS falsified...`
- New run: `215`
- Delta: `+42` (`+24.3%`)

Regional CSV comparison using exported `strat9_falsified`:

- Old: `2055 / 24045` rows = `8.55%`
- New: `2153 / 24388` rows = `8.83%`
- Delta: `+0.28` percentage points (`+3.3%` relative)

The increase is broad-based across all seven regional networks rather than isolated to one network.

## Bottom line

- **Better:** duplicate `coords_9d` no longer causes hard pipeline failure in part 1.
- **Unchanged underneath:** the same duplicate-collapse structure is still present and now surfaces as degraded geometry instead of crashes.
- **Slightly worse:** exported 9D falsification / low-rank incidence increased modestly in the new run.

## Next useful step

Inspect the remaining `17` missing run directories and determine whether they are a separate skip path, empty-input condition, or another silent failure mode unrelated to duplicate tolerance.
