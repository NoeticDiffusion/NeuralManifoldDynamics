# coords_9d duplicate cohort scan

## Goal

Quantify how often exact `coords_9d` duplicate subcoordinates occur at cohort scale, to determine whether the issue is a rare edge case or a recurring QC pattern.

## Data sources

- Primary source: full i-care cohort run
  `E:/Science_Datasets/physionet/processed/physionet_icare_2_1/neuralmanifolddynamics_physionet_icare_2_1_20260527_151403/run_errors.json`
- Comparison runs:
  - `20260526_164042`
  - `20260527_080207`
  - `20260527_101007_NeruroComBat_cleaned`

## Main findings

- In the latest broad cohort run (`20260527_151403`):
  - `88 / 3501` groupings hit the duplicate-`coords_9d` failure (`2.51%`)
  - `27 / 161` subjects had at least one affected run (`16.8%`)
  - `88 / 88` logged run errors were caused by duplicate `coords_9d` columns
- Follow-up completed intermediate-level scan across all available i-care EEG intermediates:
  - `171 / 3579` run-level intermediate files showed exact duplicate `coords_9d` subcoordinates (`4.78%`)
  - `54` subjects were affected at least once
  - `0` scan errors occurred in the completed threaded scan
- The dominant collapse pattern was overwhelmingly:
  - `m_e <- m_a` (`88` times in the failure log; `168` times in the completed intermediate scan)
- A smaller subset (`5` runs) showed broader multi-pair collapse:
  - `m_e <- m_a`
  - `d_n <- m_a`
  - `e_m <- m_a`
- The intermediate scan also surfaced a broader collapse regime where additional subcoordinates such as
  `m_o`, `d_l`, `d_s`, `e_s`, and sometimes `e_e` also became exact duplicates of `m_a`.

## Subject concentration

The issue is concentrated rather than uniform.

Top affected subjects in the latest broad run:

- `sub-0529`: `22` runs
- `sub-0320`: `11` runs
- `sub-0443`: `10` runs
- `sub-0641`: `8` runs
- `sub-0977`: `6` runs

## Cross-run comparison

- `20260526_164042`: `381 / 3501` duplicate failures (`10.88%`)
- `20260527_080207`: `88 / 3501` (`2.51%`)
- `20260527_101007_NeruroComBat_cleaned`: `380 / 3501` (`10.85%`)
- `20260527_151403`: `88 / 3501` (`2.51%`)

This suggests the phenomenon is recurring and materially affected by run configuration / preprocessing regime, not just by one isolated subject.

## Deliverable

- Added a canvas summary:
  `C:/Users/Robin/.cursor/projects/h-SourceRepo2-NeuralManifoldDynamics/canvases/coords9d-duplicate-cohort-scan.canvas.tsx`

## Interpretation

This is not a single-subject anomaly. It is also not ubiquitous. The practical picture is:

- low prevalence at grouping level,
- somewhat higher prevalence at intermediate/run level than the failure log alone suggests,
- moderate prevalence at subject level,
- very strong concentration in a small number of subjects,
- and a highly stereotyped collapse signature dominated by `m_e <- m_a`.
