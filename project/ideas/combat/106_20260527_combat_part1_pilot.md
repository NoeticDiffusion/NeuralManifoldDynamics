# ComBat part1 pilot on paired I-CARE H5 exports

Date: 2026-05-27

## Question

Does the new part1 ComBat-harmonized I-CARE export reduce site-linked structure and improve downstream CPC1-vs-CPC5 signal relative to the older paired part1 baseline export?

## Inputs

- Baseline root:
  - `data/raw/neuralmanifolddynamics_physionet_icare_2_1_20260523_173059_part1_141sub`
- ComBat root:
  - `data/raw/neuralmanifolddynamics_physionet_icare_2_1_20260526_164042_part1_ComBat`
- Analysis script added:
  - `articles/CLINICAL/comatose_ndt/src/combat_harmonization_pilot.py`
- Result bundle:
  - `articles/CLINICAL/comatose_ndt/NeuroCombat_harmonization/results/combat_pilot_20260527`

## What was run

Built a direct paired-H5 pilot rather than a full adapter rebuild.

Per readable paired run, the pilot extracted:

- subject / hospital / CPC metadata from `participant/row_json`
- median `features_raw` values across windows
- compact global dynamics summaries:
  - `mnps_rms`
  - `mnps_speed`
  - `mnj_fro`
  - `mnjdot_fro`

Then aggregated to subject level and evaluated:

1. site dependence by hospital
2. site prediction accuracy from subject summaries
3. CPC1-vs-CPC5 prediction under repeated CV
4. CPC1-vs-CPC5 prediction under leave-one-site-out

## Coverage

- common run directories scanned: `3501`
- paired readable H5 runs retained: `3066`
- runs skipped due to missing H5 in one side: `435`
- subject count per condition after aggregation: `159`
- CPC1/CPC5 subjects per condition: `139`

## Main findings

### 1. Raw feature site signal was reduced by ComBat

Subject-level raw-feature site structure decreased:

- median eta^2 by hospital:
  - baseline: `0.050`
  - ComBat: `0.028`
- p95 eta^2 by hospital:
  - baseline: `0.209`
  - ComBat: `0.110`

Raw-feature hospital prediction also dropped:

- balanced accuracy:
  - baseline: `0.591`
  - ComBat: `0.478`
- accuracy:
  - baseline: `0.656`
  - ComBat: `0.496`
- majority-class chance accuracy: `0.459`

Interpretation:
ComBat appears to remove a meaningful amount of site-linked structure from the raw feature surface.

### 2. Raw-feature CPC prediction did not improve in this pilot

Repeated CV on subject-level raw features:

- ROC AUC:
  - baseline: `0.748`
  - ComBat: `0.725`
- Brier:
  - baseline: `0.211`
  - ComBat: `0.230`

Leave-one-site-out on subject-level raw features:

- ROC AUC:
  - baseline: `0.755`
  - ComBat: `0.696`
- Brier:
  - baseline: `0.259`
  - ComBat: `0.318`

Interpretation:
The present pilot does not support the claim that ComBat improved CPC discrimination on the raw feature surface; if anything, the internal signal was slightly weaker in this paired comparison.

### 3. Compact dynamics summaries gave mixed results

For the 4 compact dynamics metrics:

- repeated-CV ROC AUC fell:
  - baseline: `0.664`
  - ComBat: `0.604`
- leave-one-site-out ROC AUC was roughly flat / slightly higher:
  - baseline: `0.613`
  - ComBat: `0.620`

But Brier under leave-one-site-out worsened:

- baseline: `0.238`
- ComBat: `0.269`

Interpretation:
There is no clean downstream clinical gain here. The tiny LOSO AUC increase on the 4-metric dynamics panel is too weak to outweigh the broader degradation in calibration-oriented scores.

### 4. Primary MNPS/MNJ tensors still shift materially

Across the `3066` readable paired runs:

- `mnps_rms`: median `+1.786%`, p95 abs `33.122%`
- `mnps_speed`: median `+1.206%`, p95 abs `30.281%`
- `mnj_fro`: median `+2.505%`, p95 abs `54.943%`
- `mnjdot_fro`: median `+2.560%`, p95 abs `56.575%`

This reproduces the earlier observation that ComBat is not a numerically negligible perturbation.

## Bottom line

Current internal evidence supports:

- **yes**: ComBat reduced site-linked structure in the raw feature surface
- **no clear evidence**: ComBat improved CPC1-vs-CPC5 prediction in this paired part1 pilot
- **possible but weak**: a very small LOSO AUC gain on the compact 4-metric dynamics panel

So the pilot supports the narrower statement:

> ComBat appears to de-batch the raw feature surface.

But it does **not** currently support the stronger statement:

> ComBat clearly improves downstream coma outcome signal in this part1 paired export.

## Follow-up ideas

1. Repeat the same pilot against the regional non-ComBat comparator if that paired export is available locally.
2. Run the same tests on patient-bin handoff outputs after a locked `timeline_analysis -> mnps_summary` rebuild, so the comparison sits closer to the paper-facing tables.
3. Split raw features into families (spectral ratios, entropy, Hjorth, etc.) to see whether ComBat helps some families while blunting others.
