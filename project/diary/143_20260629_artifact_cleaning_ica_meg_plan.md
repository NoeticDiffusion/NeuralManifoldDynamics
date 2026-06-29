# 143 — 2026-06-29 — Artifact Cleaning Layers 1, 2 + MEG Plan

## Session summary

Continuation of peer-review cleaning tasks from `cleaning.md`. Three items addressed:

---

## Lager 1: Condition-structured artifact balance audit (pr04)

**Script**: `project/scripts/pr04_artifact_balance.py`  
**Output**: `04_artifact_balance_by_condition.csv` + `04_artifact_balance_summary.md`

**Method**: For each subject, reconstructed encoding intervals from BIDS `events.tsv`
and assigned features.parquet epochs (verbalwm, 38 487 epochs, 30 subjects) to
conditions via vectorized >=25% encoding overlap. Computed per-subject × condition
medians of 5 artifact proxies and ran Friedman + BH-FDR Wilcoxon across 4 conditions.

**Key fix during development**: `Encoding_Set_Simultaneous` is the Simultaneous encoding
onset marker (not `Encoding_DigitValue_*`). Adding this to the reconstruction logic
recovered all 4 conditions (previously Simultaneous was missing from the audit).

### Results

| proxy | chi2 | p_friedman | sig? |
|-------|------|-----------|------|
| eeg_highfreq_power_30_45__g_frontal | 23.00 | 0.00004 | YES |
| eeg_highfreq_power_30_45__g_temporal | 1.80 | 0.615 | no |
| eeg_gamma__g_frontal | 23.00 | 0.00004 | YES |
| eeg_hjorth_mobility__g_frontal | 9.72 | 0.021 | YES |

**Critical direction**: Frontal high-freq ordering is
**Simultaneous > Slow > Fast > FastDelay** — the OPPOSITE of the m/d ordering
(Fast/FastDelay > Simultaneous/Slow). Simultaneous > FastDelay: p=0.0003;
Simultaneous > Fast: p=0.023.

**Interpretation**: The condition-structured frontal effect cannot explain the m/d
finding via artifact amplification — it would require artifact to LOWER m/d in the
condition with highest artifact (Simultaneous), which is what we observe. Temporal
muscle artifact is not condition-structured (p=0.61). This provides an argument
AGAINST frontal artifact as a confound for the m/d ordering.

**Claims ledger**: Updated `ds006848_verbal_wm_claims.md` with artifact audit result and
EEG artifact status table.

---

## Lager 2: ICA + RANSAC configuration

**Files updated**:

1. `mndm/config/config_ingest_common_eeg.yaml` — added generic `preprocess.ica` and
   `preprocess.bad_channels` sections (disabled by default for backward compatibility):
   - ICA: n_components=20, FastICA, EOG via `proxy_channels: [Fp1, Fp2]`, ECG optional
   - RANSAC: min_corr=0.75, unbroken_time=0.4, noise_threshold=4.0

2. `mndm/config/config_ingest_ds006848.yaml` — added dataset-specific override:
   - ICA enabled=false (ready to flip to true for ICA-cleaned rerun)
   - ECG component detection enabled (ECG co-recorded in BrainVision file)
   - RANSAC enabled=false, max_bad_fraction=0.2

All settings are **disabled** to preserve backward compatibility with existing runs.
Activation for the ICA-cleaned rerun: set `ica.enabled: true` in the ds006848 config.

---

## MEG artifact plan (ds003645)

**File updated**: `mndm/config/config_ingest_ds003645.yaml`

Added a `preprocess.ica` block for the Neuromag FIF MEG dataset with:
- `n_components: 0.999` (fraction-of-variance, auto-determines rank post-SSS)
- EOG: channels = `[EOG061, EOG062]` (Neuromag standard)
- ECG: channel = `ECG063` (Neuromag standard)
- `bad_channels.method: "correlation"` (RANSAC not ideal for MEG; SSS handles SQUID jumps)

All MEG ICA settings are disabled pending a dedicated MEG ICA rerun.

**Key MEG vs EEG differences documented in config**:
- SQUID jumps / environmental noise → MaxFilter/SSS (pre-processing, not ICA)
- Head movement → cHPI coil compensation
- ICA rank is determined by SSS (typically 64-80, not 20)
- Bad channel interpolation deferred to MaxFilter for MEG

---

## State at end of session

| task | status |
|------|--------|
| pr04 artifact balance | Completed; Simultaneous has highest frontal HF |
| ICA config (common_eeg) | Added, disabled by default |
| ICA config (ds006848) | Added, disabled by default |
| ICA config (MEG ds003645) | Added, disabled by default |
| Claims ledger update | Updated with artifact direction finding |

**Next steps**:
1. Activate `ica.enabled: true` in ds006848 config and re-run pipeline for ICA-cleaned validation.
2. Add ICA activation to the `peer_review_batch.ps1` workflow.
3. After ICA rerun, re-run 04b + 04c on ICA-cleaned data and compare Friedman p-values.
