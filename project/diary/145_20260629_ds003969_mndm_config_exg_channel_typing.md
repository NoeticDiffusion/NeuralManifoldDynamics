# 145 — ds003969 mndm ingest config + EXG channel typing

**Date:** 2026-06-29
**Scope:** Add `ds003969` (Meditation vs thinking task) to the mndm embodied-anchoring
ingest with correct non-EEG channel typing, so ECG/PPG/EOG/RESP are picked up.

## Context

`ds003969` was downloaded in full to `K:/ExternalReceivedDatasets/openneuro/received/ds003969`
(see diary 144). It is a Biosemi 64-channel EEG `.bdf` dataset (98 subjects, 4
recordings each: `task-med1breath`, `task-med2`, `task-think1`, `task-think2`).
The 16 auxiliary channels (EXG1-8, GSR1-2, Erg1-2, Resp, Plet, Temp) are typed
`MISC`/`GSR`/`RESP`/`TEMP` in the BIDS `channels.tsv`, but mndm reads the `.bdf`
directly with MNE (not the BIDS sidecar types), so the auxiliary channels need
explicit `channel_typing` rules for the embodied-anchoring ingest to find EOG,
ECG, and PPG.

## EXG sensor mapping (external, from dataset author)

From the OpenNeuro comment thread (Claire Braboszcz, cross-confirmed by
A. S. Ghatpande):

| Channel | Placement                  | Assigned type |
|---------|----------------------------|---------------|
| EXG1    | left eye corner            | eog (HEOG)    |
| EXG2    | right eye corner           | eog (HEOG)    |
| EXG3    | left eye eyebrow (above)   | eog (VEOG)    |
| EXG4    | left eye below             | eog (VEOG)    |
| EXG5    | left mastoide (M1)         | misc          |
| EXG6    | right mastoide (M2)        | misc          |
| EXG7    | middle of the collar bone  | **ecg**       |
| EXG8    | extra electrode for Fp1    | misc (see note) |

This is **external metadata (author comment)**, category 1/3 — used as ground
truth for channel roles. The "EXG5/6 = VEOG" guess in the first comment was
corrected by the author; do not use the first comment's mapping.

## Decisions and why

1. **Per-dataset overlay config, not a "main" config.** mndm has no top-level
   `config_ingest.yaml`; the canonical pattern (ds003838, ds006848) is a
   per-dataset overlay that imports `config_ingest_common_eeg.yaml`. Created
   `mndm/config/config_ingest_ds003969.yaml` following that pattern.

2. **EXG7 → `ecg`.** `preprocess.py` picks the ECG channel by `pick_types(ecg=True)`,
   so typing is required and sufficient for HRV extraction.

3. **Plet → `bio`, GSR/Temp → `misc` (subtle).** `features/ppg.py` finds the PPG
   channel by name regex `(ppg|pleth|pulse|photopleth)` and, on no match, falls
   back to **all** `bio`-type channels. The channel is named `Plet` (no `h`), so
   the name regex does **not** match and the bio fallback is used. Therefore:
   - `Plet` must be `bio` to be picked as PPG.
   - `GSR1/GSR2/Temp` must **not** be `bio`, or the bio fallback would grab them
     as PPG and corrupt PPG features. Typed `misc`. An EDA extractor + `bio`
     typing for GSR is a future step.

4. **EXG8 → `misc` (not `eeg`).** EXG8 is the replacement electrode for a
   defective, flat Fp1. Typing it `eeg` would create a second EEG channel with no
   montage position and break average-reref / CSD / regional groups. The pipeline
   has no generic channel-rename hook, so EXG8 is held as `misc` and the flat
   original Fp1 is left to auto bad-channel detection. **Future refine:** rename
   EXG8 → Fp1, drop the flat original Fp1, let EXG8 carry the Fp1 montage slot.

5. **EXG5/6 (mastoids) → `misc`.** Reference electrodes, not used as EEG under
   the default average reref.

6. **Artifacts: `eog_reg`.** Real EOG channels (EXG1-4) now exist and are typed,
   so EOG regression is light and appropriate for 98 subjects. ECG is left in the
   raw (no ICA ECG removal) so HRV sees the true cardiac signal.

7. **Embodied channels:** `ecg_hr_bpm`, `ecg_hrv_rmssd_ms`, `ppg_rate_bpm`,
   `ppg_amplitude_mean` (mirrors ds006848).

8. **Metadata:** `participants.tsv` `group` column (lowercase `ctr/htr/sny/vip`)
   normalized to `Control/Himalayan/Isha/Vipassana`; BIDS task labels collapsed to
   `meditation` / `thinking`.

## Validation (internal, category 2)

Loaded the merged config via `core.config_loader.load_config` and simulated the
`_apply_channel_typing` rule match + the preprocess ECG/PPG pick logic against
the real `sub-001_task-think1_channels.tsv` channel list:

```
ECG picks: ['EXG7']
EOG picks: ['EXG1', 'EXG2', 'EXG3', 'EXG4']
RESP picks: ['Resp']
BIO picks: ['Plet']
PPG picks: ['Plet']        # via bio fallback; no GSR/Temp contamination
misc: ['EXG5','EXG6','EXG8','GSR1','GSR2','Erg1','Erg2','Temp']
eeg count: 64              # all 64 scalp channels remain EEG
```

`channel_typing.enabled=True`, 15 dataset rules resolved, `artifacts.method=eog_reg`,
`features.ecg.hrv.enabled=True`, `mnps.embodied` channels resolved.

## What is NOT done / uncertain

- **EXG8/Fp1 rename** (decision 4) is deferred — needs a generic rename hook or a
  dataset-specific preprocess step. Until then the usable Fp1 signal is on a
  `misc` channel and Fp1 is effectively absent from the EEG montage.
- **EDA / GSR** and **Resp / Temp** have no feature extractors; they are typed
  correctly but not yet turned into embodied features.
- **block_native / event_locked / within_run_labels** not configured: the
  recordings are continuous meditation/thinking sessions without trial structure.
  Embodied anchoring operates on the MNPS window stream regardless.
- No full ingest run yet — only config + static validation. First ingest run will
  confirm end-to-end ECG/PPG feature emission.

## Files

- Created: `mndm/config/config_ingest_ds003969.yaml`
