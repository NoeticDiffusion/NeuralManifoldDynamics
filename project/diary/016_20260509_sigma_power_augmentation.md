# 016 — 2026-05-09 — Sigma-Band Power Augmentation for Event-Locked Parquets

## Session goal

Implement point 4 from the analysis repo's incremental-validity roadmap: compute
sigma-band power (and full spectral baseline) for **every** window in the
event-locked parquet — both spindle events and matched controls — so that a
"spectral-only vs spectral + MNPS geometry" classifier has equal feature coverage
across conditions.

## Problem being solved

YASA features (`yasa_amplitude`, `yasa_rms`, `yasa_abspower`, etc.) exist only for
detected spindle events, not for matched controls. Without equal-coverage spectral
features on both conditions, incremental-validity analysis cannot be done fairly.
The H5 files contain coarse spectral features but lack a `sigma` band column.

## Solution

`project/smoke_tests/augment_sigma_power.py`:

For each subject and each row in the event-locked parquet (event + control):

1. Load the raw EDF (lazy, no full preload).
2. Extract the 6 s EEG epoch at `[window_start_sec, window_end_sec]`.
3. Compute Welch PSD (`nperseg = 256`, `noverlap = 128`, `scaling="density"`).
4. Integrate (log₁₀ mean PSD) over five bands:
   - `eeg_delta_psd`: 0.5–4 Hz
   - `eeg_theta_psd`: 4–8 Hz
   - `eeg_alpha_psd`: 8–12 Hz
   - **`eeg_sigma_psd`**: 12–15 Hz  ← primary new feature
   - `eeg_beta_psd`: 15–30 Hz
5. Add `eeg_spow_ok` (bool): False if epoch is out-of-range, non-finite, or all-zero.
6. Write augmented parquet as `*_event_locked_v1_{slug}_spow.parquet` (new file,
   original parquet untouched).

## Sanity check (sub-1, PSG_C3)

```
eeg_spow_ok: 4230/4230 (100%)

Sigma (log10 µV²/Hz, mean):
  spindle_event  / event bin  : 0.952
  matched_control / control   : 0.520
  spindle_event  / pre_far    : 0.572
  spindle_event  / pre_near   : 0.673
  spindle_event  / post_near  : 0.669
  spindle_event  / post_far   : 0.579
```

Event-bin sigma is +0.43 log₁₀ units above matched controls. The temporal profile
(ramp up from pre_far → pre_near → event, then ramp down) is physiologically
consistent with spindle activity. Delta and alpha are stable across conditions,
as expected.

## Usage note for analysis repo

- `bin_label` for matched controls is `'control'`, not `'event'`.
  Use `condition` column to group: `spindle_event` vs `matched_control`.
- Spectral features are computed from the **detection channel** (C3 or F3).
- Suggested incremental-validity feature sets:
  - **Spectral-only**: `eeg_delta_psd`, `eeg_theta_psd`, `eeg_alpha_psd`,
    `eeg_sigma_psd`, `eeg_beta_psd` (+ Hjorth / entropy from H5 if joined)
  - **+ MNPS geometry**: `m`, `d`, `e`, `m_a`, `m_e`, `m_o`, `d_n`, `d_l`,
    `d_s`, `e_e`, `e_s`, `e_m`
  - **+ derivatives**: `m_dot`, `d_dot`, `e_dot`

## Full batch

128-subject C3 augmentation batch started (~25 min runtime). Output files:
`{PROC}/{sub_run_dir}/{sub}_Sleep_acq-psg_event_locked_v1_psg_c3_spow.parquet`

F3 batch can be queued separately with `--channel PSG_F3`.

## Outstanding caveat

`eeg_sigma_psd` is computed from the same EDF channel used for spindle detection.
It is not an independent spectral measure. Incremental validity analysis must note:
"spectral feature and detector share the same signal; independence requires a
different channel or modality."
