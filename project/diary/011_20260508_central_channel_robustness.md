# 011 — 2026-05-08 — Central-Channel Robustness: PSG_C3 vs PSG_F3

## Session type
Central-channel robustness check (architect-approved next step)

## What was done

### Batch pipeline parametrized for channel
`batch_event_locked.py` extended with `--channel` argument. Parquet and spindle CSV filenames now include a channel slug (`_psg_f3`, `_psg_c3`), enabling multi-channel comparison without overwriting. Source tag also includes channel name for provenance.

### PSG_C3 batch: 5 subjects, same protocol
Run with `--channel PSG_C3 --skip-mndm`. Same YASA canonical params, same N2 filter, same controls.

**C3 QC gate results:**

| Subject | Rate/min | Bins | Match | Finite | Excl | Gate |
|---------|----------|------|-------|--------|------|------|
| sub-1 | 0.75 | 5/5 | 1.00 | 100% | 16% | PASS |
| sub-2 | 0.39 | 5/5 | 1.00 | 100% | 10% | PASS |
| sub-3 | 0.26 | 5/5 | 1.00 | 100% | 13% | WARN (rate) |
| sub-4 | 0.86 | 5/5 | 1.00 | 100% | 7% | PASS |
| sub-5 | 0.68 | 5/5 | 1.00 | 100% | 1% | PASS |

sub-3 WARN: rate 0.26/min marginally below 0.3/min gate. PSG_C3 detects fewer spindles than PSG_F3 for sub-3 (76 vs 351) — marked as flagged in analysis.

### Direction analysis: C3 meets 4/5 criterion

**Event-bin (t=0 to +3 s) delta per channel and subject:**

| Subject | F3 Δm | F3 Δd | F3 Δe | C3 Δm | C3 Δd | C3 Δe |
|---------|-------|-------|-------|-------|-------|-------|
| sub-1 | −0.16 | −0.05 | +0.03 | +0.00 | +0.04 | +0.01 |
| sub-2 | +0.01 | +0.19 | +0.15 | +0.21 | +0.43 | +0.20 |
| sub-3 | +0.04 | −0.03 | +0.22 | +0.10 | +0.13 | +0.25 |
| sub-4 | −0.10 | +0.06 | −0.06 | −0.27 | +0.17 | −0.03 |
| sub-5 | +0.28 | +0.27 | −0.18 | +0.08 | −1.20 | +1.05 |

**Direction agreement:**
- PSG_F3: 3/5 for all dims (60%) — threshold not met
- PSG_C3: **4/5 for all dims** (80%) — meets pre-specified promotion criterion

**Cross-channel F3/C3 sign agreement:**
- MNPS-m: 4/5 (80%) — **promoted**
- MNPS-d: 2/5 (40%) — not promoted
- MNPS-e: 4/5 (80%) — **promoted**

### Promotion outcome

The pre-specified criterion (≥ 4/5 agreement in any dimension or coherent cross-channel) is met:
- C3 standalone: 4/5 for m, d, e
- F3+C3 cross-channel: 4/5 for m and e

**Valid promoted claim:**
> "Using YASA 0.7.0 canonical parameters on PSG_C3 N2, 4/5 subjects show event > control in MNPS-m, MNPS-d, and MNPS-e during the spindle event bin (t=0 to +3 s) vs matched N2 controls. F3/C3 cross-channel sign agreement is 4/5 for m and e. This meets the pre-specified directional consistency threshold."

## Key observations

1. **Sub-4 is the consistent outlier**: event < control for MNPS-m on both F3 and C3. This is the only subject showing consistent negative m-delta across channels.

2. **Sub-5 C3-d is anomalous**: Δd = −1.20 (large), while F3-d = +0.27. This magnitude disagreement for sub-5 d is unexplained — may reflect outlier spindle epochs or channel-specific artifact on C3.

3. **Sub-1 C3 is near-zero for all dims**: small positive deltas, consistent sign with C3 group but not with F3 (which was negative for m, d). Sub-1 F3/C3 m and d signs disagree.

4. **MNPS projection caveat**: The MNPS space is derived from the same H5 file for both channels. What changes between F3 and C3 is only which spindles are detected, not the MNPS coordinates themselves. So "channel robustness" here means detector robustness, not full MNPS channel robustness.

## What this does NOT mean

- Does not prove a spindle-locked MNPS effect. These are descriptive measurement differences.
- Does not generalise beyond ds005555 or YASA 0.7.0 detector-derived events.
- Does not support any NDT or clinical interpretation.
- Sub-5 anomaly on C3-d/e not explained.

## Next steps discussed

1. **Sub-5 audit**: investigate why sub-5 shows large C3 d/e values. Check epoch-level distributions, potential artifact epochs, outlier spindle events.
2. **Scale-mode audit** can now be considered — a channel-stable (m, e) directional pattern exists, worth checking if scale choices change it.
3. **Repeat with sub-3 excluded** (C3 rate below gate) to check if 4/5 pattern holds without the flagged subject.
4. **Longer-term**: extend to additional subjects (sub-6 to sub-10) for replication.

## Bug logged

`project/issues/issue_subject_filter_zero_padding.md` — `--subject N` zero-padding incompatibility with non-padded BIDS IDs. Non-blocking workaround: run without `--subject` filter.
