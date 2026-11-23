# ds004148 baseline QC attrition

Participant attrition from the completed, frozen v1.1 QC comparison.
Core-8 support means the same participant has at least 30 QC-eligible
windows and at least 120 unique seconds in both EC and EO for each of
the eight listed coordinates. No thresholds were changed after inspecting
the cohort. Window-level measurements remain in the baseline output root.

- Core-8 paired participants: 45/60.
- Alpha-envelope-CV paired participants: 44/60.
- Complete-9D paired participants: 44/60.
- Each EC/EO cell reports QC-eligible windows / unique supported seconds.

## Participants without a Core-8 pair

| Subject | EC retained windows / s | EC reason | EO retained windows / s | EO reason |
|---|---:|---|---:|---|
| sub-02 | 6 / 40 | eligible_windows_below_30;unique_seconds_below_120 | 55 / 240 | pass |
| sub-04 | 35 / 172 | pass | 17 / 116 | eligible_windows_below_30;unique_seconds_below_120 |
| sub-05 | 38 / 184 | pass | 22 / 112 | eligible_windows_below_30;unique_seconds_below_120 |
| sub-06 | 53 / 232 | pass | 27 / 132 | eligible_windows_below_30 |
| sub-12 | 49 / 216 | pass | 21 / 116 | eligible_windows_below_30;unique_seconds_below_120 |
| sub-23 | 31 / 144 | pass | 29 / 128 | eligible_windows_below_30 |
| sub-30 | 21 / 104 | eligible_windows_below_30;unique_seconds_below_120 | 51 / 216 | pass |
| sub-31 | 7 / 48 | eligible_windows_below_30;unique_seconds_below_120 | 26 / 140 | eligible_windows_below_30 |
| sub-33 | 42 / 192 | pass | 20 / 116 | eligible_windows_below_30;unique_seconds_below_120 |
| sub-36 | 59 / 260 | pass | 27 / 148 | eligible_windows_below_30 |
| sub-41 | 25 / 116 | eligible_windows_below_30;unique_seconds_below_120 | 35 / 168 | pass |
| sub-42 | 27 / 156 | eligible_windows_below_30 | 41 / 188 | pass |
| sub-43 | 27 / 132 | eligible_windows_below_30 | 20 / 120 | eligible_windows_below_30 |
| sub-46 | 45 / 208 | pass | 28 / 144 | eligible_windows_below_30 |
| sub-52 | 8 / 56 | eligible_windows_below_30;unique_seconds_below_120 | 38 / 196 | pass |

## Additional alpha-envelope-CV loss

sub-55: EC 55 windows / 236 s; EO 29 windows / 128 s. Pair reason: EO_eligible_windows_below_30.

## Per-participant condition data

See `participant_qc_attrition.csv` for all 60 participants, both
conditions, retained and excluded supported windows, unique seconds,
muscle-overlap counts, detector/source validity, alpha-envelope-CV
support, complete-9D support, and pair-exclusion reasons.
