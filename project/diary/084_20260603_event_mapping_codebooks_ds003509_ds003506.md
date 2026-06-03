# Event-mapping codebooks for ds003509 and ds003506

## Question

Can the compact trigger-enriched event labels for `ds003509` and `ds003506` be upgraded from raw provenance-only labels into explicit mapped stage codes inside the summarize/event-provenance layer?

## Work completed

1. Inspected the current `mndm` event provenance path.
2. Confirmed that the active BIDS-event mapping path uses:
   - `epoching.datasets.<ds>.sampling.stage_columns`
   - `mnps.stage_codebook`
   rather than a separate `event_mapping.codebook` key.
3. Added explicit compact-label codebooks to:
   - `mndm/config/config_ingest_ds003509.yaml`
   - `mndm/config/config_ingest_ds003506.yaml`
4. Ran targeted `summarize --subject 002` verification for both datasets.
5. Found and fixed an important implementation constraint:
   - stage arrays are serialized as signed `int8`,
   - so initial code assignments above `127` wrapped into negative values,
   - the codebooks were revised into safe positive `int8` ranges.

## Final codebooks

### ds003509

- Training family:
  - `training_stimulus_congruent -> 10`
  - `training_stimulus_incongruent -> 11`
  - `training_response_correct -> 12`
  - `training_response_incorrect -> 13`
  - `training_no_response -> 14`
  - `training_feedback_reward -> 15`
  - `training_feedback_punishment -> 16`
  - `training_feedback_timeout -> 17`
  - `training_feedback_error -> 18`
- Test family:
  - `test_stimulus_easy -> 20`
  - `test_stimulus_direct_conflict -> 21`
  - `test_stimulus_relative_conflict -> 22`
  - `test_stimulus_other -> 23`
  - `test_response_correct -> 24`
  - `test_response_incorrect -> 25`
  - `test_no_response -> 26`

### ds003506

- Choose family:
  - `choose_instruction -> 40`
  - `choose_stimulus -> 41`
  - `choose_response_left -> 42`
  - `choose_response_right -> 43`
  - `choose_feedback_reward -> 44`
  - `choose_feedback_punishment -> 45`
  - `choose_feedback_timeout -> 46`
  - `choose_feedback_error -> 47`
- Match family:
  - `match_instruction -> 50`
  - `match_stimulus -> 51`
  - `match_response_left -> 52`
  - `match_response_right -> 53`
  - `match_feedback_reward -> 54`
  - `match_feedback_punishment -> 55`
  - `match_feedback_timeout -> 56`
  - `match_feedback_error -> 57`

## Internal validated results

### ds003509 verification run

- Output:
  - `M:/datasets/processed/openneuro/ds003509/neuralmanifolddynamics_ds003509_20260603_073511`
- `stage_mapping_qc.json` now reports:
  - `n_event_rows_mapped: 1478` and `n_event_rows_unmapped: 0` for `ses-01`
  - `n_event_rows_mapped: 1550` and `n_event_rows_unmapped: 0` for `ses-02`
  - `mapping_mode_counts.direct`
  - populated `mapped_stage_code_counts`
  - positive `window_stage_counts` using the intended codes (`10`, `11`, ..., `26`) plus residual `-1` unlabeled windows
- Example direct-code counts:
  - `training_stimulus_congruent -> 198`
  - `training_stimulus_incongruent -> 189`
  - `training_response_correct -> 320`
  - `training_feedback_reward -> 160`
  - `test_stimulus_relative_conflict -> 128`

### ds003506 verification run

- Output:
  - `M:/datasets/processed/openneuro/ds003506/neuralmanifolddynamics_ds003506_20260603_073512`
- `stage_mapping_qc.json` now reports:
  - `n_event_rows_mapped: 1151` and `n_event_rows_unmapped: 0` for `ses-01`
  - `n_event_rows_mapped: 347` and `n_event_rows_unmapped: 0` for `ses-02`
  - `mapping_mode_counts.direct`
  - populated `mapped_stage_code_counts`
  - positive `window_stage_counts` using the intended codes (`40`, `41`, ..., `57`) plus residual `-1` unlabeled windows
- Example direct-code counts:
  - `choose_instruction -> 103`
  - `choose_feedback_reward -> 67`
  - `match_instruction -> 121`
  - `match_stimulus -> 249`
  - `match_feedback_error -> 20`

## Notes

- This change upgraded the compact event families into explicit mapped stage codes without needing a new event-mapping schema branch in code.
- The runtime still reports `stage_source: "consensus"` in the QC payload, but the event provenance tables now carry fully populated `mapped_stage_code` arrays and zero unmapped compact labels for the verified subject.
- Residual `-1` window counts remain expected because not every overlapping analysis window contains a mapped point event.
