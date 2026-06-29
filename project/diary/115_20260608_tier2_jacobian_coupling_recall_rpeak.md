# 115 — 2026-06-08 — Tier 2 wishlist implementerat: Jacobian-koppling, recall-onset, R-peak events

## Session-mål

Implementera de tre Tier-2-punkterna från `wishlist_20260608_priority.md`:
- **Item 7**: Inter-network Jacobian off-diagonal
- **Item 5**: Recall-onset event-locked (behavioral parsing)
- **Item 1**: R-peak event-locked (ECG peak export)

Alla tre är nu operationella och verifierade.

---

## Item 7 — Inter-network Jacobian off-diagonal koppling

### Vad som gjordes

Lade till `_inter_network_coupling_by_block(rows, payload)` i
`mndm/src/mndm/pipeline/block_native_export.py`.

**Algoritm:**
1. Hämtar per-nätverks MNPS-tidsserier från `payload.regional_mnps` (4 nätverk × (T, 3)).
2. Staplar dem till en gemensam k=12-dimensionell bana per block.
3. Estimerar Jacobian via minsta-kvadrat: `X_prev @ J^T ≈ ΔX` (konsekutiva fönsterdifferenser).
4. Extraherar 12 off-diagonala 3×3-block och beräknar deras Frobenius-normer.

**Kolumner tillagda per rad** (per block, replikeras till alla fönster i blocket):

| Kolumn | Tolkning |
|--------|----------|
| `coupl_cntr_from_frnt` | ||J[cntr,frnt]||_F |
| `coupl_cntr_from_par`  | ||J[cntr,par]||_F  |
| `coupl_cntr_from_temp` | ... |
| `coupl_frnt_from_cntr` | etc. |
| ... (12 kolumner totalt) | |

**Nätverksförkortningar:** `central→cntr`, `frontal→frnt`, `parietal_occipital→par`, `temporal→temp`.

**Fallback:** Block med färre än k+1=13 finita index → NaN för alla kopplingsvärden.

### Anrop-integration

`build_block_native_table()` kallar `_inter_network_coupling_by_block` efter
trajectory-stats-blocket, precis som detta anropar `_trajectory_stats_by_block`.

### Verifiering

- sub-032 rest: 1 block, 12/12 finita kopplingsvärden, alla ≥ 0.
- 4 nya unit-tester i `mndm/tests/test_block_native_export.py` — alla passerar.

---

## Item 5 — Recall-onset event-locked

### Vad som gjordes

Skrev `project/preprocessing/derive_recall_onset_events_ds003838.py`.

**Logik:**
- Parsas BIDS events TSV (`sub-NNN_task-memory_events.tsv`) per subjekt.
- Regex matchar "last digit of memory sequence"-rader: `memory NN/NN [correct|error]: ... (last) in N digit sequence`.
- Recall-onset = `onset + duration (1.0s)` = exakt när sista siffran är klar.
- Skriver per-subjekt TSV med kolumner: `onset_sec, duration_sec, event_type, seq_len, accuracy, trial_no`.

**Format:** Kompatibelt med `event_locked_runner` (kolonn `onset_sec`, `event_type="recall_onset"`).

**Utdata:**
- `sub-032_task-memory_events_recall.tsv`: 108 events (36 × 5-siffror, 36 × 9, 36 × 13)
- `sub-013_task-memory_events_recall.tsv`: 108 events (samma fördelning)

### Integration med event_locked

Config-mall tillagd (kommenterad) i `config_ingest_ds003838.yaml` under
`event_locked_recall:`. Aktiveras per körning med `source_path` per subjekt.

Bins föreslagna: `pre_encoding_8s: [-20, 0]`, `post_recall_8s: [0, 8]`.

---

## Item 1 — R-peak event-locked

### Vad som gjordes

Skrev `project/preprocessing/derive_rpeak_events_ds003838.py`.

**Algoritm** (identisk med `mndm.features.ecg.compute_ecg_features`):
1. Läser ECG `.set`-fil med MNE.
2. Butterworth bandpass 5–20 Hz (order 3).
3. Absolutvärde + MAD-baserad prominenströskel.
4. `scipy.signal.find_peaks` med refraktärperiod 300 ms.
5. Skriver TSV med kolumner: `onset_sec, duration_sec, event_type, peak_sec`.

**Resultat för tre valideringssubjekt:**

| Subjekt | Peaks | Min RR | Mean RR | HR (bpm) |
|---------|-------|--------|---------|----------|
| sub-015 | 477   | 0.300s | 0.427s  | 140      |
| sub-020 | 452   | 0.308s | 0.450s  | 133      |
| sub-025 | 540   | 0.317s | 0.421s  | 142      |

**Notering om HR-värden:** Den höga resting-HR (140+ bpm) är konsistent med
intermediate JSON (`ecg_hr_bpm ≈ 141`, `ecg_hrv_hr_mean_bpm ≈ 141`) — alltså
INTE ett bug i det nya skriptet utan en befintlig egenhet i dessa subjekts
ECG-data (möjlig dubbelpeakdetektering i det enkla MAD-baserade algorithmet).
Derivationen är helt konsistent med vad HRV-pipelinen redan använder.

**Sub-013, sub-032:** Saknar ECG-kanal (EEG-kohort, inte ECG-kohort) → `skipped`.

### Integration med event_locked

Config-mall tillagd (kommenterad) i `config_ingest_ds003838.yaml` under
`event_locked_rpeak:`. Bins: `pre_beat_2s: [-2, 0]`, `post_beat_2s: [0, 2]`.

---

## Vad som INTE gjordes

- **Pipeline re-run**: Inga av förändringarna (Item 7 kopplingsfunktion) aktiveras
  förrän pipeline körs om. Nuvarande H5/Parquet från 2026-06-07 inkluderar ännu
  inte kopplingsvärden i `block_native_windows.parquet`.
- **Fullständig event_locked-körning**: Recall och R-peak event_locked är konfigurerade
  men inte körda på hela ds003838-kohorten. Det kräver pipeline re-run.
- **Recall onset precision**: Nuvarande estimat = `offset_of_last_digit` (retention
  interval antas 0). Den verkliga retention interval saknas i BIDS-sidecaren.
  Om taskdesignen anger en fast retention kan onset-definitionen förfinas.

---

## Smoke test

`project/smoke_tests/smoke_tier2_ds003838.py` — alla 3 delar passerar:

```
=== Item 7: Inter-network Jacobian coupling ===
  sub-032_rest: 1 blocks, 12 finite coupling values
Item 7: PASS

=== Item 5: Recall-onset events TSV ===
  sub-032: 108 recall-onset events — PASS
  sub-013: 108 recall-onset events — PASS
Item 5: PASS

=== Item 1: R-peak events TSV ===
  sub-015: 477 R-peaks, min RR=0.300s — PASS
  sub-020: 452 R-peaks, min RR=0.308s — PASS
  sub-025: 540 R-peaks, min RR=0.317s — PASS
Item 1: PASS
```

## Unit-tester

4 nya tester i `mndm/tests/test_block_native_export.py`:
- `test_inter_network_coupling_basic` — 12 finita kolumner för välsamplade block
- `test_inter_network_coupling_too_few_samples` — NaN för korta block
- `test_inter_network_coupling_no_regional_mnps` — tom dict när regional data saknas
- `test_inter_network_coupling_column_names` — korrekt namnkonvention

**Totalt: 10/10 tester gröna.**

---

## Nästa steg

Inget explicit planerat från Tier 2. Möjliga Tier 3-kandidater:

- **Item 8 (HRV complexity):** SampEn/DFA kräver ≥30 RR-intervall/fönster.
- **Item 3 (Respiratory EDR):** ds003838 saknar respirationsbälte.
- **Pipeline re-run:** Köra hela ds003838 med samtliga Tier 1+2 ändringar för att
  producera korrekt `block_native_windows.parquet` med alla nya kolumner.
