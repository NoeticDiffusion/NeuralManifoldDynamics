Nu har jag allt underlag jag behöver. Här är min bedömning:

---

## Vad HRV v0.1 redan levererar

Block_native-outputen har redan:
- `m`, `d`, `e` + derivator (korrekt efter bugfixen vid nästa re-run)
- HRV: `hr_mean_bpm`, `ibi_mean_ms`, `sdnn_ms`, `rmssd_ms`, `pnn50`, `nn_count`, `artifact_fraction`, `quality_score`
- Anchor-state: `vagal_index`, `sympathetic_index`, `vascular_index`, `anchor_index` + derivator
- Labels: `task_state_label`, `task_load_label`, `task_load_n`
- Kvalitet: `ppg_quality_quality`, `pupil_quality_quality`

**Saknas:** raw pupildata, gaze, HRV-komplexitet, event-locked analyses, trajectory-mått, Jacobian off-diagonal.

---

## Bang-for-the-buck-ranking

### Tier 1 — Mycket låg insats, hög utdelning

**Item 6: Trajectory curvature per trial** ⭐ Bäst
- **Kräver noll pipeline-ändringar.** Vi har nu korrekt `m`, `d`, `e` per block-window efter bugfixen. `path_length`, `mean_curvature` och `trajectory_efficiency` är rena geometriska beräkningar i analysrepot på befintlig data.
- Testar direkt EAP-prediktionen om ankrade tillstånd → rakare manifold-trajektorier under encoding.
- Kan göras *nu* i `J:\repos\NoeticDiffusion\` utan ny pipeline-körning.

**Item 4 (del): Pupil-kolumner i block_native** ⭐
- `pupil_diameter_mean` och `pupil_dilation_velocity` finns redan i `features_raw` (de ingår i embodied-channelsarna). Exporteras bara inte till block_native för att `_raw_feature_exports_at` bara exporterar prefix `ecg_hrv_`.
- En ändring i en rad i `block_native_export.py` (lägg till `"pupil_"` i `prefixes`), eller i config. Sedan ingår det automatiskt i nästa re-run (som vi ändå måste göra efter SWI-bugfixen).
- Ger pupil som parallell anchor-variabel bredvid HRV i samma analysram.

---

### Tier 2 — Medium insats, hög vetenskaplig utdelning

**Item 7: Inter-network Jacobian off-diagonal**
- Pipeline beräknar redan 4-nätverks Jacobian-block (frontal, central, parietal\_occipital, temporal). Off-diagonal-blocken finns i matrisen men exporteras inte.
- Måttlig pipeline-utökning (lägg till Frobenius-norm per nätverkspar i region-Jacobian-outputen).
- Testar om ankartillstånd modulerar *specificitet* i inter-nätverkskoppling — mer distinkt EAP-prediktion än det nuvarande "uniform decrease during task"-fyndet.

**Item 1: R-peak event-locked MNPS**
- Event-locked-infrastrukturen är klar och körbar. R-peaks detekteras redan i HRV-pipelinen.
- Kräver: exportera R-peak-timestamps som events-TSV → mata in till `event_locked_runner`.
- Direkt test av EAP:s ~1 Hz cardiac-cycle-prediktion — det starkaste otestade påståendet.
- Medium insats: hitta var R-peaks sparas, lägga till export, test.

**Item 5: Recall-onset event-locked**
- Task-designen har 2 s SOA × sekvens-längd + retention interval → recall-timing kan rekonstrueras.
- Testar det enda riktigt otestade retrieval-påståendet (hittills har allt fokuserat på encoding).
- Medium insats: behavioral-parsing, sedan händer resten automatiskt via event_locked.

---

### Tier 3 — Medium insats, medium utdelning

**Item 8: HRV-komplexitet (SampEn, DFA)**
- Kräver `antropy`/`nolds`, minimalt 30 RR-intervall per fönster (~30 s).
- Tillför en fraktal/nonlinjär dimension till anchor-ytan som HRV-power inte fångar.
- Vet ej om det tillför förklaringsgrad bortom RMSSD — bör testa det *efter* vi vet att RMSSD redan ger signal.

**Item 3: Respiratory phase (EDR)**
- Intressant men ds003838 saknar respirationsbälte → RSA-baserad EDR är en approximation.
- Mer "nice to have" för att separera cardiac vs. respiratory anchor-bidrag.

---

### Tier 4 — Hög insats

**Item 2: Per-digit stimulus event-locked** — Medium-hög (behavioral-parsing, SOA-struktur)

**Item 9: Gaze stability/blink** — Låg insats men låg marginalvärde givet att pupil_quality redan finns

**Item 10: Cardiac-pupil coherence** — Hög insats, 30 s-fönster vid gränsen för spektralupplösning

---

## Sammanfattning — mina tre rekommendationer

| Prioritet | Item | Insats | Kräver re-run? | Kommentar |
|---|---|---|---|---|
| 1 | **#6 Trajectory curvature** | Mycket låg | Nej (analysrepo) | Gör nu på befintlig H5-data |
| 2 | **#4 Pupil-kolumner** | Minimal (1 rad) | Ja (ändå pga SWI-fix) | Lägg till i samma re-run |
| 3 | **#1 R-peak event-locked** | Medium | Ja | Starkaste otestade EAP-prediktionen |

Om vi ändå kör om ds003838 för SWI-bugfixen är det gratis att slå ihop #2 med den körningen. #6 kan påbörjas direkt i analysrepot på befintlig (korrekt för rest + sub-094 digit_span) data medan re-run körs. #1 och #5 ger de starkaste nya vetenskapliga bidragen men kräver mer arbete.