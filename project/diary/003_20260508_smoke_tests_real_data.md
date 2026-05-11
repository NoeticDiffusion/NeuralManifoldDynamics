# Session Log — 003

**Date:** 2026-05-08  
**Milestone:** Röktestvalidering mot riktiga datastager (ds005555 + ANPHY)  
**Föregående:** M5 event-locked export (session 002)

---

## Goal

Validera att hela pipeline-kedjan (event-table → alignment → matched controls → export)
körs utan fel mot riktiga datasettfiler.

Viktigt: Ingendera datasett har riktiga spindle-annotationer ännu.
Syntetiska spindles injicerades (var 3:e N2-epok, centrerat i epoken).
**Detta testar rörmekaniken, inte sömnspindelbiologi.**

---

## Datasets Tested

### ds005555 / sub-1 (PSG acquisition)

| Parameter | Värde |
|---|---|
| Källa | `M:\datasets\received\openneuro\ds005555\sub-1\eeg\sub-1_task-Sleep_acq-psg_events.tsv` |
| Staging-epoker | 915 × 30 s |
| N2-epoker | 398 |
| Syntetiska spindles | 133 (var 3:e N2) |
| Alignade events | 111 (22 exkluderade vid stage-transitioner) |
| Spindle-rader i export | 222 |
| Kontroll-rader i export | 399 (3 per event, match_rate=1.00) |
| MNPS finite frac | 1.0 |
| coords_9d present | Ja |
| Elapsed | 0.28 s |

### ANPHY / EPCTL01

| Parameter | Värde |
|---|---|
| Källa | `M:\datasets\received\ANPHY\Subjects\EPCTL01\test1.txt` |
| Staging-epoker | 958 × 30 s |
| N2-epoker | 471 |
| Syntetiska spindles | 157 (var 3:e N2) |
| Alignade events | 136 (21 exkluderade vid stage-transitioner) |
| Spindle-rader i export | 272 |
| Kontroll-rader i export | 471 (3 per event, match_rate=1.00) |
| MNPS finite frac | 1.0 |
| coords_9d present | Ja |
| Elapsed | 0.40 s |

---

## Output Files

```
project/smoke_tests/results/smoke_ds005555_sub-1.csv      (50 kolumner, 621 rader)
project/smoke_tests/results/smoke_ds005555_sub-1_qc.json
project/smoke_tests/results/smoke_ANPHY_EPCTL01.csv       (50 kolumner, 743 rader)
project/smoke_tests/results/smoke_ANPHY_EPCTL01_qc.json
project/smoke_tests/results/smoke_summary.json
```

CSV-schema verifierad: 50 kolumner inkl. alla mandatory och optional fält.

---

## Observations

### Bin-distribution: `pre_far` + `event` dominerar

Båda datasetten visar bara binsar `pre_far` och `event` i spindle-räknarna.
Binsar `pre_near`, `post_near`, `post_far` är tomma.

**Orsak:** Fönsterstorlek = 30 s (full staging-epok). Bin-gränserna är definierade för
kortare MNPS-fönster (~4-8 s). Med 30 s fönster hoppar fönstercenter med 30 s åt gången,
vilket innebär att ingen fönstercentrum hamnar i [-10, 0] eller [3, 10] s relativt event.

**Konsekvens:** Detta är ett viktigt designbeslut att förmedla till arkitekten.
För meningsfull event-locked analys med spindle-tidsskalor (~0.5-3 s) behöver MNDM
köras med kortare epocher (t.ex. 4 s fönster, 2 s steg) — inte 30 s sleeping-staging-epoker.

→ **Staging-data är inte direkt kompatibel med spindle-skala.** MNDM måste re-processera
EDF-filer med kortare fönster för event-locked analys.

### Stage-transition exclusion funkar

22 resp. 21 events exkluderades korrekt pga. staging-transitioner inom 30 s-marginalen.

### Match rate = 1.00 på båda datasets

Alla events fick 3 matchade N2-kontroller. Meningsfullt resultat givet att N2 är
dominerande (398/471 epoker).

---

## Claims

### Internal validated result
- Pipeline körs komplett mot riktiga staging-filer (EDF EEG läses ej — bara events TSV).
- 50-kolumns CSV produceras korrekt med provenance, MNPS, coords_9d, event-metadata.
- Stage-transition exclusion fungerar korrekt.
- Matched controls ger match_rate=1.00 på båda datasets.

### Plausible interpretation
- Bin-fördelningen beror på fönsterstorlek. 30-s staging-epoker ger bara `pre_far` och `event`.

### Critical finding (skickas till arkitekt)
- **30 s staging-epoker är inkompatibla med spindle-tids-bins (~1-10 s).**
- Event-locked MNPS-analys kräver MNDM-körning med 4-8 s fönster mot EDF-filer,
  inte mot staging-tabeller.
- ds005555 och ANPHY har EDF-filer på disk — kan processas med MNDM om config finns.

---

## Next Smallest Step

Beslutsträd för arkitekten:

**Alt A — Kör MNDM features+summarize mot ds005555 sub-1 EDF (4-8 s fönster)**
→ Producerar HDF5 med MNPS-tidsaxel i 4-8 s upplösning
→ Spindle-event pipeline kan sedan köras mot riktiga MNPS-data
→ Kräver fungerade MNDM-config för ds005555

**Alt B — Syntetisk MNPS med korrekt tidsskala för nästa röktestiterasion**
→ Generera 4-s fönster MNPS istället för 30-s staging-epoker
→ Snabbare — validerar bin-distribution utan MNDM-körning
→ Fortfarande syntetiska MNPS-värden

**Alt C — Importera externa spindle-annotations om de finns**
→ Kolla om ds005555 eller ANPHY har separat annotation-fil med spindles
→ Skalbar om annotations finns; kräver extra kartläggning

Rekommendation: Alt A, men det kräver MNDM-config för ds005555 (finns: `config_ingest_ds005555.yaml`).
