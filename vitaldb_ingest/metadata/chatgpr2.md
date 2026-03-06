Nedan är en **implementatörs-spec** (ingest-repo) för att ladda ner VitalDB på ett sätt som är eLife-/open-science-kompatibelt: **config-drivet, cache-at, provenance-loggat, och med tydlig separation mellan “download” och “parse/extract”.** Jag ger två vägar (VitalDB-API via `vitaldb` och PhysioNet-mirror). Välj den som passar era access-villkor.

Jag utgår från att ni vill **propofol-only + EEG-waveform finns**, och att ni vill kunna reproducera exakt samma cohort senare.

---

## 0) Viktiga constraints (måste in i ingest-design)

* **Data Use Agreement**: respektera VitalDB:s villkor; spara inte tokens i repo; logga endast icke-hemliga metadata. VitalDB har egen docs + exempel. ([vitaldb.net][1])
* **Credentialed access** kan gälla via PhysioNet för VitalDB. ([physionet.org][2])
* **Reproducerbar cohort**: cohort definieras som en *deterministisk query* över metadata (caseids + spårkrav), sparas som manifest + hash.

---

## 1) Föreslagen ingest-struktur i repot

### 1.1 CLI entrypoints

* `nd_ingest_vitaldb_make_manifest`
  Skapar en cohort-manifestfil med caseids + vilken fil som ska hämtas + krav på tracks.
* `nd_ingest_vitaldb_download`
  Laddar ner `.vital` filer i bulk enligt manifest; cache + resumable.
* `nd_ingest_vitaldb_extract`
  Extraherar valda tracks (t.ex. `BIS/EEG1_WAV`, propofol infusion track) till ett internt format (HDF5/Parquet) + QC.

Separationen gör att download kan vara “data logistics” utan analyslogik.

### 1.2 Config (YAML)

Exempel på config-nycklar (princip, inte slutlig syntax):

* `source: vitaldb_api | physionet`
* `download_dir`
* `manifest_out`
* `filters` (propofol-only; eeg track required; min duration; optional surgery type)
* `required_tracks: [...]`
* `preferred_propofol_tracks: [...]` (fallbacklist)
* `max_cases`
* `rate_limit`
* `resume: true`

### 1.3 Provenance

För varje run:

* `config_hash`
* `manifest_hash`
* `download_source` + versions (pip package version)
* antal filer begärda / nedladdade / misslyckade
* exempel på misslyckade caseids + felkoder
* checksums (SHA256) för alla `.vital` lokalt

---

## 2) Väg A: VitalDB API via Python-paketet `vitaldb`

Detta är ofta enklast om ni har VitalDB-konto och kan logga in.

### 2.1 Grundfunktionalitet (enligt VitalDB docs)

VitalDB docs visar:

* `pip install vitaldb`
* `vitaldb.login(id=..., pw=...)`
* `vitaldb.download(filename)` och sedan `vitaldb.VitalFile(...)` för att lista tracks. ([vitaldb.net][1])

Exempel (minimalt) från docs-mönstret:

* logga in
* ladda ner en fil
* print track names. ([vitaldb.net][3])

### 2.2 Bulk download (enligt “File Download Guide”)

VitalDB har en guide med bulk-download via filelist + download API. ([vitaldb.net][4])
Implementera detta som:

* (i) hämta fil-lista via API eller via metadata-CSV (om tillgängligt)
* (ii) loopa och ladda ner med rate-limit + retries
* (iii) spara `.vital` i `download_dir/vital_files/{caseid}.vital`

### 2.3 Login hantering (inget hemligt i repo)

* Läs `VITALDB_ID`, `VITALDB_PW` från env vars
* Stöd `.env` lokalt men ignorera i git
* Fel vid saknad auth: tydligt meddelande (ingen fallback tyst)

---

## 3) Väg B: PhysioNet-mirror (credentialed) + bulk download

PhysioNet har VitalDB datasetet med `vital_files/*.vital` och en `clinical_parameters.csv`. ([physionet.org][2])
Fördel: stabilt arkiv/versionssatt. Nackdel: kräver PhysioNet credentialed access för många användare.

### 3.1 Hur implementera i ingest

* Använd en robust downloader (t.ex. `wget`/`curl` med credential/cookies eller PhysioNet tooling) i en isolerad “download module”.
* Resumable downloads (stora filer)
* Logga dataset version: `vitaldb/1.0.0` etc.

### 3.2 Manifest byggs från PhysioNet metadata

* Läser `clinical_parameters.csv` (om relevant)
* Filnamn kan vara `0001.vital` etc. (se exempel i PhysioNet filstruktur). ([physionet.org][5])

---

## 4) Manifest-byggare: propofol-only + EEG track exists

Här är den svåra delen: du måste definiera “propofol-only” och “EEG exists” deterministiskt.

### 4.1 “propofol-only”

VitalDB community nämner propofol pump track-namn som `Orchestra/PPF20_VOL` i vissa fall. ([ResearchGate][6])
Men track-namn kan variera mellan utrustning/sjukhus. Gör därför:

* `preferred_propofol_tracks` = lista av möjliga track names (i prioriterad ordning)
* “propofol present” = *någon* av dessa tracks finns **och** har non-missing data över min duration
* “only propofol” (om ni verkligen vill) kräver också att andra hypnotika-tracks saknas eller är noll (svårare). Jag rekommenderar i första version:

  * “propofol present” + exkludera tydliga inhalationsanestetika om tracks finns.

### 4.2 “EEG exists”

VitalDB forum/doks visar BIS EEG waveform-tracks som `BIS/EEG1_WAV` (exempel). ([vitaldb.net][7])
Implementera:

* `required_tracks` inkluderar `BIS/EEG1_WAV`
* In manifest-byggaren: för varje candidate case, öppna filens track list (utan att extrahera allt) och checka existence.

**Obs:** att “peeka” track list kräver att ni kan ladda ner filen eller metadata. Om API erbjuder track list utan full download: använd det. Om inte: gör en 2-stegs pipeline:

1. Ladda ner en liten candidate-pool (eller metadata)
2. Filtrera genom att läsa track list lokalt
3. Skapa manifest på de som matchar.

---

## 5) Downloader-implementation: robusthet och etikettloggning

### 5.1 Nätverksrobusthet

* retries med exponential backoff
* rate limiting (konfig)
* resume: om fil finns och checksum matchar → hoppa

### 5.2 Checksums

* SHA256 för varje `.vital`
* spara `checksums.tsv` i download_dir
* logga total “cohort hash” = hash(manifest + checksums) (detta blir er DOI-vänliga identitet)

### 5.3 Filstruktur

```
data/vitaldb/
  manifests/
    propofol_eeg_v1.jsonl
  vital_files/
    00001.vital
    00002.vital
  logs/
    ingest_run_YYYYMMDD.json
  checksums.tsv
```

---

## 6) Extract-steget (bara för att implementatören inte ska blanda in analys)

Efter download:

* extrahera bara det nödvändiga:

  * `BIS/EEG1_WAV` waveform (resample policy)
  * propofol infusion track
  * event/timestamps om tillgängligt
* QC-features: missingness, clipping, flatlines, artifacts (mask)
* skriv ut till HDF5/Parquet med:

  * sample_rate
  * start_time
  * channel metadata
  * provenance fields

**Viktigt:** Ingest ska vara empiriskt: inga NDT-tolkningar, bara extraktion och QC.

---

## 7) “Implementatörs-LLM” prompt (copy-paste)

Här är en färdig prompt du kan ge till implementatörs-LLM:

* Implementera en config-driven ingest pipeline för VitalDB med tre CLI: make_manifest, download, extract.
* Stöd två källor: (A) VitalDB API via `vitaldb` python library (`vitaldb.login`, `vitaldb.download`, `vitaldb.VitalFile(...).get_track_names()`), (B) PhysioNet mirror.
* Bygg manifest för “propofol present” och “EEG waveform exists” (required track `BIS/EEG1_WAV`), med fallbacklista för propofol-tracks.
* Alla hemligheter via env vars; inga credentials i repo.
* Download är resumable, rate-limited, retry-robust; spara SHA256 checksums.
* Logga provenance: config hash, manifest hash, package versions, counts, failed ids.
* Extrahera endast required tracks + QC mask och skriv HDF5/Parquet med metadata + provenance.
* Skriv en minimal README “How to obtain VitalDB data” inklusive DUA/credential notes och ett exempel-kommando per CLI.
* Använd enhetstester för (i) manifest building determinism, (ii) checksum caching, (iii) track existence check (mock).

Referenser: VitalDB python docs + bulk download guide + PhysioNet VitalDB dataset page. ([vitaldb.net][1])

---

## 8) En sak jag skulle låsa direkt (för att undvika att ni fastnar)

**Sätt “propofol present” (inte “propofol only”) som v1.**
“Only” är mycket lätt att göra fel p.g.a. dolda co-meds och varierande track names, och riskerar att ni spenderar veckor på cohort-renhet i stället för att få en fungerande ingest.

---

Om du vill kan jag även skriva en *konkret* YAML-template + pseudokod för varje CLI (utan att behöva se ert repo), men ovan borde räcka för att en implementatör ska kunna bygga en korrekt, reproducerbar downloader i ingest-lagret.

[1]: https://vitaldb.net/docs/?documentId=1j702xfDef7RNhfZANQQt7tkzzbin2FVe9GAnleDSyzI&utm_source=chatgpt.com "Vital File API - Docs : VitalDB"
[2]: https://physionet.org/content/vitaldb/?utm_source=chatgpt.com "VitalDB, a high-fidelity multi-parameter vital signs database ..."
[3]: https://vitaldb.net/docs/?documentId=1jLTcF4JYbRTuSM2mZeTMmvzxMmrqUjEEp6p02cFEs_Q&utm_source=chatgpt.com "Endpoints - Docs : VitalDB"
[4]: https://vitaldb.net/docs/?documentId=16oyZ4L2zC1oCauMtxG6DiDDN6eqPysRx4L6J_hfTB_U&utm_source=chatgpt.com "File Download Guide - Docs : VitalDB"
[5]: https://www.physionet.org/content/vitaldb/1.0.0/vital_files/0002.vital?utm_source=chatgpt.com "VitalDB, a high-fidelity multi-parameter vital signs database ..."
[6]: https://www.researchgate.net/publication/366642755_PulseDB_A_Large_Cleaned_Dataset_Based_on_MIMIC-III_and_VitalDB_for_Benchmarking_Cuff-less_Blood_Pressure_Estimation_Methods?utm_source=chatgpt.com "(PDF) PulseDB: A large, cleaned dataset based on MIMIC ..."
[7]: https://vitaldb.net/boards/?article=707&id=bugreport&utm_source=chatgpt.com "Issue Accessing .vital File Data via vitaldb Python Library #707"
