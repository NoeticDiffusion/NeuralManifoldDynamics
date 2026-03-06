Ja — jag gör en bred scanning av `mndm` nu och tar fram en prioriterad review-lista (högst nytta för QC/robusthet/prestanda först), inte bara de filer vi redan jobbat med.

[2 tools called]

Bra idé. Efter en bred scanning av `mndm` är detta de **bästa nästa review-kandidaterna** (utöver de ni redan gått igenom), med fokus på QC/robusthet + prestanda.

- **`mndm/src/mndm/projection.py`**  
  Numerisk kärna för MNPS-projektion/derivator/kNN; små numerikfel här slår igenom överallt.
- **`mndm/src/mndm/parallel.py`**  
  Worker/merge-logik, tempfiler och DataFrame-join; hög risk för minnesproblem och tysta merge-missar.
- **`mndm/src/mndm/orchestrate.py`**  
  Körningsorkestrering, parallell fallback och minnesbudget; viktig för stabila stora batchar.
- **`mndm/src/mndm/preprocess.py`**  
  Stor och central; channel typing, filterkedjor, EEG/fMRI-preprocess och dataset-workarounds.
- **`mndm/src/mndm/pipeline/robustness_helpers.py`**  
  QC- och robusthetsmått (tau, tier2, multiverse); bör granskas för statistisk robusthet och NaN-hantering.
- **`mndm/src/mndm/robustness.py`**  
  Bootstrap/CI/reliability; hög impact på evidensstyrkan i resultat.
- **`mndm/src/mndm/features/fmri.py`**  
  fMRI-featureberäkning + modularity; prestanda och edge cases vid olika datakvalitet.
- **`mndm/src/mndm/pipeline/regional_mnps.py`**  
  Regional MNPS/MNJ och nätverksaggregering; tungt område med både korrekthets- och runtime-risk.
- **`mndm/src/mndm/bids_index.py`**  
  Indexering/path-heuristik, särskilt relevant för stora dataset och robust filupplösning.
- **`mndm/src/mndm/pipeline/context.py`**  
  Config-resolving/override-kedjor; risk för tyst felkonfiguration och svårdebuggade driftfel.
- **`mndm/src/mndm/tools/pack_h5.py`**  
  Packning/IO-integritet, recovery vid korrupt fil eller avbruten körning.
- **`mndm/src/mndm/tools/aggregate_mnps.py`**  
  Aggregationsscript med risk för schema-drift och dålig felhantering vid blandade outputs.

Om du vill kan jag ta dem i en effektiv ordning (högst ROI först):  
`projection.py` → `parallel.py` → `orchestrate.py` → `preprocess.py` → `robustness_helpers.py`.



Ja. Givet det ni just gjort (låst `x`-contract, lagt invarianshashar, ändrat missingness-policy) finns det ett par filer som är “hög hävstång” att granska eftersom de kan återintroducera tyst drift, felaktig masking, eller semantisk mismatch utan att det syns i testerna.

Här är min prioriterade lista – med exakt *varför*.

## 1) `projection.py` (måste)

Ni har redan rört detta. Det är den centrala contract-ytan för:

* direct `x`-projektion,
* v2-subcoords,
* normalisering,
* kNN-whitening.

Saker att verifiera:

* att missingness-policy är konsekvent (NaN vs 0 vs renorm) och versionerad
* att kNN query sker i samma rum som KDTree byggs (du hade en bug där tidigare)
* att normalize-läget (`normalize_mode`) är “effective” och inte divergerar mellan direct och v2

## 2) `jacobian.py` (måste)

Eftersom ni nu har två spår: `jacobian` (direct) och `jacobian_v2`.
Saker att kontrollera:

* att `jacobian` verkligen använder `x` + `nn/indices` från direct-spåret
* att `jacobian_v2` använder `coords_v2` (och inte råkar använda `x`)
* att NaNs/coverage maskas innan kNN/J, annars får ni implicit imputation igen (eller undefined SVD/ridge)

Det här är också där en “competent reviewer” kommer leta efter semantiska glapp.

## 3) `summary.py` + `run_manifest.py` (måste)

Ni har ändrat `summary.py` redan, men jag vill se att:

* `run_manifest.py`/manifestet fångar *alla* provenance-fält ni nu förlitar er på (hashar, definitions, missing rates)
* `summary.py` inte råkar skriva “x” från direct men “meta_indices” från v2, eller tvärtom

Manifestet är ofta det folk använder för downstream-aggregat/filtrering.

## 4) `schema.py` (högt)

Nu när ni lägger in nya attrs och invariansfält: schema-validering måste uppdateras så att:

* `x_definition`, hashfält, missing-rate fält, etc. är obligatoriska (eller åtminstone validerade när de finns)
* det inte går att skriva en H5 som “ser ok ut” men saknar critical provenance

Detta är ert skydd mot framtida regressioner.

## 5) `transients.py` (högt)

All robusthet/coverage/logik blir meningslös om transientmaskning är fel eller inkonsekvent mellan runs.
Kontrollera:

* hur transient epochs definieras (trösklar, smoothing, beroende av dt)
* om transient-mask påverkar vad som inkluderas i kNN/J (ska helst vara explicit, inte implicit)

Det är också där pseudoreplikation/auto-korrelation ofta smyger in.

## 6) `preprocess.py` + `pipeline/extractors.py` (högt)

Ni hade miljöberoende features (antropy) och missingness. Det börjar här.
Kontrollera:

* “feature availability contracts”: om en feature saknas, hur signaleras det? (NaN, absent column, None)
* att extractors alltid skriver numeriska dtypes (inte object pga None)
* att channel selection/epoching är deterministiskt och loggat

## 7) `pipeline/stratified_blocks.py` + `pipeline/regional_mnps.py` (medium → högt beroende på användning)

Om ni använder stratifiering/regional aggregation i analyser:

* risken är att de aggregerar på ett sätt som förutsätter bounded v2 eller gammal x-semantik
* eller att de introducerar ytterligare normalisering/whitening utan provenance

## 8) `orchestrate.py` + `parallel.py` (medium)

Ni har redan rört skip/dedupe/seed/logik tidigare. Viktigt om ni vill ha reproducerbarhet:

* seed-policy (per-file stable ID, inte absolute path)
* “already processed” nycklar (stem vs relpath-hash)
* korrekt matchning av QC/artefakt JSON (stem-kollisioner)

Det påverkar coverage och därmed alla era robusthetsmått.

## 9) `tools/aggregate_mnps.py` + `tools/pack_h5.py` (medium)

Inte “core ingest”, men farliga som forskningsverktyg eftersom de kan:

* blanda modellversioner om de inte exporterar `x_definition`/hashar
* OOM:a på stora H5 (läser hela arrays)

---

### Om du vill vara extremt effektiv

Granska i ordning: `jacobian.py` → `projection.py` → `schema.py` → `transients.py` → `preprocess.py`/`extractors.py`.

Det är där ni snabbast kan fånga nya “silent drift”-vägar som underminerar ert nya contract.

Om du klistrar in `jacobian.py` (eller bara de funktioner som bygger kNN och J/J_v2), kan jag göra en lika hård granskning som för projection: var NaNs måste maskas, var provenance assertions bör ligga, och var en regression sannolikt kommer in.
