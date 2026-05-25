Jag har läst igenom idé-noten och gått igenom hur normaliseringen faktiskt fungerar i `mndm/src/mndm`. Här är min reflektion strukturerad enligt rubriken i noten, men hårt förankrad i vad koden gör idag.

## 1. Vad gör pipelinen faktiskt idag?

Innan jag spekulerar – det är värt att slå fast nuläget exakt, för det är *inte* lika illa som notens "subject-wise z-scoring" antyder, men inte lika bra som man kan tro:

- **Skalpolicy per feature** definieras i config, t.ex. i `mndm/config/config_ingest_common_eeg.yaml`:

```122:135:mndm/config/config_ingest_common_eeg.yaml
  feature_standardization:
    default: ["robust_z", "clip"]
    eeg_delta: ["log10", "robust_z", "clip"]
    eeg_theta: ["log10", "robust_z", "clip"]
    eeg_alpha: ["log10", "robust_z", "clip"]
    eeg_beta: ["log10", "robust_z", "clip"]
    eeg_gamma: ["log10", "robust_z", "clip"]
    eeg_alpha_theta: ["robust_z", "clip"]
    eeg_beta_alpha: ["robust_z", "clip"]
    eeg_sample_entropy: ["robust_z", "clip"]
    eeg_permutation_entropy: ["robust_z", "clip"]
    eeg_spectral_entropy: ["robust_z", "clip"]
    eeg_hjorth_complexity: ["robust_z", "clip"]
    eeg_hjorth_mobility: ["robust_z", "clip"]
```

- **`robust_z` implementeras som median + 1.4826·MAD** på den DataFrame som skickas in:

```370:374:mndm/src/mndm/projection.py
            elif step_str in ("robust_z", "robust"):
                t_finite = transformed[mask]
                t_median = np.nanmedian(t_finite)
                t_mad = np.nanmedian(np.abs(t_finite - t_median)) * 1.4826
                transformed[mask] = (t_finite - t_median) / (t_mad + 1e-9)
```

  Anropsvägen är `project_features → _normalize_used_columns`, och DataFrame som skickas in är features för **en fil/session** (en subjects epochs). I praktiken är detta alltså **per-session robust z**, vilket är funktionellt nära det noten kritiserar som "per-subject z-scoring".

- **Råa features bevaras redan** i exporten. `build_feature_export_bundle` skapar **två parallella ytor** plus metadata:

```304:310:mndm/src/mndm/projection.py
    return {
        "raw_values": raw_values,
        "raw_names": list(feature_cols),
        "robust_z_values": robust_z_values,
        "robust_z_names": list(feature_cols),
        "metadata": normalized_metadata,
    }
```

  Och H5-schemat dokumenterar det:

```28:39:mndm/src/mndm/reporting/schema_docs.md
- `/features_raw/values` – `float32[T, K]`
  - Raw per-epoch feature matrix in original scale.
- `/features_raw/names` – `str[K]`
  - Feature names aligned to `/features_raw/values`.
- `/features_raw/metadata/*` – arrays of length `K`
  - Machine-readable feature provenance and usage flags.
- `/features_robust_z/values` – `float32[T, K]`
  - Strict robust-z version of the exported feature matrix.
- `/features_robust_z/names` – `str[K]`
  - Feature names aligned to `/features_robust_z/values`.
- `/features_robust_z/metadata/*` – arrays of length `K`
  - Same per-feature metadata layout as `/features_raw/metadata/*`.
```

- **Men koordinaterna** – `mnps_3d = [m, d, e]` och `coords_9d` – beräknas **enbart från den standardiserade ytan**. Det finns ingen `coords_3d_raw_anchored` eller `coords_9d_batch_anchored` idag. Och därtill: `robust_z_center`/`robust_z_scale` per feature lagras visserligen som metadata (rad 222–223, 275–276 i `projection.py`), men beräknas inom **den enskilda H5:n**, inte mot någon delad referens.

Så den exakta diagnosen är: **rå nivå är räddad på feature-nivå, men inte på koordinat-nivå, och anchor är per session.** Det är där det biologiska signalförlustproblemet biter.

## 2. Diagnos: varför per-session-z är farligt just för NDM

Per-session median/MAD-skalning gör tre saker som rimmar dåligt med NDM:s mätkontrakt:

1. **Subjects-mean och subjects-MAD tvingas mot 0 respektive ≈1** per axel. Om koma-patienter med dålig prognos genuint har lägre permutation-entropi eller lägre Hjorth-mobility *i absolut värde*, så fördelas signalen ut över koordinataxeln så att varje subject ändå fyller sitt eget rörelseutrymme.
2. **Dynamiska kvantiteter blir skalfria på fel sätt.** Volym, hastighet, Jacobianer, reachability-koner, anisotropi — alla skalar med kovariansens egenvärden. Om dispersionen normaliseras inom subject blir trajectory-volymen approximativt invariant mellan grupper *per konstruktion*. Du kan inte längre falsifiera "poor-outcome har mindre trajectory volume" från koordinaterna — bara från råa features.
3. **Rigiditet och attraktor-djup försvinner.** Om en patient sitter fast i en macrostate (låg varians, låg entropi) så får just denna patient *större* z-score-utslag på små förändringar, eftersom MAD är liten. Det är raka motsatsen till vad teorin förutsäger ska synas i koordinaterna.

Detta är inte ett implementationsfel — det är ett designval som var rimligt när målet var **inom-subjekt-geometri** (Jacobianer, lokal anisotropi, reachability inom en trajektoria). Det är fel val när målet är **mellan-grupp-jämförelser** av absoluta dynamiska egenskaper, vilket är hela grunden för coma-artikeln och CPC-stratifieringen.

## 3. Strategier – jämförelse

| Strategi | Bevarar absoluta gruppskillnader? | Inom-subject-geometri OK? | Tvärdataset? | Läckage-risk | Implementations­kostnad mot nuvarande kod |
|---|---|---|---|---|---|
| **Per-session robust_z** (nuvarande default) | Nej | Ja | Nej | Ingen | – (baseline) |
| **Per-session robust_z + bevara raw + metadata med center/scale** | Indirekt (kan rekonstrueras post-hoc) | Ja | Begränsad | Ingen | Klart för features (✓), saknas för koordinater |
| **Batch-anchored robust_z** (median/MAD över hela analysbatchen) | Ja | Mest, men subject-skala bidrar lite | Inom dataset: ja. Mellan: nej | Ingen om anchor inte beror på outcome-label | Måste tvåpass-passa: först aggregera, sen projicera |
| **Controls-only anchor** (median/MAD bara över kontrollgrupp / CPC1, sen applicerat på alla) | Ja, och **gruppskillnad bevaras explicit** | Ja | Inom dataset | **Inget outcome-läckage** om "controls" definieras pre-hoc (t.ex. CPC=1) men man bör reportera om subgrupp-definition är hypotesnära | Tvåpass; måste hantera när "controls" saknas i ett dataset |
| **Fold-fitted anchor** (median/MAD på train-fold endast) | Ja inom CV-protokoll | Ja | Inom dataset | Inget om man verkligen fryser anchor från train | Kräver att projection.py får inta en extern anchor-tabell |
| **Atlas-anchor** (frusen referens från extern adult-rest-cohort, sleep-cohort, etc.) | Ja, **och cross-dataset** | Ja | Ja | Site/montage/device-confound — måste rapporteras | Stor — behöver atlas-byggare, version och hash |
| **Quantile / rank-transform per session** | Förstör absoluta amplituder mer än robust_z | Ja, men gör all variation skalfri | Nej | Ingen | Liten |
| **Median/IQR per session** | Som robust_z men något mer outlier-tolerant | Ja | Nej | Ingen | Trivial |
| **Råa features → projektion utan skalning** | Ja | Trasigt om features har olika enheter (μV² vs entropi) | Ja om enheter är fixa | Ingen | Måste lösa enhets-vägning |
| **Subject-relative + atlas-anchored dual export** | Båda lagrade, läsaren väljer | Ja på den första | Ja på den andra | Hanterbart | Måttlig — duplicerar koordinat-yta |
| **Covariate-residualized anchor** (residualisera ålder/site/device innan anchor) | Ja för signal, **förstör om confound är hypotesnära** | Ja | Ja om residualmodell är fryst | Stor om confound är kopplad till outcome (t.ex. ålder vs prognos) | Måttlig |

## 4. Rekommenderat default för NeuralManifoldDynamics

Eftersom mätkontraktet redan är versionerat och dubbel-yta finns på feature-nivå, är det rena steget att **lyfta dual-yta-idén till koordinat-nivå** och göra anchor-policy *explicit och versionerad*. Konkret:

**Primär koordinatyta:** `coords_3d_batch_anchored` + `coords_9d_batch_anchored`
- Skalas via en **frusen anchor-tabell** `feature_anchors` (median, MAD) beräknad över **hela analysbatchen** för respektive modalitet (EEG/fMRI/iEEG separat), exklusive subject med kandidat-outcome om coma/clinical-kontrast är hypotes — alternativt **CPC=1-only** för coma-fallet.
- Det är denna yta som drivs in i statistiken: trajectory volume, Jacobiantråden, reachability, gruppjämförelser.

**Sekundär koordinatyta:** `coords_3d_subject_anchored` + `coords_9d_subject_anchored`
- Det vi har idag. Den behålls för inom-subject-geometri, för derivat- och Jacobian-skattning där MAD ≈ 1 är numeriskt fördelaktigt, och som diagnostik.

**Tertiär (när tillgänglig):** `coords_*_atlas_anchored`
- Mot en frusen adult-rest-EEG-referens (kanske byggd från en av de friska kohorterna i repo:t, t.ex. ds004796 eyes-closed rest, eller motsvarande för fMRI). Med versionshash. Endast för tvärdataset-jämförelser och måste alltid rapporteras med site/montage-confound-flagg.

**Råa features** behålls obegränsat på sin nuvarande plats (`/features_raw/values`). Det är den enda yta som tål alla framtida re-anchor-beslut.

Anchor-valet ska *inte* gömmas i en kommentar. Det ska in i payload-attrs och i manifest-hashen, så att två filer med samma `coords_9d_subject_anchored` men olika `coords_9d_batch_anchored.anchor_hash` är uppenbart inte jämförbara.

## 5. Sensitivitetsanalyser som bör krävas

För varje "serious paper" (coma, psykos, sömn, propofol, dementia):

1. **Anchor-swap-test.** Rapportera huvudresultatet (t.ex. "poor-outcome har lägre `e_e`") på alla tre anchor-ytor. Om det bara finns i `subject_anchored` är det inom-subject-geometri och inte en gruppskillnad.
2. **Drop-controls-anchor-test.** Räkna om controls-only anchor med 50 % bootstrap av controls. Variationen i effektstorlek bör vara liten.
3. **Permuterad anchor.** Slumpmässigt subset av batchen som anchor – konfidensintervall.
4. **MAD vs IQR vs std.** Visa att resultatet inte beror på robust-skala-valet.
5. **Klippnivå-sensitivitet.** `clip_threshold` 4, 6, 9. Att log10 + clip flyttar massa mellan extremvärden är reellt; klippet är osynligt idag i `projection.py:382`.
6. **Site/device-stratifiering.** Inom coma-kohorten: är gruppskillnaden i `coords_9d_batch_anchored` lika stark inom ett site som mellan sites?
7. **Subject-mean-vs-feature-mean.** Plotta `raw_abs_median` mellan grupper innan någon skalning. Om den syns där men inte i koordinaterna har skalningen ätit upp signalen.

Den enklaste lackmusprovet: gör en figur där samma metric beräknas på de tre anchor-ytorna i en panel. Om panelerna ser kvalitativt olika ut är anchor-valet din primärfynd.

## 6. Föreslaget H5/payload-schema (utbyggnad av befintligt)

Befintligt:
- `/features_raw/{values,names,metadata/*}`
- `/features_robust_z/{values,names,metadata/*}`
- `/mnps_3d` (en yta)
- `/coords_9d/{values,names}` (en yta)

Förslag (icke-bakåtkompat-brytande – nya nycklar):

```
/feature_anchors/
    spec/                       # frusen anchor-definition (yaml-serialiserad)
        anchor_id : str         # t.ex. "icare_v2.1_cpc1_pre2026"
        anchor_hash : str       # sha256 över definitionsblock
        anchor_source : str     # "controls_only" | "all_subjects" | "atlas:adult_rest_v1"
        cohort_filter : str     # mänskligt läsbart villkor
    per_feature/
        feature_name : str[K]
        center : float32[K]     # batch-anchor median
        scale  : float32[K]     # batch-anchor 1.4826·MAD
        n_used : int32[K]       # finite samples som anchor byggde på

/coords_3d_subject_anchored/{values,names}      # = nuvarande mnps_3d
/coords_3d_batch_anchored/{values,names}        # NY (primär)
/coords_3d_atlas_anchored/{values,names}        # NY (frivillig, om atlas_id satt)

/coords_9d_subject_anchored/{values,names}      # = nuvarande coords_9d
/coords_9d_batch_anchored/{values,names}        # NY (primär)
/coords_9d_atlas_anchored/{values,names}        # NY (frivillig)

attrs på roten:
    primary_coordinate_layer : str  # "batch_anchored" by default
    anchor_id                : str
    anchor_hash              : str
```

Implementationsmässigt är detta en *liten* förändring i `projection.py`:

- `_normalize_used_columns` får ta in en valfri `external_anchor: Mapping[str, tuple[center, scale]]`. Om satt används den istället för median/MAD beräknad lokalt.
- Anchor-tabellen byggs i ett separat förskede (eget CLI-subcommand `anchors-fit`) som läser alla `features_raw/values` från ett kohort-set och skriver ut en versionerad YAML/parquet.
- `build_feature_export_bundle` får returnera båda standardiserade ytorna när anchor finns: en `subject_anchored` och en `batch_anchored`. Båda får sin egen `metadata.projection_transform_steps` så provenance är spårbar.

Det är ungefär en eftermiddags arbete plus tester, inte en arkitekturmigration.

## 7. Claim-discipline

**Etablerade statistiska fakta**
- Median/MAD är en konsistent skattare av lokalisering och skala och är robust mot outliers; 1.4826·MAD ≈ σ under normalitet. Detta är inte ifrågasatt.
- Per-subjekt z-skalning förstör absoluta amplitud- och varianskillnader mellan grupper. Detta följer triviellt av att en affin transformation per subject mappar varje subjects (mean, std) till (0, 1).
- Trajectory-baserade dynamiska kvantiteter (volym, hastighet, kovariansens spår) är skalkänsliga.

**Plausibla metod­val**
- Batch-anchored MAD är ett bra default för inom-dataset kliniska kontraster. Skälet är att den preserverar gruppskillnader utan att kräva en extern referens. Risken är att om batchen är obalanserad i outcome så drar majoriteten anchor-positionen.
- Controls-only anchor (i coma-fallet: CPC=1) är defensivt rimligt, eftersom det gör implicit antagandet "friska/återhämtade patienter definierar nollpunkten" till ett explicit antagande som kan ifrågasättas.
- Att behålla `subject_anchored` som sekundär yta är klokt för Jacobian-skattning där lokala skalor måste vara jämförbara över subjects.

**Spekulativa utvidgningar**
- En frusen `atlas:adult_rest_v1`-anchor skulle göra `coords_9d` cross-dataset-meningsfulla. Men det förutsätter att site/montage/device/refer-strategy-confounds är kvantifierade. Bör testas först på ett konstruerat fall där samma subjects spelats in på två system.
- Covariate-residualisering (ålder, site) innan anchor är teoretiskt fint men i coma-fallet potentiellt katastrof: ålder och outcome är korrelerade. Då skulle residualisering av ålder *radera* en del av signalen. Använd inte residualisering på variabler som kan tänkas mediera outcome.
- Quantile/rank-transformer per session är en "starkare" subject-z – ännu mer förstörande för absolut signal. Inkludera inte som default.

**Failure-modes som måste rapporteras**
- Om `n_used` för en feature i anchor-tabellen är liten (t.ex. <100 epochs över hela kohorten), är center/scale instabila och man får inte använda anchor för den feature:n.
- Om batch-anchor används på en outcome-obalanserad kohort (90 % poor-outcome) drar anchor mot poor-outcome och *gör* gruppskillnaden mindre.
- Site/device-skift inom samma kohort kan ge falska gruppskillnader om grupper är site-sneda. Måste sanity-checkas med site-stratifierad bootstrap.
- Atlas-anchor från ett dataset med annan referensstrategi (t.ex. mastoid vs average reference) ger systematiskt skift i absolut amplitud per kanal — ALL atlas-anchored jämförelse måste villkoras på matchad reref.

## 8. Konkret rekommendation – primär vs sekundär

- **Primär koordinatyta** för publicerad analys: `coords_*_batch_anchored` med `anchor_source = "controls_only"` när en pre-hoc-definierad kontrollgrupp finns (t.ex. CPC=1 för coma), annars `"all_subjects"`. Anchor versioneras och hashas in i manifestet.
- **Sekundär** för inom-subject-geometri (Jacobianer, lokala reachability-koner, derivat-baserade transienter): `coords_*_subject_anchored`.
- **Diagnostisk** för cross-dataset-utforskning, *aldrig* för primära påståenden i ett enskilt papper: `coords_*_atlas_anchored`.
- **Råa features** är alltid den slutgiltiga sanningen för någon som vill räkna om allt. Behåll dem oförändrade.

Det innebär konkret att kontraktet "vad mäter vi?" får en explicit `primary_coordinate_layer`-flagg. Det stänger en av de svagaste interpretativa läckorna i nuvarande pipeline utan att förkasta något befintligt.

---

Om du vill kan jag som nästa steg:
1. Skriva en kort design-doc i `project/ideas/` som föreslår exakt vilka funktioner i `projection.py` och `schema.py` som behöver ändras, med signaturer och en migrationssekvens som inte bryter befintliga H5-filer, **eller**
2. Skissa CLI:n `mndm anchors-fit` och hur den lagrar `feature_anchors`-bunden YAML/parquet med hash, **eller**
3. Göra en liten experiment-plan: ta ett befintligt I-CARE-batch-H5, räkna fram `coords_9d_batch_anchored` post-hoc från `/features_raw/values`, och jämför `e_e`-gruppskillnad mellan CPC1 och CPC5 på de två ytorna – som första empiriska sanity-check innan man rör pipelinen.

Säg till vilket du föredrar.

# Fråga om IQR istället för MAD:

Bra fråga. Korta svaret: **nej, ingen avgörande fördel**, men det finns en handfull specifika små vinster med IQR och några specifika nackdelar — och i NDM-pipelinen är skalskattaren faktiskt inte den mest läckagebenägna designbeslutet. Här är resonemanget.

## 1. Vad skiljer dem matematiskt

Båda är robusta skalskattare som under Gaussianitet ger ett konsistent σ-värde med en konstant faktor:

- MAD: \( 1.4826 \cdot \mathrm{median}|x - \mathrm{median}(x)| \approx \sigma \)
- IQR: \( (Q_{75} - Q_{25}) / 1.349 \approx \sigma \) (eftersom \( 2 \Phi^{-1}(0.75) = 1.349 \))

På symmetriska, tunntailade data är de **i praktiken utbytbara** — de skiljer sig på 2–3:e decimalen. Det betyder att för NDM:s post-log10-bandpowers, entropier, Hjorth — alltså i stort sett alla features som anchor-tabellen byggs på — kommer skillnaden i `scale` att vara försumbar.

Skillnaderna dyker upp i tre specifika regimer.

## 2. Där IQR har en reell fördel

**a) Decomposerbarhet och asymmetri-diagnostik.** IQR delar sig naturligt i `Q75 − Q50` och `Q50 − Q25`. Du kan **gratis** spara båda i `feature_anchors/per_feature/` och få en feature-by-feature-skevhetsmått som biprodukt:

```
asymmetry = (Q75 - Q50) / (Q50 - Q25)
```

Värden ≠ 1 säger "denna feature är sned på batch-nivå". MAD ger inte detta utan extra beräkning. För clinical-kontraster är det värdefullt eftersom du då direkt ser om en grupp-effekt sitter i ena halvan av distributionen, vilket är den vanliga signaturen i koma-data (CPC5 trycker `e_e` mot låga värden men knappt mot höga).

**b) Klinisk konvention och granskbarhet.** Kliniska tabeller rapporterar nästan alltid `median [IQR]`. Om `anchor_scale = IQR` är `robust_z_scale`-kolumnen direkt jämförbar med vad reviewers förväntar sig se. MAD är välkänd i signalbehandling men ovan utanför.

**c) Beteende vid bimodal anchor-population.** Det här är kanske den enda *metodologiska* poängen som matter för coma-kontrasten. Om batch-anchor byggs på en 50/50-blandning av CPC1 och CPC5 (vilket är planen i `031_20260518_balanced_next140_cpc15_download.md`), så ger MAD respektive IQR olika skalor:

- MAD ≈ halva avståndet mellan toppar — efter skalning hamnar modena vid \( z \approx \pm 1 \).
- IQR ≈ hela avståndet mellan toppar — efter skalning hamnar modena vid \( z \approx \pm 0.5 \).

Båda **bevarar signalen**, men IQR ger en enhet som motsvarar "hela gruppskillnaden" snarare än "halva". Det gör tolkningen av effektstorlek i z-enheter mer intuitiv för en bimodal anchor.

## 3. Där MAD har en reell fördel

**a) Breakdown point.** MAD har 50 % breakdown, IQR har 25 %. För artefakttunga modaliteter (rå EEG före QC, fMRI med spike-frames) är detta en defensiv kvalitet. För NDM:s anchor-fit byggs anchor på `features_raw` *efter* `qc_ok_*`-filtrering, så contamination är realistiskt under 5–10 %. Breakdown-argumentet är därför mest teoretiskt här, men det är inte noll.

**b) Befintlig kod.** `ROBUST_MAD_TO_SIGMA = 1.4826` är hårdkodad i `mndm/src/mndm/projection.py:39` och `_robust_center_and_scale` (rad 138–147) använder MAD i två oberoende vägar. Migration är en eftermiddags arbete, inte gratis.

**c) Klipptröskeln är kalibrerad för MAD.** `clip_threshold: 6.0` i config tolkas idag som ±6 σ-ekvivalenter (eftersom inputen är `(x − median) / (1.4826·MAD)`). Med IQR/1.349 hamnar du på samma σ-skala om du gör konstantnormeringen rätt, men det är ett ställe där en migration kan slarvas och tysta bort signal genom dålig klipp-kalibrering. Värd att uppmärksamma.

## 4. Vad som faktiskt är mer high-leverage än MAD vs IQR

Det här är poängen jag vill betona: i diskussionen om batch-anchor är **skalskattaren** ett tredje- eller fjärde-ordningsbeslut. Ordningen ser ut så här i påverkan på huvudfyndet:

1. **Vad anchor är byggt på.** All-subjects vs controls-only (t.ex. CPC=1) vs balanced-CPC1+CPC5. Detta kan vända tecken på en gruppskillnad.
2. **Site/montage-konfunder.** Anchor över datasets med olika reref ger systematisk skift.
3. **Klipp-policyn** efter skalning. `clip_threshold = 6` kan tysta riktiga extrema-attractor-states.
4. **Skalskattaren (MAD vs IQR vs Qn vs std).** Förändrar effektstorlek med ≲ 10 % på post-log-data.

Att låta beslut #1–3 vara fastlagda i config och versionerade i anchor-hashen är viktigare än att välja "rätt" mellan MAD och IQR.

## 5. Om man ändå vill ha bättre än både MAD och IQR

För fullständighetens skull: om effektivitet matter (mindre N → mer brusig anchor → mer brusiga koordinater), så är varken MAD eller IQR i topp. Rousseeuw & Croux 1993:

| Skattare | Breakdown | Asymptotisk effektivitet vs std |
|---|---|---|
| std | 0 % | 100 % |
| IQR / 1.349 | 25 % | ≈ 37 % |
| MAD · 1.4826 | 50 % | ≈ 37 % |
| Sn | 50 % | ≈ 58 % |
| Qn | 50 % | ≈ 82 % |

**Qn** är `c · {|x_i − x_j| : i < j}_(k)` där k är ungefär \( \binom{h}{2} \) med \( h = \lfloor n/2 \rfloor + 1 \). 82 % effektivitet **med** 50 % breakdown är en strikt bättre Pareto-punkt än både MAD och IQR. Det är dyrare beräkningsmässigt (O(n log n) med Croux-Rousseeuw-algoritmen, ≪ minuter på ~10⁶ epochs).

Om man ändå rör pipelinen är Qn det mer ambitiösa valet. För NDM:s storleksordning (≲ 5·10⁵ epochs i en typisk coma-batch) är det fullt görbart.

## 6. Min konkreta rekommendation

Ändra inte en hardkodad konstant från 1.4826 till 1/1.349. Istället:

- Gör `anchor_scale` till en config-parameter: `"mad" | "iqr" | "qn"`, defaultar till `"iqr"` *när man bygger anchor-tabellen*.
- I anchor-tabellen `feature_anchors/per_feature/`, lagra **alla** av: `median`, `mad`, `q25`, `q50`, `q75`, samt en valfri `qn`. Det är 5–6 floats per feature × ~50 features ≈ ingenting på disk, och post-hoc kan man räkna om koordinaterna under valfri skalskattare utan att röra `features_raw`.
- Lämna `robust_z` *inom* H5:n (per-session) som det är — där är breakdown-argumentet starkare och MAD är ett vettigt default.
- Klipp-tröskel uttrycks i σ-ekvivalenter och appliceras *efter* att skalskattaren skalats om till σ med rätt konstant (1.4826 för MAD, 1/1.349 för IQR), så `clip_threshold = 6.0` betyder samma sak oavsett val.

Då har du gjort skalvalet till ett **återbesökbart** beslut snarare än ett irreversibelt. Vilket är hela poängen med ett mätkontrakt: man får inte gissa fel en gång.

## 7. Claim-discipline

- **Etablerat:** MAD har 50 % breakdown, IQR har 25 %. Båda har ≈ 37 % asymptotisk effektivitet under Gauss. Qn dominerar både på effektivitet vid samma breakdown.
- **Plausibelt:** För NDM:s post-log/post-QC-features är MAD och IQR numeriskt nästan identiska; valet är ett kommunikationsbeslut snarare än ett statistiskt.
- **Spekulativt:** Att IQR ger bättre bimodal-tolkning för coma-anchor är intuitivt men inte empiriskt visat på just NDM-data. Värt att verifiera med en figur på balanced-140-batchen där `e_e` plottas under MAD- respektive IQR-anchor.
- **Failure mode:** Att byta från MAD till IQR i samma kod utan att även justera `clip_threshold`-tolkningen ger systematiskt annorlunda clipping och kan tystare bort eller släppa igenom andra extremvärden än tidigare. Det måste fångas i sensitivitetsanalysen "klipp-tröskel 4/6/9 × scale MAD/IQR/Qn".

Vill du att jag skriver ihop ett kort sensitivity-skript som tar en befintlig I-CARE-batch-H5, räknar fram batch-anchor under MAD, IQR och Qn separat på `/features_raw/values`, och plottar hur stor effektstorleken `e_e[CPC1] − e_e[CPC5]` blir under de tre skalvalen? Det skulle ge ett empiriskt svar på exakt din fråga för just det här datat, vilket är mer värt än mer abstrakt teori.