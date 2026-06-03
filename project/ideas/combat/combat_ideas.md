Jag har läst igenom ditt utkast (`normalization_anchor_subject_cohort_ideas.md`), basconfigen (`config_ingest_physionet_i-care_2_1.yaml` med det avstängda `normalization`-blocket) och `anchors.py`. Här är min läsning som forskningslead, organiserad kring den centrala frågan du ställer.

## 1. Det matematiska grundsvaret

**Strikt taget: nej.** Inom-dataset-harmonisering är ett *underbestämt* identifierbarhetsproblem. Du observerar

\[ X = f_{\text{biologi}}(\text{patient, tid}) + g_{\text{site}}(\text{sjukhus, hårdvara}) + \varepsilon \]

och du har bara *en* mätning per cell. Du kan inte separera \(f\) och \(g\) utan minst en av följande:

1. **Antagande** om strukturen hos \(g\) (t.ex. additiv + multiplikativ shift, ortogonal mot signalrum, glatt i tid).
2. **Referens** (kontroller, sham, baselineepoch, eller subgrupp som är fysiologiskt jämförbar).
3. **Strukturell invarians** (en feature-form som per konstruktion är okänslig för hårdvarusignaturen).

Allt vi kan göra inom dataset är att välja en av dessa och vara *explicita* om vad det innebär att förstöra. Allt annat är självbedrägeri.

## 2. Den specifika fällan i i-CARE

Detta är viktigt att se klart innan vi väljer metod. i-CARE är inte ett "matchat" multi-site-dataset:

- **Sjukhus är konfunderat med patient-mix.** Olika sjukhus har olika referralmönster, olika ICU-protokoll, olika andel tidig-vs-sen-arrival, olika andel TTM (targeted temperature management). Det ger en kausal struktur:

  `hospital -> EEG` (hårdvara) men också `hospital -> patientpopulation -> outcome -> EEG` (biologi).

- Om du kör ComBat med `batch_key: hospital` utan att skydda outcome/time/age, så raderar du *både* hårdvaran och den biologiska sjukhus-mix-signalen. Det är då du "förstör riktig data".

- ComBat utan covariater på i-CARE är ett klassiskt över-korrektionsscenario. Du måste *minst* skydda: `outcome (CPC)`, `time_bin (0-12/12-24/24-48/48-72h)`, `age`, eventuellt `TTM_status` om tillgängligt.

## 3. En lager-modell som är ärlig om vad varje steg kostar

Jag skulle ramma in det som **fyra lager**, från "garanterat oskadligt" till "kraftfullt men riskabelt":

### Lager 0 - Invariansfix gratis (bör göras alltid)

Saker som inte är "harmonisering" i statistisk mening, men som tar bort hårdvarusignatur per konstruktion:

- **Average reference** (du har det redan).
- **Spektrala kvoter och kvotbaserade index** (alfa/teta-ratio, band-power-fraktioner). En förstärkare med okänd skala försvinner i kvoten.
- **Spektralt exponent (1/f-slope) från FOOOF.** Hårdvarans broadband-vinst påverkar interceptet, inte slopen.
- **Sample entropy / DFA på z-scoreade segment.** Skalinvariant per konstruktion.
- **Ensembles per region** (du har också detta). Att medelvärdesbilda över `frontal`/`central`/... dämpar elektrod-idiosynkrasier.

Det här lagret är "matematiskt rent" - du tar bort en monoton hårdvaru-vinst utan ett enda batch-antagande.

### Lager 1 - Site-balanserad cohort-anchor (mycket billig utvidgning av det du har)

Din `_build_anchor_artifact` är *subject-balanced* (median över subjekt). Den är inte *site-balanced*. Om Hospital A bidrar med 60% av subjekten dras anchorn mot Hospital A.

Konkret tillägg: i `_build_anchor_artifact`, lägg ett mellanled:

1. Beräkna `subject_median_per_feature` (du gör detta).
2. Gruppera dessa per `site` -> `site_median_per_feature`.
3. Beräkna `center = median(site_medians)` och `scale = IQR(site_medians) / 1.349` (eller hierarkisk MAD).

Det här är **inte** harmonisering av data - det är bara att flytta referenspunkten så att ingen site dominerar. Inga datapunkter ändras, ingen biologi raderas, det är reversibelt och artikel-vänligt. Det är `Fas 0.5` i din egen plan, och jag tycker du ska sätta det före `combat`.

### Lager 2 - Skyddad ComBat / longComBat (pragmatiskt, ditt nuvarande Fas 1)

Här lägger jag tre konkreta råd ovanpå utkastet:

- **Använd longComBat (eller mixed-effects ComBat) snarare än vanlig ComBat.** i-CARE är longitudinell; varje patient har många timbins. Standard-ComBat behandlar dem som oberoende och underskattar within-subject-korrelation. `longCombat`-paketet (Beer 2020) hanterar detta korrekt.
- **Covariate-listan i din YAML är minimal.** `["group", "age", "sex"]` saknar de mest kritiska: `time_bin` och `outcome (CPC)`. Outcome måste in - annars rensar du bort sambandet hospital↔outcome som *är* en del av signalen.
- **Harmonisera per feature-familj, inte globalt.** Spektral-effekt, entropi, konnektivitet har olika skalor och olika site-känslighet. En global ComBat-fit över alla features blandar familjer och över-korrigerar de robusta.

### Lager 3 - Geometrisk omcentrering (Riemannian, forskningsgren)

Som du själv noterar: hög kostnad, svår att kommunicera, känslig för referenspolicy. Två konkreta varianter värda att hålla isär:

- **Per-subject re-centering** (varje subjekt -> identitet på kovariansmanifolden). Tar bort *all* between-subject-variation - bra för cross-session BCI, **dåligt här** eftersom du vill jämföra subjekt.
- **Per-site parallel transport** (transportera varje subjekt från site-medel till cohort-medel). Tar bort between-site men bevarar within-site-biologi. Detta är den "rätta" Riemann-varianten för din fråga, men implementationskostnaden är hög.

Lämna detta som isolerad gren, precis som du planerat.

## 4. Hur man *vet* att man inte förstört signal (kritiskt steg)

Detta är det viktigaste tillägget till din plan. Innan och efter varje harmoniseringssteg, kör tre probes:

1. **Site-prediktion från features.** Träna en enkel klassificerare (logreg/RF) på hospital från features. *Före*: hög accuracy (annars finns ingen site-effekt att rätta). *Efter*: nära chans.
2. **Outcome-separation (CPC good vs poor).** Mät samma probe för outcome. *Måste hålla i sig eller förbättras*. Om den faller är du i över-korrektion.
3. **Within-subject longitudinell signal.** Mät t.ex. spektral-exponent-trend under 0-72h per patient. *Måste vara stabil mellan rå och harmoniserad version*. Om within-subject-trender plattas ut har harmoniseringen ätit upp dynamiken.

Detta är operationaliseringen av "förstör inte riktig data". Utan dessa tre tal är harmoniseringen blint trolovad med antagandet att den fungerar.

## 5. Konkreta tillägg jag skulle göra till din befintliga plan

Med risk för att gå utanför scope för "fundera", här är de små men konkreta deltas till det du redan skrivit:

1. **Lägg till Fas 0.5: site-balanced anchor.** Det är ~30 rader i `_build_anchor_artifact`, inget data ändras, fullt reversibelt. Egen branch.
2. **Utvidga covariate-listan i YAML-skissen:**
   ```yaml
   normalization_covariates: [outcome_cpc, time_bin, group, age, sex, ttm_status]
   ```
3. **Lägg till `longitudinal: true` som flag** och dirigera till longComBat när satt.
4. **Lägg till en `validation` subsektion i normalization-konfigen** som specificerar de tre probesarna ovan - och låt summarize-steget rapportera dem.
5. **Notera explicit i dokumentet att Fas 0 (baseline) ska mäta `site_predictability` på rå data.** Om den är låg redan från början är hela harmoniseringsövningen onödig och du sparar dig en stor risk.

## 6. Det jag *inte* skulle göra

- Kör inte ComBat utan outcome-covariat på i-CARE. Det är den mest sannolika vägen till en publikation som senare rivs.
- Kör inte global feature-ComBat över alla familjer på en gång.
- Kör inte Riemann-omcentrering till identitet på subjektnivå om du vill jämföra subjekt.
- Använd inte `within_dataset` reference policy utan att även rapportera resultat under `frozen_external` om/när du kan koppla till en oberoende cohort - det är en av de starkaste sanity-checkarna mot leakage.

## 7. Sammanfattning av kategoriskillnader (per ditt eget research-discipline-rule)

- **Externt validerat:** ComBat, longComBat och Riemann-alignment har peer-reviewed evidens i neuroimaging/BCI - men deras *fördelar och fallgropar* är väl dokumenterade.
- **Internt validerat i ditt repo:** Subject-balanced robust anchor (din `anchors.py`).
- **Plausibel utvidgning:** Site-balanced anchor (Lager 1) - bygger direkt på en mekanism som redan finns.
- **Spekulativ:** Att inom-dataset-harmonisering på i-CARE kan bevara outcome-signalen utan explicit outcome-covariat. Bör behandlas som motbevisbar tills probes (avsnitt 4) visar annat.
- **Möjligt att förkasta tidigt:** Per-subject Riemann re-centering för cross-subject jämförelser - matematiskt fel ramverk för uppgiften.

Vill du att jag skriver detta som en utvidgning av `normalization_anchor_subject_cohort_ideas.md` (lägga till Fas 0.5, validerings-probes, kausal-noten om i-CARE), eller som en separat diarie-entry under `project/diary/` enligt din regelboks-konvention?