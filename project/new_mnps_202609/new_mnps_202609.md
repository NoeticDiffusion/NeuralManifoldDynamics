# Alternativ 9D-karta med fryst kanonisk MNPS 3D — implementeringsplan

Status: **explorativ plan, ej implementationsbeslut** (2026-09-25). Ingen ny
koordinat, Jacobian, HDF5-yta eller biologisk slutsats är kvalificerad genom
denna text.

## 1. Mätfråga och namn

Kan en separat, mer konstruktorienterad 9D-observationskarta ge tillförlitlig,
icke-redundant information om neural organisation och skattbar lokal dynamik,
samtidigt som den historiska `mnps_3d = [m,d,e]` och dess MNJ förblir numeriskt
jämförbara med tidigare körningar? Den efterföljande forskningsfrågan är vilka
*förregistrerade slutsatser* som är robusta mot valet av observationskarta.

`project/mnps_v3/mnps_v3_implementation_plan.md` använder redan **MNPS v3**
för en generation av dynamiska familjer ovanpå fryst 3D/9D. Den planen ändrar
inte kartan. Använd därför arbetsnamnet **`mnps9_alt_202609`** för denna
9D-kandidat. Beteckningen ”MNPS-9D v3” får bara användas efter ett separat
beslut om kartversion och namnmigrering. Förväxla inte kartversion,
programgeneration, estimatorversion och HDF5-schemaversion.

## 2. Beslutad gräns och kontraktspåverkan

Den kanoniska 3D-referensen, nuvarande `coords_9d` och deras historiska MNJ
förblir frysta. I aktiva EEG- och fMRI-referenskonfigurationer skapas 3D ur
nuvarande 9D med en fast viktad, kolumnnormaliserad projektion och
stödberoende renormalisering. Den exakta 3D-referensen omfattar därför mer än
namn och nominella vikter: rådataväg, feature-extraktion, transformations- och
ankarkontrakt, tidsfönster, saknade värden, täckningsregler, faktisk projektion,
derivata, MNJ-estimator och proveniens. Ändring av något av detta kräver en
separat kompatibilitetsbedömning. Befintlig `direct_features`-gren är **inte**
automatiskt numeriskt likvärdig med den aktiva `from_v2`-grenen.

Föreslagen observationsstruktur, endast som kandidat:

```text
Y ── befintlig, fryst väg ──> coords_9d (nuvarande) ──> mnps_3d [m,d,e]
  └─ ny, separat väg ───────> mnps9_alt_202609 (endast om kvalificerad)
```

Det nya 9D-objektet får inte skriva över `/coords_9d`, `/mnps_3d`, deras
ankrade motsvarigheter, `/jacobian` eller `/jacobian_9D`. Under pilotfasen
används en separat output root och en tydligt märkt forskningsartefakt; ett
eventuellt senare HDF5-namnområde kräver eget schema-, writer- och
readerbeslut. Avsaknad av stöd representeras som saknad/avvisad mätning med
orsak, aldrig som noll eller en tyst proxy. `computed` och vetenskapligt
`qualified` är skilda statusar.

Den nya kartan är en **parallell 9D-beskrivning**. Den får kallas
”stratifiering av kanonisk 3D” bara om en på förhand definierad och prövad
rekompositionsrelation faktiskt finns. En eventuell projektion från ny 9D
till *experimentell* 3D får ett eget namn och ersätter inte kanonisk 3D.

## 3. Kandidater att operationalisera, inte fastställda koordinater

| Familj | Kandidat och mätavsikt | Viktig prövning före användning |
| --- | --- | --- |
| `m` | kollektiv metastabilitet, t.ex. variation i regional fassynkroni | Fasskattning, frekvensband, CSD/leakage, antal regioner och fönsterlängd. |
| `m` | persistens eller mobilitet, t.ex. avklingning/autokorrelation | Tillräcklig tidshorisont, gap, långsam drift och överlapp med downstream-persistens. |
| `m` | omkonfiguration av regional aktivitet eller koppling | Definition av jämförelsefönster; undvik att smuggla in en derivata som sedan deriveras igen för MNJ. |
| `d` | integration–segregation | Välj en specificerad grafestimand; effektivitet, modularitet och participation är inte utbytbara. |
| `d` | lokal kontra distribuerad koppling | Regional identitet, referens/CSD, volymledning och kopplingsmått. |
| `d` | spatial eller representational dispersion | Topografi/RDM-definition, kanal- och parcellationskänslighet. |
| `e` | temporal irregularitet, t.ex. permutation entropy | Parametrar, antal sampel, brus och artefakter. |
| `e` | komprimerbar struktur, t.ex. normaliserad Lempel–Ziv | Binarisering, sekvenslängd och redundans mot entropi. |
| `e` | effektiv regional dimension | Kovariansrang, brus, shrinkage, kanalantal och fönsterlängd. |

Denna 3×3-indelning är en **begreppshypotes**, inte nio ortogonala variabler.
Varje kandidat behöver före datakontrast: formel, riktning, enhet, signaltyp,
regional konstruktion, tidsstöd, fönster/steg, normalisering, ankare,
saknadspolicy, kvalitetsvillkor, provenance och negativ kontroll. EEG, MEG och
fMRI får modalitetsspecifika realiseringar; gemensamt namn innebär inte
fysiologisk eller temporal ekvivalens. Saknad regional upplösning betyder att
den aktuella koordinaten är `unsupported` för den inspelningen.

Nuvarande `e_m` och dess källberoende fallback lämnas orörda i legacy-kartan.
För den alternativa kartan prövas kroppsliga/autonoma mått som separat
`anchor_state`/kontext, inte som en implicit neural 9D-koordinat. Gamla
gamma-, alpha- och Hjorth-features bevaras som mätbara legacy-features och
möjliga baslinjer; gamla resultat döps inte om till nya konstrukt.

## 4. Grindar i föreslagen ordning

### G0 — Frys och reproduktion av referensen

1. Inventera för valda referenskörningar den *upplösta* konfigurationen,
   schema, feature- och koordinatnamn, normalisering/ankare, vikter,
   supportpolicy, klocka/gap och MNJ-estimator. Läs faktiskt skrivna manifest
   och HDF5-metadata, inte enbart mallkonfiguration.
2. Frys en avgränsad referensmängd med EEG och, om stöd finns, fMRI; dokumentera
   source-, config- och output-hash. En ny körning av samma väg i samma
   kontrollerade miljö ska återge koordinater, giltighetsbeslut och Jacobianer
   enligt i förväg angiven numerisk tolerans. Historiska filer skrivs inte om.
3. Granska befintlig fallback från `from_v2` till `direct_features` när 9D
   saknas. För referensjämförelse måste effektiv väg vara explicit och
   lika; oväntad fallback får inte räknas som bevarad 3D.

**Utgång:** en signerad legacy-baslinje och en testbar definition av
”oförändrad 3D”. Om detta inte kan fastställas stannar jämförbarhetsanspråket.

### G1 — Mätdefinition och möjlighet per modalitet

1. Skriv ett kort kontrakt per kandidat med exakt operationell definition.
   Pröva om regioner, kanaler, kontinuerligt signalstöd och tidslängd finns i
   verkliga källor. CSD och regionala kanalgrupper är möjliggörare, inte bevis
   för giltig koppling eller anatomisk lokalisation.
2. Bestäm temporal skala innan extraktion. Persistens, grafestimat och
   kovariansrank kan kräva längre eller annat stöd än befintliga korta EEG-
   fönster. Definiera vad som händer vid partiella fönster, luckor,
   inspelningsgränser och ändrat kanalantal. Ingen interpolering över gap.
3. Förregistrera reliabilitet, tillåten saknadhet, toleranser för redundans
   och estimatorstöd utan att välja dem efter koma-, psykos- eller
   spindleffekt. Beslut om vilka kandidater som *kan* mätas föregår
   villkorskontraster.

**Utgång:** versionerat förslag med acceptans-/avvisningsregler per koordinat;
de nio platserna fylls inte med godtyckliga fallbackvärden för att nå 9D.

### G2 — Isolerad feature- och koordinatpilot

1. Implementera bakom ett explicit pilot-overlay, separat output root och
   oförändrat legacy-flöde. Om ny temporal sampling behövs extraheras den
   från verifierad råsignal. Återanvänd inte cachade sampel-features som bevis
   för en ny kontinuerlig signal eller ny tidsbas.
2. Kör först syntetiska fall: konstant signal, brus, kända oscillerande och
   nätverksmönster, duplicerade regioner, saknade kanaler, icke-finita värden,
   gap, oregelbundna tidssteg och otillräcklig längd. Kontrollera känd riktning
   där den är matematiskt definierad och explicit avvisning annars.
3. Gör en begränsad realdatapilot utan utfallsstyrt urval. Spara per fönster
   käll-/kanalidentitet, nominellt och faktiskt tidsstöd, gapstatus,
   featureversion, transformationsordning, ankare, support, validity och
   kvalificeringsstatus. Matcha endast faktiskt gemensamma fönster med legacy.

**Utgång:** reproducerbara, namngivna pilotmått. Serialisering räcker inte
för vetenskaplig kvalificering.

### G3 — Kartans mätkvalitet före MNJ

Pröva test–retest inom relevant tidskala, känslighet för kanaluppsättning,
referens/CSD, regionindelning och ankare; korrelation och effektiv rang mellan
koordinater; beroende av gemensamt spektralt råmaterial; samt stabilitet under
fördefinierade signal- och etikettfria negativa kontroller. Jämför mot
legacy-features och rimliga enklare baslinjer. En ”mer ortogonal” karta ska
styrkas med observerad icke-redundans och reliabilitet, inte med koordinatnamn
eller ett lägre p-värde. Dokumentera även mått som faller bort.

**Utgång:** ett fryst kandidatset eller `METHOD-LIMITED`/`FAIL` för 9D om
fullt stöd, rang eller reliabilitet inte nås. Partiella koordinater får finnas
som deskriptiva mått utan att kallas full 9D-MNJ.

### G4 — Separat MNJ-metodgrind

MNJ på `mnps9_alt_202609` är ett nytt estimationsobjekt. Syntetisk
sanningskänd prövning ska täcka linjära och olinjära 9D-system, kollinearitet,
brus, partiell observation, ändrad samplingsintervall, gap och begränsat
grannstöd. Redovisa bias, varians, rang, avvisningsandel och stabilitet mot
grannval, regularisering och ankare. En begränsad realdatakontroll undersöker
endast stöd och numerisk känslighet. Estimatorval och trösklar ägs av
`nmd-analysis`; ingest får inte välja dem efter önskat biologiskt utfall.

Vid exakt inverterbar linjär koordinatändring gäller idealiskt
`J_new = T J_old T^-1`. Här byts mätfunktioner, så den relationen gäller
inte generellt. Även under ideal similarity är matrisnormer och vanliga
anisotropimått koordinatberoende. Vid en 9D→3D-projektion saknas i allmänhet
en invers, och 3D-flödet kan vara observations- och historieberoende. Varken
en giltig 9D-serialisering eller en god enstegsprediktion kvalificerar därför
en autonom lokal neural Jacobian.

**Utgång:** antingen kvalificerad *chart-level* 9D-estimator för angivet
stödintervall och modalitet, eller inga alternativa 9D-MNJ-anspråk.

### G5 — Analysägd jämförelse och kartrobusthet

`nmd-analysis` förregistrerar estimand, grupper, tidsankare, primära mått,
osäkerhetsmått, multiplicitetsfamilj, nullkontroller och kriterier för
överensstämmelse innan kontrastresultat öppnas. De tre banorna körs på samma
råkällor och så långt möjligt samma faktiska fönster och kvalificerade stöd:
fryst 3D/MNJ, nuvarande 9D/MNJ och alternativ 9D/MNJ. Redovisa både
matchat stöd och supportförlust; fyll inte bortfall med noll. Använd äldre
spindel-, psykos- och komamaterial som feasibility/sensitivitet eftersom deras
resultat redan är kända. Slutlig prövning kräver material som inte valde
koordinater eller trösklar.

Jämför fördefinierade *riktningar, osäkerhet och robusthet* hos kontraster,
inte råa Jacobianelement eller absoluta normer mellan olika kartor. Undersök
om en alternativ karta ger information utöver enkla spektrala baslinjer,
legacy-kartan och registrerade störfaktorer. Samma effekt i 3D och två 9D-
kartor är en känslighetskontroll, inte tre oberoende replikationer: 3D härleds
historiskt ur legacy-9D och alla kartor delar råsignal. ”Kartrobust dynamisk
struktur” får bara användas med denna begränsning och efter oberoende
datasetprövning.

### G6 — Eventuell kontraktspromotion

Först efter G0–G5: beslut om alternativet ska stanna som analysartefakt,
fortsätta som pilot eller bli en versionerad ingest-export. Vid promotion:
separat namnområde och schema-ID, versions- och modalitetsmetadata,
feature-/ankare-/tidsstödsproveniens, giltighets- och kvalificeringsfält,
reader-capability, dokumenterad bakåtkompatibilitet och ett uttryckligt
val av aktiv kartversion. Legacy-värden, namn och historiska HDF5-filer
ändras inte. Frånvaro av ny karta ska vara läsbart utan att feltolkas som
legacy-9D. Eventuell Jacobian-export har egen capability och status.

## 5. Arbetsdelning och validering vid genomförande

- **Mätledare i NMD:** frys legacy-kontrakt, mätfråga, filägande,
  pilotöverlapp och kompatibilitetskrav; kontrollera att v3-namnet inte
  kolliderar med befintlig programgeneration.
- **Ingest-implementering:** minsta möjliga isolerade feature-/exportväg,
  upplöst konfiguration, schema och tester. Ingen ny vetenskaplig estimand
  eller utfallsoptimering i ingest.
- **`nmd-analysis`:** estimatorval, syntetisk recovery, nuller, statistiska
  kontraster och biologisk tolkning.
- **Oberoende granskare efter implementation:** slutdiff, verifierade tester,
  referensutdata, gap, stöd, rank, metadata, backward compatibility och
  negativa resultat. Implementatörens sammanfattning räcker inte.

Testmatrisen behöver regressionskontroll av fryst 3D/v2-9D och deras MNJ;
syntetiska failure tests för gap, icke-finita och unsupported inputs;
schema-/reader-roundtrip och avvisning av fel kartversion; samt en bounded
realdatapilot med explicit config och egen output root. Kör inte en hel kohort
för att avgöra en featuredefinition. Varje substantiellt steg får en diarienot
med exakta kommandon/resultat, källor och begränsningar.

## 6. Nuvarande verdict och nästa beslut

**PARTIAL PASS som arkitekturhypotes; INCONCLUSIVE som ny mätning.**
Kontraktspåverkan är föreslagen och additiv, ingen produktionsimplementation
är gjord. Säker utsaga nu: legacy 3D kan bevaras medan en separat 9D-kandidat
prövas, förutsatt att hela legacy-vägen fryses. G0 innehåller nu historisk
inventering, en rådatareplay av EEG från aktuell arbetskopia och en fMRI-gap-
supportgranskning; detaljer och begränsningar finns i avsnitt 7 och i
`project/new_mnps_202609/g0_reference_lock_candidate_20260926.json`. EEG-replayn
är bitvis deterministisk mellan två körningar i samma runtime och matchar
historiska feature-, tids- och coverage-ytor, men projektion/Jacobian skiljer
sig utan en fördefinierad acceptanstolerans. fMRI:s numeriskt giltiga MNJ har
blandat tidsstöd över ett 96-sekunders gap. G0 etablerar därför inte oförändrad
3D/MNJ eller gap-säker fMRI-support. Det är inte heller etablerat att de nio
föreslagna koordinaterna är mätbara, icke-redundanta eller ger en giltig
9D-MNJ, eller att någon biologisk slutsats är kartrobust. Jämförelseregeln är
nu fryst för framtida prövningar: bitvis likhet inom samma implementation;
ingen kors-implementationstolerans utan oberoende, förhandsmotiverad regel.
Denna regel godkänner inte den redan observerade EEG-differensen. Första
G1-stödsskärmen visar att nuvarande EEG-kandidat saknar belägg för regional
struktur och att fMRI-summaryns MNJ-support korsar ett 96 s gap. ANPHY-smoken
har nu gått från råkanalåtkomst till en bounded, features-only jämförelse av
en enda explorativ `e`-kandidat mot den nuvarande globala EEG-PE-funktionen:
29/29 fönster beräknades med 83 kanaler och matchande legacy features. Det är
`PARTIAL PASS` för teknisk genomförbarhet, inte för full 9D, MNJ, CSD eller
vetenskaplig kvalificering. Nästa steg är att definiera och stödgranska övriga
kandidater kandidat för kandidat, oberoende av utfall; olösta historiska
configdelar och fMRI-MNJ över tidsluckor förblir utanför kompatibilitetsclaim.

### Fryst regel för numerisk kompatibilitet (2026-09-27)

Regeln gäller framtida reproduktions- och migrationsbeslut; den ändrar inte
G0-bedömningen av redan jämförda EEG-arrayer.

1. **Reproducerbarhet inom samma implementation:** jämförelsen kräver samma
   källhashar, kodsnapshot (ren commit eller commit plus hash av lokala diffar),
   resolved config efter att endast output-platsmetadata normaliserats, runtime
   och numeriska beroenden. Namn/ordning, former, tidsaxlar, fönstergränser,
   saknadhet, support och validity måste vara exakt lika. Numeriska arrayer
   jämförs bitvis; detta omfattar även NaN-positioner och NaN-payloadbitar.
   Annan kod, runtime eller config är
   inte ett same-implementation reproduktionstest.
2. **Kompatibilitet mellan implementationer:** ingen generell numerisk
   tolerans antas. En eventuell acceptansregel måste ange tolerans per
   outputyta och motiveras av mätfelbudget eller oberoende numerisk
   stabilitetsanalys före den jämförelse som ska bedömas. Stöd, validity,
   saknadhet och tidsgeometri måste fortfarande matcha exakt. Saknas en sådan
   regel blir utfallet `INCONCLUSIVE`; differensens storlek får inte ensam
   användas för att välja gränsen.
3. **Redan observerad G0-parjämförelse:** den historiska/aktuella EEG-paret
   förblir `INCONCLUSIVE` för numerisk kompatibilitet. Den nya regeln gör inte
   den observerade maxdifferensen `9.54e-7` till ett godkänt resultat. Historisk
   config saknar fortfarande komplett, självständig resolved-YAML-arkivering.

### G1:s första referensstödsskärm

Skärmen nedan inventerar bara dokumenterat ingångs- och tidsstöd; den väljer
inte formler eller estimatorer och har inte granskat utfall.

| Kandidatfamilj | EEG: ds005555/sub-99 | fMRI: ds007216/sub-001/ses-001 | G1-status |
| --- | --- | --- | --- |
| Regional fas eller koppling (`m`) | Rå-EDF-piloten har sex scalp-kanaler; CSD hoppades över under minimikravet 16. G0-artefakten etablerar ingen regional parcellering. | Derivatet innehåller 200 Schaefer-regioner, men sammanfattningen har 15 fönster och ett 96 s otäckt intervall; fasdefinition är inte fastställd. | Ingen jämförbar regional/metoddefinition. Håll som hypotes. |
| Persistens/omkonfiguration (`m`) | Källan omfattar vald 1 800 s rå-EDF-sträcka; replayn exporterade 60 separata 30 s-epoker. Gap- och kontinuitetsrevision för den råa tidsaxeln saknas. | Ingången redovisar 293 BOLD-volymer, men HDF5-fönsteraxeln har gapet och MNJ-support korsar det. Rå tidsserie har inte reviderats här. | Kräver separat rå tidsaxel, gapdelning och estimand före mått. |
| Graf/integration-segregation (`d`) | Sex scalp-kanaler och utebliven CSD räcker inte som belägg för stabil regional organisation; ingen grafestimand är vald. | 200 regioner finns i derivatet; de 15 sammanfattningsfönstren och deras gap validerar inte ett tidsserieestimat för grafen. | Kan inte kvalificeras från nuvarande summary-HDF5. Pröva rå regional tidsserie och estimatorstöd separat. |
| Irregularitet/komprimerbarhet (`e`) | EDF ger samples för en bounded feasibility-prövning, men band, filtrering, kodning/embedding, artefaktgrind och fönsterkrav är inte fastställda. | 293 volymer är noterade, men ingången är ett normaliserat derivat; TR-/gapkontrakt och operationell definition behöver frysas. | Feasibility är öppen; inget mått räknas som definierat eller kvalificerat. |
| Effektiv dimension (`e`) | Sex scalp-kanaler begränsar observerad spatial rang till högst sex i denna ingång; det ger inte stöd för nio oberoende spatiala koordinater. | 200 regioner mot 293 volymer i derivatet; ingen shrinkage-, stabilitets- eller temporal-gapprövning finns. | Endast rådimensionerna är kända; skattbarhet och reliabilitet är oprövade. |

G1 kan därför gå vidare med **definition och stödgränser**, inte med ett fullständigt
9D-kandidatset. Nästa konkreta underlag per behållen kandidat är formel/riktning/
enhet, exakta kanaler eller regioner och transformationer, tidsstöd och
gapdelningsregel, saknadhets-/avvisningsregel, förväntad mätfelbudget samt
etikettfria reliabilitets- och redundanskontroller. Kandidater utan dessa fält
förblir `unresolved` eller `unsupported`; inget fallbackvärde fyller en plats.

#### ANPHY: avgränsad råkanal-smoke (2026-09-27)

ANPHY ger en starkare kanalstödskandidat än ds005555. `EPCTL01` har en
EDF med 28 738 ensekunders datarecords: 93 datakanaler plus en
EDF-annotationskanal, vid 1 000 Hz. Datasettets
`Co-registered average positions.pos` listar 83 kanalpositioner och samtliga
83 etiketter finns i EDF; de övriga tio data-kanalerna är `ChEMG1/2`, `RLEG-/+`,
`LLEG-/+`, `ECG1/2` och `EOG1/2`.

En standardbiblioteksbaserad smoke läste bara de första 30 datarecords (30 s,
5 583 420 byte) och verifierade EDF-record-layouten samt 2 490 000
int16-sampel över de 83 positionerade kanalerna. Hashar för EDF-header,
positionfil och avgränsat payloadblock finns i
`g0_reference_lock_candidate_20260926.json`; hela 5,35 GB EDF-filen hashades
inte. Det här verifierar rå payload och etikettmatchning, inte featurevärden
eller sensorernas fysiska geometri. Positionfilen deklarerar inte koordinatens
enhet i dess rader/header, så koordinaterna skalades inte och användes inte för
CSD.

Basoverlayns `config_ingest_anphy.yaml` begär CSD med minimum 16 EEG-kanaler,
20 s fönster, 10 s steg och stage-stratifierade block. Den anger E:-vägar som
saknar ANPHY-källdata här; faktiskt råmaterial finns under K:. Ett MNE-baserat
30 s-försök avslutades efter cirka fyra minuter utan resultat; inget pipeline-
HDF5 skrevs. Smoke-försöket är därför **PARTIAL PASS för råkanalåtkomst**,
**INCONCLUSIVE för CSD och pipeline-extraktion**. En read-only kodsökning fann
ingen belagd inläsning av `.pos` i EDF-preprocessvägen; koordinatfilens
användning måste fastställas före CSD-claim.

En redan existerande ANPHY-körning från 2026-09-05 är inte en ny replay: dess
EPCTL01-summary redovisar 600 fönster, 10 s mediansteg och 6 310 s största
tidsintervall (`time_grid.status=warning`). Den nuvarande stage-stratifierade
profilen får därför inte användas för en sammanhängande MNJ utan explicit
segment-/gapstöd. Nästa bounded steg genomfördes för en enda förspecificerad
inspelning med två separata overlays och egna output-rötter; resultatet som
features-only smoke beskrivs nedan. CSD hölls avstängt i båda armarna eftersom
koordinatenheterna/-ramen ännu inte är fastställda.

#### ANPHY: features-only smoke av en alternativ e-kandidat (2026-09-28)

En avgränsad jämförelse körde samma EPCTL01-ingång och samma första 300 s i två
separata output-rötter, med 20 s fönster och 10 s steg. Stage-stratifierad
sampling stängdes av; CSD var avstängt i båda armarna. Endast `features` kördes;
`summarize`, HDF5-sammanfattning och MNJ lämnades utanför. Overlayfilerna är
`project/new_mnps_202609/anphy_smoke_legacy_20260927.yaml` och
`project/new_mnps_202609/anphy_smoke_alt_candidate_20260927.yaml`.

Legacy-armen körde befintlig features-pipeline. Alternativ-armen körde samma
pipeline och lade till en projektlokal sidecar-kandidat, utan produktionskod- eller
HDF5-kontraktsändring: `e_irregularity_channel_median_pe_v0`. För varje 20 s
fönster beräknas normaliserad ordinal permutation entropy separat för de fasta
83 EEG-kanalerna som listas i ANPHY `.pos`-filen, därefter median över kanaler:
`H_c=-sum_p(q_cp*ln(q_cp))/ln(5!)`; kandidat=`median_c(H_c)`. Ordning är 5,
fördröjning 1 sample vid 250 Hz, normalisering på och ties följer befintlig
`numpy.argsort(kind='mergesort')`-väg i
`mndm.features.eeg._compute_permutation_entropy`. Positioner används enbart
för fasta etiketter/ordning; inga koordinater eller fysiska enheter används.
Hela fönstret avvisas med orsak vid saknad kanal, icke-finit sample, konstant
kanal eller otillräckligt sample-/tidsstöd; inga nollvärden eller interpolation
fyller luckor.

**Resultat: PARTIAL PASS.** Båda CLI-körningarna skrev 29 fönster. De två
legacy `features.csv`-filerna var byte-identiska, SHA256
`aa6811450082687c717faeb2bfedc888ad56232f32c05abf5377ed4a76ee1687`; tabellerna
var `(29, 285)`. Kandidaten räknades i 29/29 fönster; 0 avvisades, och samtliga
29 jämförelser var kvalificerade enligt den förhandsbestämda jämförbarhetsregeln
`eeg_entropy_metric == permutation_entropy` och
`eeg_entropy_degraded_mode == false`. Den projektlokala köraren återskapade de
sex valda legacykolumnerna exakt mot CLI-tabellen. En oberoende granskare fann
inga materiella fynd och bekräftade byte-identiska legacytabeller, fönster
`0–20` till `280–300 s`, 83 kanaler, 5 000 samples per kanal/fönster och
matchande kanalordningshash.

Beskrivande parvisa värden utan acceptansgräns: Pearson `r=0.77217`, Spearman
`rho=0.70887`, kandidatmedian `0.60369`, legacymedian `0.89458`, median skillnad
(kandidat−legacy) `−0.28857`. Dessa värden är inte tröskeltest, belägg för
icke-redundans, biologisk effekt eller kvalificering. De två Parquet-filerna
har olika hashar och hävdas inte vara identiska; den byte-exakta paritetsclaimen
avser CSV/featurevärdena.

Käll-EDF: `K:/ExternalReceivedDatasets/ANPHY/Subjects/EPCTL01/EPCTL01 -
fixed.edf`, storlek 5 348 568 452 byte, full SHA256
`a16ddc068edcc4ce2ff8c499cca51b9a890b76b7ddb569f9a96978013f8b5e8c`;
`.pos` SHA256 `b5bbc85ef1e840e0ca703c62c22a1000dd3252f32210e251b8d0e81123fa1bd2`.
Preprocess gav 83 EEG-kanaler vid 250 Hz, cropens faktiska slut var 300.001 s
på sample-grid och inga BAD-intervall observerades i cropen. CSD var avstängt;
koordinatenheter/ram är fortsatt okända. Output-rötter, utdatahashar, körkommando,
runtime, kodsnapshot och reviewstatus finns i
`project/new_mnps_202609/g0_reference_lock_candidate_20260926.json` och
`project/diary/483_20260928_anphy_alt_candidate_smoke.md`. Ingen hel kohort,
full alternativ 9D, MNJ, biologisk kontrast eller produktionsintegration
ingick. Nästa grind är att operationellt definiera och stödgranska kvarvarande
kandidater; koordinatkontrakt krävs innan CSD kan prövas.

#### ANPHY: G1-stöd för kvarvarande kandidater (2026-09-28)

Detta är en tillgänglighets- och blockerarscreen från redan inspekterad ANPHY-
metadata, bounded råcrop och features-smoke. Den väljer ingen estimator, formel,
tröskel, acceptansgräns eller full 9D-definition; inga nya signalmått kördes.

| Kandidatfamilj | ANPHY-underlag | Öppet krav / status |
| --- | --- | --- |
| `m`: regional fassynkroni | 83 EEG-kanaletiketter finns; pilotcroppen gav 83 preprocessade kanaler vid 250 Hz. | `.pos`-koordinatenhet/ram, referens/CSD, band, fasskattare, regional gruppering och temporal support är inte fastställda. Regional kandidat förblir `unresolved`; CSD-claim `unsupported` tills geometri är verifierad. |
| `m`: persistens/autokorrelation | Rå EDF har ensekundsdatarecords; den processade första 300 s gav 29 överlappande 20 s-epochs utan observerade BAD-intervall. | Estimand, tidsskala/lag, preprocessing, avbrotts-/gapregel och reliabilitetskrav saknas. Input är tillgänglig för en bounded stödprövning, men inget mått är definierat. |
| `m`: omkonfiguration | Samma 300 s-källa och fönsteryta finns. | Jämförelsefönster/segmentering och omkonfigurationsestimand saknas; att använda förändring av featurevärde vore inte ratifierat. `unresolved`. |
| `d`: integration–segregation/graf | 83 kanaler finns, men inget validerat regionalt nodset producerades. | Grafestimand, kanter/viktning, temporal aggregation och CSD/volymledningskontroller saknas. `unresolved`. |
| `d`: lokal kontra distribuerad koppling | 83 kanaler finns; positionfilens etiketter matchar kanalerna. | Lokaliseringsram, referens/CSD och kopplingsestimator saknas. Kanalantal räcker inte för regional claim. `unresolved`. |
| `d`: spatial/representational dispersion | 83 positionsetiketter finns. | Koordinatenheter/ram och RDM/topografi-estimand saknas. Geometribaserad variant `unsupported` tills positionerna kan tolkas; representational variant `unresolved`. |
| `e`: permutation entropy | `e_irregularity_channel_median_pe_v0` är körd för 29/29 matchade fönster; se föregående avsnitt. | `PARTIAL PASS` endast som dimensionslös teknisk feature-kandidat. Reliabilitet, känslighet, redundans och kvalificering är oprövade. |
| `e`: normaliserad Lempel–Ziv | Bounded rå-EDF kan preprocessas till 83 kanalserier; föregående körning sparade inte sin in-memory signalarray. | Symbolisering/binarisering, sekvenslängd, normalisering, support och negativ kontroll saknas. `unresolved`; inget värde beräknades. |
| `e`: effektiv regional dimension | Bounded rå-EDF kan preprocessas till 83 kanalserier; föregående körning sparade inte sin in-memory signalarray. | Regional definition, estimator, shrinkage/brusbehandling och tids-/samplingkrav saknas. 83 observerade kanaler är inte belägg för 83 oberoende dimensioner; `unresolved`. |

Ratifieringsbeslut från `/root/ratification_review` (gpt-6-sol, medium):
**INCONCLUSIVE; ingen ny operationell kandidat ratifieras från nuvarande
evidens.** Temporal persistens/autokorrelation rekommenderas som nästa
tekniska prioritet eftersom bounded rå-EDF och overlay kan återskapa 83
preprocessade EEG-serier vid 250 Hz för en 300 s crop; den tidigare körningens
signalarray sparades inte som artefakt, och inga BAD-intervall observerades i
den cropen. Hypotesen kräver inte `.pos`-geometri. Detta är inte ett
estimatorval eller bevis på oavbrutet tidsstöd. Innan extraktion krävs ett kort operationellt
kontrakt med estimand/signalrepresentation, lag/tidsskala, preprocessing,
tidsstöd/gap- och avvisningsregel samt reliabilitetskrav; estimatorval ägs av
`nmd-analysis`. Därefter granskas/ratificeras kontraktet. Ingen ny feature eller
`summarize` ska köras före det. Geometriberoende `m/d` väntar dessutom på
koordinatenhet/ram, referens/CSD och regional konstruktion. Beslutet och
full kandidatstatus finns även i `g0_reference_lock_candidate_20260926.json`,
blocket `ANPHY_remaining_candidate_support_20260928`.

Read-only kodinventering hittade redan `compute_tau_summary` i MNDM och
`_autocorr_tau_sec` i `nmd-analysis`, men ingen av dem används här på rå EEG-
kanalprover. MNDM:s standard-summary/tau-block tillämpar sin helper på MNPS-
och 9D-koordinatserier; en separat, valfri baseline-QC-väg kan även tillämpa
den på valda feature-serier. MNDM:s helper har som standard 1/e-tröskel och 60 s
maxlag med strict/interpolate-policy; analysis-helpern använder upp till 300 s
maxlag och tar bort icke-finit samples före ACF, vilket kan komprimera
tidsstödet. De skiljer sig alltså i support-/saknadhetsbeteende och ingen av
dem har ratifierats för en ny neural kanalserie. Granskningen motiverar gap-
regeln som ett öppet kontraktsfält, inte ett förslag att återanvända endera
implementationen. Oberoende read-only review fann efter preciseringarna inga
kvarvarande materiella fynd i stödmatrisen eller ratifieringsstatusen.

#### Föreslagen EEG-kanal-ACF-tidsskala: kontraktsutkast och ratifiering (2026-09-28)

Ett konkret men ej ratifierat utkast finns i
`project/new_mnps_202609/anphy_m_persistence_candidate_contract_draft_20260928.md`.
Kandidat-ID är `eeg_channel_median_acf_tau_v0`; det är en beskrivande
signal-dekorrelationstid och har inte placerats under `m` eller `e`. Förslaget
tar medianen över 83 kanalers första positiva ACF-lag under `1/e`, per samma
20 s/10 s fönster. Pipelineordningen är resampling 1 000→250 Hz, därefter
50 Hz notch och 1–45 Hz bandpass, därefter average rereference; CSD är av.
Maximalt observerbart lag är 19.996 s, men den föreslagna sökhorisonten `K` är
inte vald. MNDM-hjälparens 60 s standard är bara referens. För den nuvarande
biased ACF-formeln gäller `sum(r(k), k=1..N-1)=-1/2` för en mean-centered,
icke-konstant kanal; om alla laggar söks är alltså en `<1/e`-korsning
matematiskt garanterad. Censurering blir meningsfull först med en kortare vald
horisont eller en annan estimator. Den finita fönsterestimatorn kan därför
korsa långt före populationens tidsskala.

Den första ratifieringen (`gpt-6-sol`, medium) blev **INCONCLUSIVE**. Efter
oberoende precisering och matematisk korrigering omratificerade samma agent den
aktuella revisionen SHA256
`49c42edb336125089d06d0c7fc018b2c7e3137aca6f4e4e8fc7e88d956547ffd` och
beslutade **INCONCLUSIVE även för en bounded support/calculation-pilot**.
`nmd-analysis` måste först (1) godkänna/avslå estimand och biased ACF med första
`1/e`-korsning, (2) välja och motivera sökhorisont `K` ≤ 4 999 samples relativt
20 s-fönstret, (3) fastställa censorering och all-83-support/aggregation, samt
(4) godkänna/avslå signalrepresentationen 1–45 Hz, average reference och
250 Hz. Resultatet kan hållas som en deskriptiv EEG-tidsskala; slutlig `m`/`e`
placering kan skjutas upp om pilotartefakten inte gör chart-claims.

En current-`m`-jämförelse behövs inte för en ren supportpilot. Nuvarande EEG-`m`
bygger på viktade delta/theta/alpha/beta-features i common EEG-config; det
tidigare features-only-smoket exporterar inte ett jämförbart `m`-resultat på
dessa fönster. Global EEG-PE används inte som ersättande comparator.

Utkastet innehåller ännu orörda syntetiska kontroller (white noise, AR(1),
10 Hz oscillation, långsam AR(1) som visar fönstertapering, konstant/icke-finit
kanal samt timestamp-gap/BAD). En oberoende granskning fann att den tidigare
förväntningen om högercensurering för `phi=0.9999` inte stämmer med biased
finite-window ACF: även om populationens crossing ligger runt 40 s, kan
fönstertapering ge en crossing inom 20 s. Två oberoende reviewers verifierade
att rättelsen är korrekt, hash/JSON parsar och inget extraktionstillstånd
impliceras. Nästa grind är analysägarens metodbeslut och därefter körs
syntetiska kontroller mot den fixerade kontraktsdefinitionen före någon
realdataextraktion. En ny positiv ratifiering krävs före ANPHY-smoke. Inga
mått, features eller `summarize` körs före positiv ratifikation; inget
fullständigt 9D, MNJ, CSD, biologisk jämförelse eller kvalificering ingår.

#### G1: operationella mätdefinitionsutkast (2026-09-28)

Ett separat beslutssheet formaliserar de nio föreslagna `m/d/e`-kandidatfamiljerna
utan att tilldela dem fasta koordinater:
`project/new_mnps_202609/g1_measurement_definitions_draft_20260928.md`.
Oberoende `gpt-6-sol`/medium-granskning gav **PASS** för korrigerade kontrakts-
och stödbeskrivningar, men själva nio-koordinatkartan är fortsatt
**INCONCLUSIVE**. G0 historisk 3D/MNJ-kompatibilitet är också fortsatt
INCONCLUSIVE.

Endast ANPHY:s kanalmedian-permutation-entropy har en avgränsad teknisk
beräkning; reliabilitet, sensitivitet, redundans och kvalificering återstår.
ACF:s estimand, K, censurering, 83-kanalsaggregation och signalrepresentation
är fortfarande INCONCLUSIVE enligt den aktuella ratifieringen; det tidigare
K=250-förslaget är inte ratifierat. Övriga `m/d/e`-kandidater stannar som
olösta eller unsupported där regional geometri/tidsstöd saknas. Nästa grind är
analysägarens val av estimand och oberoende kriterier för support, reliabilitet
och redundans. Ingen ny kod, signalextraktion, ACF eller `summarize` kördes för
detta beslutssheet. Den separata ds004511 v1.2/v2.0-smoken jämför endast
befintliga EEG-mallar och stänger inte denna G1/G2-grind.

## Underlag inom arbetsytan

- `project/mnps_v3/mnps_v3_implementation_plan.md` — befintlig v3-generation
  med fryst MNPS-karta.
- `mndm/config/config_ingest_common_eeg.yaml`,
  `mndm/config/config_ingest_common_fmri.yaml`,
  `mndm/src/mndm/projection.py`, `mndm/src/mndm/pipeline/summary.py` — aktiv
  projektion, ankare, support och fallback som ska verifieras mot upplöst körning.
- `J:/repos/nmd-analysis/articles/NeuralManifoldDynamics/` — metodmanuskript
  om det versionerade 3D/9D-kontraktet.
- `J:/repos/nmd-analysis/articles/RVC_SLEEP/`,
  `J:/repos/nmd-analysis/articles/CLINICAL/psychosis_first_time_episode/` och
  `J:/repos/nmd-analysis/articles/CLINICAL/comatose_ndt/` — redan sedda
  empiriska exempel, lämpliga för hypotes- och stödgranskning men inte
  orörd slutvalidering.
- `project/astra/m3_extension_20260906/report.md` — syntetisk begränsning:
  enstegsskattning certifierar inte observerad tillståndssufficiens.

## 7. G0-förinventering (2026-09-26)

En historisk EEG-kandidat hittades i ds005555: `sub-99`, begränsad till de
första 1 800 sekunderna av PSG-inspelningen. Körningen skapade 60 fönster med
30 sekunders längd och 30 sekunders steg. Den kom från den separata, grova
kontinuerliga EEG-piloten; den är inte en spindelanalys och ersätter inte den
gemensamma EEG-profilens 8/4-sekundersfönster.

Det faktiskt skrivna HDF5-manifestet anger `mnps_3d.mode_requested` och
`mode_effective` som `from_v2`, definitionen
`derived_mde_from_v2_fixed_weighted_projection` och `fallback_reason: null`.
Den skrivna 9D-ordningen är `[m_a, m_e, m_o, d_n, d_l, d_s, e_e, e_s, e_m]`.
V1-kolumnerna är L2-normaliserade var för sig och saknat stöd renormaliseras
per axel. I artefakten är alla 60 rader och samtliga axeltäckningsvärden
fullständiga. Den lagrade MNJ:n har 58 giltiga centrum i både 3D och 9D.

Referensen visar också varför `common_eeg` inte räcker som upplöst
konfiguration: pilotens fönster är 30/30 sekunder, medan HDF5-geometrikontraktet
redovisar 4 Hz och 8 sekunder i den separata `mnps`-konfigurationen. Faktiskt
tidssteg är 30 sekunder och matchar datasetens epoch-override, men inte
`mnps`-formeln. Denna tids-/MNJ-konflikt måste lösas innan piloten kan frysa
MNJ som gemensam EEG-baslinje. Legacy-`e_m` kommer från fallback-feature
`ppg_rate_bpm` med `semantic_equivalence=false`; det är en del av historisk
proveniens, inte stöd för en ny neural koordinat.

Primärkörningens `run_manifest.config.excerpt.mnps` låser även estimatorns
historiska inställningar: Savitzky–Golay 7/3, robust derivata `max_jump=5` och
`min_seg=9`, kNN `k=20`, ridge `alpha=1` med 5 CV-folds, whitening och
`super_window=3` (fs_out 4 Hz, fönster 8 s, overlap 0,5). Detta är
konfigurationsproveniens, inte stöd för att estimatorvalen passar den
observerade 30-sekundersrutans MNJ.

Två historiska körningar, båda från Git-revision `eb0ff86`, har exakt lika
arrayer för 3D, 9D, tidsrutnät, täckning, 3D-MNJ och 9D-MNJ. Deras kompletta
HDF5- och konfigurationshashar skiljer sig eftersom körningarna har olika
metadata/config-digest; likheten gäller de uppräknade numeriska ytorna och
utgör historisk reproduktion, inte oberoende evidens om nuvarande checkout.

En fMRI-kandidat finns också för ds007216, `sub-001/ses-001`, i den särskilda
`lightweight_norm_v1`-derivatkällan och dess separata processed-root. Den
faktiska HDF5-manifesten anger `from_v2`, ingen fallback och normaliserad
projektionshash
`b0da037a2feed96aa24c3a3b4d38d4e174ab93f306db883323c38f3adbd9d359`; den
realiserade m-kolumnen använder `m_a` och `d_s`, medan d använder `d_n` och e
`e_e`. HDF5 innehåller 200 Schaefer-regioner och 15 kompletta fönster. Men
fönstren är 44 sekunder långa med mediansteg 14 sekunder, trots konfigurerade
45 sekunder och nominellt 15,3 sekunders steg; ett fönstercentrum-/startintervall
är 140 sekunder. Med fönsterlängd 44 sekunder återstår 96 sekunder utan
fönstertäckning.
`geometry_contract` är `adjusted` och tidsrutans status `warning`. `row_source`
och `source_format` är dessutom `unknown`; samtidigt är `/row_source/has_eeg=1`
på samtliga 15 fMRI-rader. Den EEG-flaggan är en proveniensanomali och räknas
inte som modalitetsstöd. Ingest-ingången är ett lightweight-normaliserat
BOLD-derivat, inte rå BOLD. Historisk estimatorconfig var Savitzky–Golay 5/2,
kNN `k=10`, ridge `alpha=1` med 3 CV-folds, whitening och `super_window=2`
(fs_out 0,5 Hz, fönster 45 s, overlap 0,66); även detta är konfigurations-
proveniens, inte estimatorvalidering. En senare `summarize`-körning ger bitvis
lika koordinat-/MNJ-arrayer och samma valda subjekts feature-namn/-värden.
Globala `features_snapshot.json` skiljer sig dock (124 respektive 212 rader),
så detta visar återanvändning av valda subject-feature-ytor, inte identiska
globala snapshots eller en oberoende rådatapipeline-replay. Detta är en
användbar fMRI-projektionskandidat; dess tids-/gapkontrakt och proveniens
stänger inte MNJ-grinden.

Nuvarande HEAD är `2e58fcd` och arbetskopian är smutsig; våra körningar läste
denna befintliga arbetskopia utan kodändringar. Den ursprungliga Python 3.12.0-
vägen fastnade vid MNE-import. Två försök med Codex Python 3.12.14 stoppade
först på saknade `pytz`/`threadpoolctl`, sedan på saknad `matplotlib`; den
senare skrev bara file index och fellogg. Med processlokal dependency-bootstrap
från installerade paket kunde två bounded rådata-replayer slutföras i separata
roots. Båda extraherade 60 EEG-epoker direkt från EDF under `--force-features`;
CSD hoppades över eftersom sex scalp-kanaler är färre än minimikravet 16.

Den första aktuella replayen har HDF5-hash
`90d20e01ecc2b69568e229af202106a849bc4b1262adf55e0c99280e442b86db` och
config-digest `cd40dd0432f1e322750516542ef3715d6874be2727aa0d1fef8877c7315e7fa3`.
Estimatorutdraget och normaliserad V2→V1-matris-hash matchar den historiska
körningen; resolved-config-digesten gör det inte. Features, feature-namn,
tidsrutnät och coverage är bitvis lika historiken. 3D-/9D-koordinater och
Jacobianer är reproducerbara bitvis mellan de två aktuella körningarna, men
inte bitvis lika den historiska HDF5:n: största observerade absoluta skillnad
är `9.54e-7`. Ingen tolerans var fördefinierad, så numerisk kompatibilitet får
inte godkännas med en efterhandsvald gräns. Även i den nya körningen visar
time-grid audit `status=ok` samtidigt som faktisk dt 30 s skiljer sig från
konfigurerad 4 s och 30 s-fönstren från konfigurerade 8 s.

fMRI-gapgranskningen visar att 140 s mellan centrum/start vid 64→204 s ger
96 s utan fönstertäckning. Den historiska derivatasegmenteringen tittar på
värdehopp, inte tidsstämplar; gränshoppet 3,06 ligger under dess tröskel 4,60
och delar inte sekvensen. 49/150 kNN-länkar korsar gapets tidsdelning och
12/13 Jacobiancentrum har lokalt stöd från båda sidor. `J_dot` använder
gradient över indexordning med konstant 14 s trots 140 s vid gapet. Därför är
fMRI:s numeriska validitetsstatus inte belägg för gap-säker MNJ-support.

**G0 verdict:** PARTIAL PASS för historisk inventering, verifierad aktuell
EEG-rådataextraktion och same-runtime bitvis determinism; INCONCLUSIVE för
historisk-versus-aktuell numerisk parity utan fryst tolerans och komplett
historisk config; METHOD-LIMITED för gap-säker fMRI-MNJ-support. Säker utsaga
är att EEG:s verifierade featureytor/tider/coverage matchar historiken, medan
projektion/Jacobianvärden endast visar små men icke-noll skillnader mot den
historiska artefakten. Båda modaliteternas HDF5 anger lägesnivå
`mode_effective=from_v2` och `fallback_reason=null`; EEG:s legacy-`e_m` använder
ändå fallback-feature `ppg_rate_bpm` med `semantic_equivalence=false`. Inget
av detta kvalificerar ny koordinat eller MNJ. En generell kompatibilitetsregel
är nu dokumenterad i avsnitt 6; den observerade historisk/aktuell-differensen
förblir `INCONCLUSIVE`. Nästa steg är G1:s operationella mätdefinitioner och
avvisningsregler utan utfallsinsyn. Historisk configrekonstruktion begränsas
till vad arkiverade artefakter faktiskt stöder, och fMRI gap-säker support
förblir utanför claim tills estimatorn respekterar tidssegmenten.
