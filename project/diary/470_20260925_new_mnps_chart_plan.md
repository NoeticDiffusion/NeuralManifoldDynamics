# 470 — 2026-09-25 — plan för alternativ MNPS-9D-karta

## Mätfråga och grind

Kan en separat alternativ 9D-observationskarta prövas utan att ändra den
historiska kanoniska 3D-banan eller befintlig 9D? Detta är en **planeringsgrind**,
före featureimplementation, kartskattning, MNJ-prövning och vetenskapliga
kontraster.

## Verdict och kontraktspåverkan

**PARTIAL PASS som arkitekturhypotes; INCONCLUSIVE som ny mätning.**
`project/new_mnps_202609/new_mnps_202609.md` beskriver en additiv,
separat 9D-kandidat och G0–G6-grindar. Ingen produktionskod, resolved config,
schema, HDF5-fil, numeriskt beteende eller mätsemantik ändrades. Den
kanoniska `[m,d,e]`-ordningen, nuvarande 9D-definitioner, MNJ-ytor,
missingness, validity och provenance förblir som tidigare.

## Tidigare beteende och källor

Aktiva EEG- och fMRI-referenser använder `mnps_3d.mode=from_v2`:
nuvarande 9D projiceras med fasta vikter och stödberoende renormalisering.
`mndm/src/mndm/pipeline/summary.py` har även en `direct_features`-väg och
fallback när 9D inte finns, vilket planen kräver explicit kontroll av vid
G0. `project/mnps_v3/mnps_v3_implementation_plan.md` avser redan en annan
v3-generation för dynamiska familjer och fryser nuvarande karta. Den nya
planen använder därför arbetsnamnet `mnps9_alt_202609`.

Den explorativa bedömningen byggde på lokala metod- och empiriska manuskript
i `J:/repos/nmd-analysis/articles/NeuralManifoldDynamics/`, `RVC_SLEEP/`,
`CLINICAL/psychosis_first_time_episode/` och `CLINICAL/comatose_ndt/` samt
syntetisk M3-evidens i `project/astra/m3_extension_20260906/report.md`.
Artiklarna ger motiv för prövning men ingen kvalificering av nya koordinater.

## Ändringar och filer

- Ny plan: `project/new_mnps_202609/new_mnps_202609.md`.
- Denna diarienot.

Planen anger kontraktsgräns, kandidatmått, stöd-/gapkrav, syntetiska
failure tests, bounded pilot, MNJ-metodgrind, analysägd kartjämförelse,
eventuell separat schemapromotion och oberoende granskning efter framtida
implementation. Kroppsliga mått föreslås som separat kontext i alternativet;
legacy-`e_m` lämnas oförändrat.

## Validering, negativ evidens och begränsningar

Endast dokumentkontroll gjordes: `git status --short` före skrivning;
`rg`/`Get-Content` av relevanta lokala kontrakt, manuskript och föregående
diarienot; `Test-Path project/new_mnps_202609/new_mnps_202609.md` gav
`True`; `rg -n '^(#|##|###)' project/new_mnps_202609/new_mnps_202609.md`
visade G0–G6; `git check-ignore -v --
project/new_mnps_202609/new_mnps_202609.md` visade att nya filer under
`project/` annars ignoreras. `git add -N -f --` användes endast på plan och
diarienot för att göra diffen synlig, utan att stage:a innehållet.
`git diff --check --` på de två nya filerna passerade utan whitespace-fel;
Git gav endast LF/CRLF-notiser.
**Unit/regressionstester, syntetiska tester, realdatapilot och oberoende
slutgranskning: ej körda**, eftersom ingen implementation eller numerik
ändrades. Ingen ny 9D-karta eller Jacobian beräknades. Överensstämmelse
mellan 3D och två 9D-kartor skulle inte utgöra tre oberoende replikationer;
3D är historiskt härledd ur legacy-9D och alla delar råsignal.

## Kompatibilitet, tolkning och nästa steg

Säker utsaga: en separat karta kan planeras utan att skriva över befintliga
utdata. Det är inte fastställt att kandidaterna har tillräckligt tidsstöd,
reliabilitet, icke-redundans eller skattbar 9D-MNJ. Nästa grind är G0:s
legacy-frysning mot faktiskt upplösta körningar, följd av G1:s exakta
modalitetsspecifika mätdefinitioner utan utfallsstyrd optimering.
