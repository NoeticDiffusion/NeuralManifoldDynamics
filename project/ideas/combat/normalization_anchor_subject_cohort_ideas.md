# Plan: Normalisering av anchor-forskjutning (subject/cohort)

## Syfte
- Bevara biologisk centroid-forskjutning mellan kliniska grupper.
- Minska site- och maskin-effekter som kan dominera cohort-anchor.
- Halla pipeline reproducerbar och tydlig over flera dataset.

## Beslutskriterier
- Signalbevarande: hur val bevarar biologisk separation.
- Robusthet: hur val fungerar mellan sjukhus, maskiner och sessionsskillnader.
- Tolkningsbarhet: hur enkelt det ar att forklara resultat i artikel/validering.
- Implementationstakt: hur snabbt metoden kan testas utan stor ombyggnad.
- Riskprofil: overkorrektion, leakage och svag generalisering.

## Alternativ A: ComBat (feature-niva)
### For
- Etablerad standard i neuroimaging for batch/site-harmonisering.
- Relativt snabb att testa i befintlig feature-baserad pipeline.
- Explicit hantering av covariater (diagnos/grupp) i modelleringen.
- Lat integration: kan laggas som valbart steg fore projection/summarize.

### Nackdelar
- Antaganden (additiv/multiplikativ batchstruktur) kan vara for enkla.
- Risk for overkorrektion om batcher ar sma eller obalanserade.
- Jobbar pa feature-niva, inte direkt pa tidsseriens ursprungliga struktur.
- Kraver stabil och korrekt metadata for site/batch.

## Alternativ B: Spectral whitening (signal-niva, EEG-specifikt)
### For
- Angriper hardware-signatur tidigt i kedjan, fore geometri/MNPS.
- Kan minska skillnader mellan forstarkare/filter utan att byta modellfamilj.
- Passar bra for EEG med MNE/FOOOF-verktyg.

### Nackdelar
- Hogre preprocess-komplexitet och fler tuningval.
- Kvaliteten beror pa robust fit av aperiodisk komponent.
- Felaktig parameterisering kan dampa relevant fysiologisk signal.
- Mindre direkt overforbar till icke-EEG-modaliteter.

## Alternativ C: Riemannian alignment (kovarians/manifold)
### For
- Geometriskt kompatibel med manifold-baserad analysfilosofi.
- Kan bevara relativa avstand vid korrekt alignment.
- Stark evidens i BCI/cross-session kalibrering.

### Nackdelar
- Hogst implementation- och valideringskostnad.
- Svarare att felsoka och kommunicera till bred publik.
- Kraver robust referensstruktur (t ex kontroller per site).
- Hoger risk for regressions i befintlig produktion om det infors for tidigt.

## Rekommenderad stegplan (utan implementation i detta steg)
1. **Fas 0 - Baseline och audit**
   - Fast baseline utan harmonisering.
   - Mata site-separation vs grupp-separation i samma metrikset.
2. **Fas 1 - ComBat pilot (forstahandsval)**
   - Kor ablation: `none` vs `combat`.
   - Utvardera om gruppsignal bevaras och site-signal minskar.
3. **Fas 2 - Spectral pilot for EEG**
   - Endast om Fas 1 inte racker eller om hardware-signatur kvarstar.
4. **Fas 3 - Riemannian exploratory branch**
   - Isolerad forskningsgren, inte default i huvudpipeline.

## Beslutsregel (go/no-go per fas)
- Bevara eller forbattra biologisk separation.
- Minska site-effekt utan att oka instabilitet mellan reruns.
- Inga regressions i centrala MNDM-output (H5/manifest/kontrakt).

## Konfigurationsskiss for framtida styrning
- `normalization_method: none | combat | spectral_whitening | riemann`
- `normalization_scope: pre_features | post_features`
- `normalization_batch_key: site_or_hospital`
- `normalization_covariates: [group, age, sex, ...]`
- `normalization_reference_policy: within_dataset | frozen_external`

## Kort slutsats
- Kort sikt: **ComBat** ar mest pragmatiskt (hog effekt/lag kostnad).
- Mellansikt: **Spectral whitening** ar starkt komplement for EEG.
- Lang sikt: **Riemannian alignment** ar lovande men bor hanteras som avancerad forskningslinje.
