Ja. Min slutsats: ni har redan räddat råsignalen på **feature-nivå**, men inte på **koordinat-nivå**.

I koden finns `/features_raw` och `/features_robust_z`, men `mnps_3d` och `coords_9d` beräknas fortfarande från `mnps_projection.normalize`, typiskt `robust_z`, på den `sub_frame` som bearbetas per subject/session/run. Så biologiskt viktiga absoluta skillnader kan fortfarande försvinna ur själva manifold-koordinaterna.

Jag skulle tänka så här:

- Behåll nuvarande `subject/session robust_z` som ett **subject-relative lager**. Det är bra för inom-subjekt-geometri, lokala Jacobianer och form på trajektorier.
- Gör inte helt rå projektion till default. Rå features har olika enheter och skala, så råa koordinater blir lätt “den största enheten vinner”.
- Lägg till ett **cohort/batch-anchored koordinatlager** som primärt för kliniska gruppjämförelser. Det bör byggas från frusna feature-anchors: median/MAD eller median/IQR per feature, fit på analyskohorten eller extern referens.
- För prediktiva analyser: fit anchor bara på train-fold. Annars får ni normaliseringsläckage.
- För I-CARE/CPC: jag skulle vara försiktig med `CPC1-only` som primär anchor om CPC är outcome. Det kan vara en tydlig deskriptiv referens, men för prediction bör primär vara fold-fitted eller pre-outcome/external anchor.

Det viktigaste tillägget är alltså inte “sluta z-score:a”, utan att exportera flera koordinatkontrakt:

```text
/mnps_3d                         # bakåtkompatibelt, nuvarande lager
/coords_9d                       # bakåtkompatibelt, nuvarande lager

/coords_3d_subject_anchored       # nuvarande subject/session-relative
/coords_9d_subject_anchored

/coords_3d_cohort_anchored        # primär för gruppskillnader
/coords_9d_cohort_anchored

/feature_anchors/...              # center, scale, n_used, source, hash
```

En extra viktig detalj: anchor-fit bör vara **subject-balanced**, inte bara “alla epochs staplade”. Annars får subjects/runs med längst recording oproportionerligt inflytande på nollpunkt och skala.

Så ja: rådata finns, men dagens primära koordinater kan fortfarande maskera det du är orolig för. Jag skulle göra `cohort_anchored` till primärt analyslager för coma/recovery-paper, behålla `subject_anchored` som dynamisk/diagnostisk yta, och kräva sensitivity plots där huvudresultatet jämförs över båda.