Det du ser är helt normalt, och handlar nästan alltid om **(1) filformat + kompression**, **(2) om datan är kontinuerlig råsignal eller epokad/komprimerad**, och **(3) sampling rate + bitdjup + antal kanaler** – inte “hur mycket information” det finns i experimentet.

## 1) Chennu-datasetet (Cambridge/PLOS 2016) är *preprocessat och epokat*

Chennu-datasetet i Cambridge-repo beskrivs som:

* ~7 minuter per condition
* 4 conditions per person
* 91 kanaler
* **filtrerat, epokat i 10-sekunders epoker, rensat och average-referenced** ([repository.cam.ac.uk][1])

Detta betyder att datasetet **inte är “rå kontinuerlig recording som sparats oförändrad”**, utan redan har reducerats till ett mer kompakt analysformat (EEGLAB `.set/.fdt` är vanligt här). När man epokar, rensar och sparar i ett format som är effektivt, blir volymen ofta förvånansvärt låg.

### Grov back-of-the-envelope

7 min × 4 = 28 min per person. 20 personer ⇒ 560 min total EEG.

Om sampling rate är t.ex. 250 Hz och 32-bit float:

* per sekund: 91 ch × 250 samp × 4 bytes ≈ 91 000 bytes ≈ 0.087 MB/s
* per minut ≈ 5.2 MB
* per person (28 min) ≈ 145 MB
* 20 personer ≈ 2.9 GB

Det ligger helt i linje med “~3.7 GB” när man räknar med overhead, metadata och variation i fs/format.

## 2) ds005620 verkar vara *mycket mer rå/odownsamplad kontinuerlig data*

Din observation “~2 GB per subject” tyder på att ds005620-filerna är sparade som **långa kontinuerliga råsignaler**, potentiellt med:

* hög sampling rate
* många kanaler / extra aux-kanaler
* eventuellt 16/32-bit men utan aggressiv kompression
* BrainVision `.eeg` är en binär råfil som ofta blir stor om man sparar länge och snabbt ([Brain Products GmbH][2])

Dessutom: ds005620 innehåller flera acquisitions och runs per subject (awake EC/EO, sedation runs, sed2 snippets, ev tms), och du nämnde själv stora per-subject-volymer.

Så: **ds005620 kan vara större även om det är “svagare” för MNJ/reachability**, eftersom storlek här mest speglar *lagringsval*, inte experimentets styrka.

## 3) Varför “större dataset” inte betyder “bättre för MNJ”

För MNJ/reachability är det centralt att du har:

* **tillräckligt långa, homogena segment per condition**
* rimlig artefaktprofil
* stabila windowing-möjligheter

Chennu har explicit ~7 min per condition och tydliga conditions (baseline/mild/moderate/recovery) ([repository.cam.ac.uk][1]).
ds005620 har en design som inkluderar korta sed2-snippets (40–44 s) och blandade awake acquisitions, vilket är exakt den typ av protokoll som gör estimatorer känsliga (och gav er C3-artefakt-risk).

## 4) Varför `.set/.fdt` ibland känns “litet” trots 91 kanaler

EEGLAB-formatet är ofta:

* antingen epokad data (inte ett enda enormt kontinuerligt block),
* ibland sparat med effektiv datatyp (t.ex. float32),
* plus: preprocess kan innebära att man redan **tagit bort kanaler** (t.ex. ansikte/nacke) och artefaktsegment. FieldTrip/Chennu-beskrivningar nämner just att de behöll 91 scalp-kanaler och exkluderade andra ([fieldtriptoolbox.org][3]).

## 5) Slutsats: det finns ingen paradox här

* **Chennu**: mindre på disk eftersom det är **rensat + epokat + kompakt** men *mycket starkt* för dynamiska mått (långa clean segments, flera sedation-nivåer). ([repository.cam.ac.uk][1])
* **ds005620**: större på disk eftersom det är **råare/mer högupplöst/mer kontinuerligt**, men kan ändå vara *svagare* för era mått p.g.a. segmentlängd och protokoll.

---

Om du vill kan jag göra en ännu mer exakt sanity check med ett litet schema: “om ds005620 har X kanaler och fs=Y och Z minuter per subject, då blir storleken ungefär …” – men det kräver bara att du säger (eller tittar upp) sampling rate och antal kanaler i `*_eeg.json` för en typisk subject.

[1]: https://www.repository.cam.ac.uk/items/b7817912-50b5-423b-882e-978fb39a49df?utm_source=chatgpt.com "Research data supporting \"Brain connectivity during propofol ..."
[2]: https://www.brainproducts.com/support-resources/brainvision-core-data-format-1-0/?utm_source=chatgpt.com "BrainVision Core Data Format (BVCDF) | Brain Products GmbH"
[3]: https://www.fieldtriptoolbox.org/workshop/madrid2019/eeg_sedation/?utm_source=chatgpt.com "Details on the resting-state EEG dataset recorded with ..."
