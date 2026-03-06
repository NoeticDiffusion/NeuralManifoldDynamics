Här är ett komplett Python-skript som hjälper dig att filtrera fram och ladda ner 100 filer som innehåller både EEG och Propofol-data från VitalDB.

### Förberedelser

Du behöver installera biblioteket först:

```bash
pip install vitaldb pandas

```

### Python-skript för nedladdning

Skriptet gör följande:

1. Laddar ner metadata för alla spår i databasen.
2. Hittar de `caseid` som har både EEG-vågformer (t.ex. från BIS-monitorn) och Propofol-data (från TCI-pumpar).
3. Laddar ner de 100 första matchande filerna till en lokal mapp.

```python
import vitaldb
import pandas as pd
import os

# 1. Skapa en mapp för nedladdningarna
output_dir = "vitaldb_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Hämtar metadata från VitalDB...")
# 2. Ladda ner metadata om alla spår (tracks)
df_trks = vitaldb.load_trks()

# 3. Filtrera fram fall med EEG och Propofol
# Vi letar efter rå EEG-vågform (ofta 'EEG1_WAV' eller 'EEG2_WAV')
eeg_cases = set(df_trks[df_trks['tname'].str.contains('EEG.*WAV', case=False, na=False)]['caseid'])

# Vi letar efter Propofol-spår (kan heta 'PROPOFOL' eller 'PPF' i infusion-pumpar)
prop_cases = set(df_trks[df_trks['tname'].str.contains('PROPOFOL|PPF', case=False, na=False)]['caseid'])

# Hitta de fall som har båda
target_cases = list(eeg_cases.intersection(prop_cases))
target_cases.sort() # Sortera för konsistens

print(f"Hittade totalt {len(target_cases)} fall som matchar dina kriterier.")

# 4. Ladda ner de 100 första filerna
limit = 100
download_count = 0

for caseid in target_cases:
    if download_count >= limit:
        break
        
    filename = os.path.join(output_dir, f"{caseid}.vital")
    
    # Kontrollera om filen redan finns (för att kunna återuppta avbruten nedladdning)
    if os.path.exists(filename):
        print(f"Skippar {caseid}, finns redan.")
        download_count += 1
        continue
        
    try:
        print(f"Laddar ner fall {caseid} ({download_count + 1}/{limit})...")
        # VitalFile(caseid) hämtar automatiskt från VitalDB:s molnserver
        vf = vitaldb.VitalFile(caseid)
        vf.to_vital(filename)
        download_count += 1
    except Exception as e:
        print(f"Kunde inte ladda ner fall {caseid}: {e}")

print(f"\nKlar! {download_count} filer har sparats i mappen '{output_dir}'.")

```

### Viktiga saker att tänka på:

1. **Lagringsutrymme:** EEG-vågformer (samplade i 128Hz eller mer) gör filerna stora. 100 filer kan lätt ta upp **10–20 GB** eller mer. Se till att du har tillräckligt med diskutrymme.
2. **Spårnamn (Tracks):** * **EEG:** I skriptet letar jag efter `EEG.*WAV` (råvågform). Om du bara vill ha det beräknade BIS-indexet kan du ändra till `BIS/BIS`.
* **Propofol:** VitalDB använder ofta förkortningen `PPF` för Propofol i sina TCI-pumpar (t.ex. `Orchestra/PPF_CE` för effekt-site koncentration). Mitt skript täcker både `PROPOFOL` och `PPF`.


3. **Filformat:** Skriptet sparar filerna i `.vital`-format. Om du hellre vill ha dem direkt som **CSV** eller **NumPy-arrays** kan du byta ut `vf.to_vital(filename)` mot:
* `vf.to_pandas(track_names=['BIS/EEG1_WAV', 'Orchestra/PPF_RATE']).to_csv('filnamn.csv')`
* `data = vf.to_numpy(track_names=['BIS/EEG1_WAV', 'Orchestra/PPF_RATE'])`



Behöver du hjälp med att extrahera specifika tidssegment (t.ex. bara under själva induktionen) från dessa filer så kan jag hjälpa dig med det skriptet också!