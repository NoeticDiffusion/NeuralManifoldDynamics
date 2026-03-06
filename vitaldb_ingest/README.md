# VitalDB Ingest

Config-driven downloader for VitalDB cases, focused on extracting recordings
that contain both EEG and propofol-related tracks.

## Setup

Install required packages in your Python environment:

```bash
pip install vitaldb pandas pyyaml numpy
```

For raw `.vital` downloads, set credentials in environment variables:

```bash
set VITALDB_ID=your_id
set VITALDB_PW=your_password
```

## Configure

Edit `vitaldb_ingest/config/vitaldb_config.yml`:

- `paths.received_dir`: where downloaded case files are written
- `selection.*`: track filters and number of cases
- `selection.min_eeg_tracks_per_case`: enforce minimum EEG channels per case
- `selection.include_case_ids` / `exclude_case_ids`: explicitly steer which cases to keep/drop
- `download.interval_seconds`: resampling interval for `vitaldb.load_case`
- `export.file_format`: `csv.gz` (resampled export) or `none`
- `export.raw_vital.enabled`: download original `.vital` files to `vital_files/`
- `auth.id_env_var` / `auth.pw_env_var`: env var names used for VitalDB login

Default target directory is:

- `E:/Science_Datasets/vitaldb/received`

## Run

From repo root:

```bash
python -m vitaldb_ingest.script.download_vitaldb --config vitaldb_ingest/config/vitaldb_config.yml
```

Outputs:

- `cases/case-<id>.csv.gz` time series per case
- `vital_files/<caseid>.vital` original VitalDB raw file (when enabled)
- `metadata/case_manifest.csv` summary table
- `metadata/download_manifest.jsonl` per-case download log

## Full-resolution EEG

Use `download.interval_seconds: 0.0078125` for 128 Hz export (1/128 s).
Larger files are expected at this setting.

## MNPS note

Current VitalDB selection uses BIS waveform channels (`BIS/EEG1_WAV`, `BIS/EEG2_WAV`).
That is enough for baseline MNPS ingest. For mndm advanced multi-channel modules
(synchrony ROI pairs, richer graph topologies), you may need a custom config
override because default ROI pairs (e.g. `F3-P3`) are not present in BIS naming.
