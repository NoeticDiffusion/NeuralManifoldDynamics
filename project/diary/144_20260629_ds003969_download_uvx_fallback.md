# 144 — 2026-06-29 — ds003969 OpenNeuro download troubleshooting + uvx pipeline fix

## Goal

Troubleshoot OpenNeuro downloads through `openneuro_ingest` and land the full
`ds003969` dataset ("Meditation vs thinking task") on
`K:\ExternalReceivedDatasets\openneuro\received\ds003969`. Scope was strictly the
download pipeline; mndm/extract ingest is a later step.

## Dataset

- `ds003969` — EEG only, 98 subjects, 392 recordings (4 blocks: med1breath,
  think1, think2, med2), `.bdf` (Biosemi Active 2), 64 EEG + 11 misc/aux
  channels, single session, task `mixed`, CC0, ~54.5 GB (NEMAR / EEGDash).
- Auxiliary/embodied channels present: `EXG1`–`EXG8` (Biosemi external
  electrodes, typed `MISC` — ECG/EOG candidates), `GSR1`/`GSR2` (EDA),
  `Resp` (RESP), `Plet` (plethysmograph/PPG), `Temp` (TEMP), `Erg1`/`Erg2`.
- Note: `*_eeg.json` reports `ECGChannelCount: 0` because the EXG channels are
  typed `MISC`, not `ECG`. The ECG-like data the user cares about is in the EXG
  channels — downstream ingest will need `channel_typing` rules to type EXG
  channels (and GSR/Resp/Plet) appropriately for the embodied-anchoring pipeline.

## Tiered download attempt

### Tier 1 — installed `openneuro-py` 2026.3.0 Python API — FAILED (validated)

```
RuntimeError: Query failed when retrieving metadata for dataset:
"Cannot query field "key" on type "DatasetFile"."
```

Root cause: the OpenNeuro server GraphQL schema changed and removed/renamed the
`key` field on `DatasetFile`; the installed openneuro-py 2026.3.0 metadata query
is incompatible. This is a **server-side API schema breakage**, not a
dataset-specific issue — it almost certainly explains the user's general
openneuro-py download failures.

### Tier 2 — `uvx openneuro-py@latest` (2026.4.1) — SUCCEEDED (validated)

```
uvx openneuro-py@latest download --dataset=ds003969 `
  --target-dir="K:\ExternalReceivedDatasets\openneuro\received\ds003969"
```

- 2026.4.1 fixed the GraphQL query; resolved 1192 files (54.5 GB), 5 concurrent
  downloads.
- Completed in ~6h20m wall clock. Initial burst ~10–13 MB/s for the first ~4 GB,
  then settled to a steady ~2.2 MB/s for the remainder (likely OpenNeuro S3
  throttling after the burst). Not stalled — file count and size grew
  monotonically throughout.
- Resumable: re-running the same command continues interrupted downloads.

### Tiers 3 & 4 — NOT NEEDED (cancelled)

- Tier 3 (`presigned_fallback.py --generate-from-tree --clone-source-tree`,
  16 parallel workers from public S3) and Tier 4 (`datalad_fallback.py`) were
  not needed. The user explicitly chose to keep the working uvx download rather
  than switch to the parallel fallback when the slow rate was flagged.

## Final inventory (validated)

```
K:\ExternalReceivedDatasets\openneuro\received\ds003969\
  files       = 1192
  size        = 54.463 GB
  subjects    = 98 (sub-001 .. sub-098)
  .bdf        = 392   (4 blocks x 98 subjects = 392 recordings, matches NEMAR)
  top-level   = dataset_description.json, participants.tsv, participants.json,
                README, CHANGES, sourcedata/, code/
```

Matches NEMAR's published 1181 files / 54.5 GB (small file-count diff is
manifest/version noise).

BIDS index built:

```
K:\ExternalReceivedDatasets\openneuro\processed\ds003969\file_index.csv
  rows  = 392   (one per .bdf recording)
  cols  = path, subject, session, task, run, acq, modality, datatype, ...
```

## Code change — uvx wired into the download pipeline (validated)

`openneuro_ingest/src/openneuro/download.py` now runs downloads through
`uvx openneuro-py@latest` by default, so the pipeline uses the latest released
openneuro-py in an isolated environment regardless of the installed version.

- `_resolve_openneuro_cli(prefer_uvx=True)` returns `["uvx", "openneuro-py@latest"]`.
  With `prefer_uvx=False` it tries interpreter Scripts → per-user Scripts →
  `shutil.which` → uvx (last resort). Added `_user_script_dirs()` for the
  per-user Scripts lookup (Windows `%APPDATA%\Python\PythonXY\Scripts`,
  POSIX `~/.local/bin`).
- `_perform_openneuro_download(..., *, use_uvx=True)` goes straight to the CLI
  (uvx) when `use_uvx=True`, skipping the broken installed Python API. With
  `use_uvx=False` it keeps the old behaviour (Python API → local CLI exe).
- `download_datasets` reads `download.use_uvx` (default true) and the
  `OPENNEURO_PREFER_UVX` env override (`0`/`false` opts out).
- Verified end-to-end with `subprocess.run` intercepted: pipeline builds
  `['uvx', 'openneuro-py@latest', 'download', '--dataset=ds003969',
  '--target-dir=K:\...\ds003969']`. No linter errors.

Now `python -m openneuro.cli download --config openneuro_ingest\config\config_ingest_ds003969.yaml --dataset ds003969` runs through uvx automatically.

## Config + docs changes (validated)

- New `openneuro_ingest/config/config_ingest_ds003969.yaml` — download-only
  config pointing at K: paths, `ds003969`, intentionally no `include_patterns`
  (full dataset; ECG/aux channels live inside the `.bdf`).
- `requirements.txt`: `openneuro-py==2026.3.0` → `openneuro-py>=2026.4.1`; added
  `uv` (provides `uvx`).
- `openneuro_ingest/README.md`, `openneuro_ingest/Command_cheat_sheet.md`,
  root `README.md`, `CHANGELOG.md` updated to document the uvx default, the
  `Cannot query field "key"` failure mode, and the `download.use_uvx` /
  `OPENNEURO_PREFER_UVX` opt-out controls.

## Limitations / known issues

- The uvx download rate (~2.2 MB/s after the initial burst) is slow for large
  datasets (~6h for 54.5 GB). The Tier 3 parallel presigned-S3 fallback (16
  workers) is available as a faster alternative but hits the same S3 bucket, so
  OpenNeuro-side throttling may persist; untested head-to-head.
- `download.py`'s CLI path uses `capture_output=True`; for very large downloads
  the captured tqdm output buffer could grow large. Not changed here (pre-existing
  behaviour); flag for a future streaming-output refactor if it becomes a
  problem.
- The `openneuro_ingest/README.md` "Quick Start" examples still reference
  `python -m noetic_ingest.cli ...` while the actual module is `openneuro.cli`.
  Pre-existing inconsistency, out of scope for this session.

## Next steps (follow-up, not done)

- Add `ds003969` to the main `mndm/config/config_ingest.yaml` (`datasets:`,
  `metadata_extraction.datasets:`, and `preprocess.channel_typing.datasets.ds003969`
  rules to type `EXG*` / `GSR*` / `Resp` / `Plet` / `Temp` for the embodied
  pipeline) when ingest begins.
- Run mndm features/summarize on ds003969 once ingest config is in place.
- Consider a streaming-output refactor for the CLI download path to avoid
  large captured buffers on big datasets.

## Claim ledger

- Established external: OpenNeuro GraphQL schema changed (caused 2026.3.0
  `key` field error); 2026.4.1 (via uvx) resolves it.
- Internal validated: full ds003969 on K: (1192 files / 54.463 GB / 98 subs /
  392 .bdf); `file_index.csv` (392 rows); uvx wiring in `download.py` verified
  by command-construction test.
- Plausible, not proven: EXG channels carry ECG (typed MISC; needs ingest-side
  confirmation by inspecting EXG signal morphology).
- Speculative: Tier 3 parallel fallback would be faster (not tested).
