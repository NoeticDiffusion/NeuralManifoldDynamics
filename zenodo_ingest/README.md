# Zenodo ingest

Download public Zenodo records into a local received directory. The client is
built for multi-gigabyte files: it streams to disk, resumes an interrupted
transfer from a `.partial` file with an HTTP `Range` request, and checks the
checksum Zenodo publishes (usually MD5) before renaming the file into place.

A rerun of the same command skips a file when the local size still matches and
`acquisition_receipt.json` records that the checksum was already verified.
Delete that receipt to force a new checksum read.

## Requirements

- Python 3.11+
- `pyyaml` for config files

Run it from the repository root so `zenodo_ingest` is on the Python path.

## Commands

```powershell
# Show the selected files and their sizes. No payload is written.
python -m zenodo_ingest.cli list --config zenodo_ingest/config/zenodo_4456079.yaml

# Download the files selected in the config.
python -m zenodo_ingest.cli download --config zenodo_ingest/config/zenodo_4456079.yaml

# Download one record without a config file.
python -m zenodo_ingest.cli download --record 4456079 --out-dir E:\Science_Datasets\zenodo\received

# Quote globs in PowerShell so the shell does not expand them locally.
python -m zenodo_ingest.cli download --record 10.5281/zenodo.4456079 --out-dir E:\Science_Datasets\zenodo\received --include "*.zip"
```

`--record` accepts a recid (`4456079`), a DOI (`10.5281/zenodo.4456079`), or a
record URL. A sandbox URL selects `https://sandbox.zenodo.org/api`. Files land
in `<received_dir>/<resolved record id>/`.

Restricted records use a token from the environment named by
`download.token_env` (default `ZENODO_TOKEN`). The token is sent as a bearer
header and is not written to the receipt or the log.

## Config

```yaml
record: "4456079"
paths:
  received_dir: E:/Science_Datasets/zenodo/received
selection:
  include:
    - "*.zip"
  exclude:
    - "*.pdf"
download:
  workers: 2
  chunk_bytes: 8388608
  retries: 5
  timeout_seconds: 120
  resume: true
  verify_checksum: true
  continue_on_error: true
  token_env: ZENODO_TOKEN
```

`workers` runs separate files in parallel and is capped at 8. With
`continue_on_error: false` the run stops on the first failed file and stays
sequential. A failed file stays out of the final name; its `.partial` remains
so the next run can resume. The receipt lists failed keys.

`verify_checksum: false` skips a file on size alone. Leave verification on for
archives that will be measured later.

## Tests

```powershell
python -m pytest zenodo_ingest/tests -q
```
