# NEMAR ingest

`nemar_ingest` downloads public NEMAR BIDS releases into a received/raw tree
like the existing OpenNeuro ingest. It uses the versioned anonymous NEMAR data
plane and `manifest.json`, then verifies manifest-declared size and checksum
when present. Annexed files use SHA-256. Files stored in the git tree use
`checksum_algorithm: git`, which is the SHA-1 of `blob <nbytes>\\0` plus the
file bytes. `backend: cli`/`backend: auto` can use the official `nemar` CLI;
the default `manifest` backend needs no Node installation.

Every finished manifest run writes an atomic `acquisition_receipt.json` below the
dataset root. It records the explicit release, manifest URL and SHA-256 of the
manifest response, plus the ordered selected file entries. With
`continue_on_error` (the default), a per-file failure such as HTTP 500 is
recorded under `failed_entries` and the loop continues. Files already verified
on disk are skipped, so a rerun retries only the failed paths. The process
exits non-zero when any file failed. A receipt from a
different dataset release is rejected rather than allowing a new release to
be claimed in the old tree. In `backend: auto`, only an absent CLI executable
falls back to the manifest backend; a CLI process failure is propagated.

## `on005385` release

The supplied config pins `on005385` to `v1.0.0`, leaves selection empty so every
manifest path is kept, and writes the tree to
`K:/ExternalReceivedDatasets/nemar/on005385`. The release is about 50 GB:

```powershell
python -m nemar_ingest.cli --config nemar_ingest/config/on005385.yaml --dry-run
python -m nemar_ingest.cli --config nemar_ingest/config/on005385.yaml
```

NEMAR file URLs follow
`https://data.nemar.org/<dataset>/<version>/<bids-path>` and its manifest is
the corresponding `.../<version>/manifest.json`. The official CLI equivalent
is `nemar dataset download <id>` with BIDS filters.

Validation is fully bounded and synthetic:

```text
python -m pytest nemar_ingest/tests -q
16 passed
```

No live NEMAR payload was transferred in this environment; run the pinned
`on005385` pilot with network access before downstream preprocessing.
