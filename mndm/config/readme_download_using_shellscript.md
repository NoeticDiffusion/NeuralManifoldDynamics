### Downloading an OpenNeuro “shell script” and placing files into a chosen folder

OpenNeuro can provide a dataset “shell script” that contains many `curl ... -o ...` lines (often with presigned S3 URLs). This is useful when normal download tooling fails or when you want a reproducible, file-by-file download list.

Example: ds003059 v1.0.0 shell script is available from the OpenNeuro download page:
- [OpenNeuro ds003059 v1.0.0 download page](https://openneuro.org/datasets/ds003059/versions/1.0.0/download#)

---

### Step 1 — Save the shell script locally

1. Open the dataset download page (link above).
2. Download the **shell script** option (it’s typically a `.sh` file containing `curl` commands).
3. Save it into this repo (recommended):
   - `openneuro/config/shellscript_download/ds003059-1.0.0.sh`

> Note: presigned S3 URLs can expire. If downloads start failing with HTTP 403/Signature errors, re-download a fresh script from OpenNeuro.

---

### Step 2 — Run the script safely via Python (recommended on Windows)

Running `.sh` scripts directly on Windows can be fragile (PowerShell parsing of `&`, quoting issues, etc.). Instead, use the repo’s Python downloader which **parses the script** and downloads files robustly.

From the repo root (PowerShell):

```powershell
python .\openneuro\src\presigned_fallback.py `
  --script ".\openneuro\config\shellscript_download\ds003059-1.0.0.sh" `
  --dataset ds003059 `
  --data-dir "E:\Science_Datasets\openneuro\received" `
  --config ".\openneuro\config\config_ingest.yaml" `
  --jobs 16 `
  --build-index
```

What this does:
- Downloads all files referenced in the script into:
  - `E:\Science_Datasets\openneuro\received\ds003059\...`
- Optionally builds `file_index.csv` into:
  - `E:\Science_Datasets\openneuro\processed\ds003059\file_index.csv`

Optional flags:
- **Overwrite existing files**:
  - add `--overwrite`
- **Dry run** (show what would be downloaded without downloading):
  - add `--dry-run`
- **If the script writes outputs under a leading folder like `ds003059-1.0.0/...`**:
  - add `--strip-first-component`
  - (the downloader also attempts to auto-detect this and strip it when appropriate)

---

### Step 3 — Compute features / summarize (ingest pipeline)

Once files are in `received/` and an index exists, you can run:

```powershell
# Optional preflight check before the first run
python -m mndm.cli prerequisite-check --dataset ds003059

python -m mndm.cli features --dataset ds003059 --n-jobs 16 --mem-budget-gb 20
python -m mndm.cli summarize --dataset ds003059
```

