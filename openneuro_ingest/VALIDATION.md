# Production Validation Checklist

## ✅ Implemented Robust Parallelization

### Critical Safety Features

1. **Safe CSV Aggregation**
   - ✅ Workers write to `features_<hash>.csv` per file
   - ✅ Parent merges all temp files atomically
   - ✅ Temp files cleaned up after merge
   - ✅ No concurrent writes to same file

2. **Worker Initialization**
   - ✅ `worker_init()` sets BLAS env vars before NumPy import
   - ✅ Per-worker seeds: `base_seed + worker_id`
   - ✅ Windows-compatible "spawn" backend (`backend="loky"`)

3. **Error Isolation**
   - ✅ Returns `WorkerResult` with success/error status
   - ✅ Failed files logged to `failed_files.txt`
   - ✅ Pipeline continues after individual failures

4. **File Scheduling**
   - ✅ Sort by size (small files first)
   - ✅ Stabilizes memory profile
   - ✅ Subject filter applied pre-queue

5. **Memory Monitoring**
   - ✅ `psutil` optional (graceful fallback)
   - ✅ Warning at 80% budget threshold
   - ✅ Monitoring every 3 files (responsive but not chatty)

6. **Single-Writer HDF5**
   - ✅ Only parent writes H5 (no worker access)
   - ✅ Safe concurrent architecture

7. **Requirements Pinned**
   - ✅ `psutil==5.9.8`
   - ✅ `filelock==3.16.1`

### Enhanced Logging

- ✅ Logs where merged features.csv is written
- ✅ Reports how many temp files were merged
- ✅ Shows file count after subject filtering
- ✅ Memory warnings include budget context
- ✅ Failed file count in warning message

### Edge Case Handling

- ✅ Leftover temp files cleaned up on resume
- ✅ Idempotent runs (delete temps before start)
- ✅ Subject filter applied before work queue
- ✅ Failed files manifest supports continue command

## Validation Commands

### Quick Smoke Test

```bash
# Subject-scoped, parallel features and summarize
python -m noetic_ingest.cli features --dataset ds003490 --subject 001 --n-jobs 2
python -m noetic_ingest.cli summarize --dataset ds003490 --subject 001

# Full resume (skips completed steps, re-summarizes)
python -m noetic_ingest.cli continue --dataset ds003490

# Idempotency check (optional)
python -m noetic_ingest.cli features --dataset ds003490 --subject 001
# Check features.csv hash is identical
python -m noetic_ingest.cli features --dataset ds003490 --subject 001
# Verify hash matches
```

### Robustness / QC Validation (ds003490, ds005114)

```bash
# Recompute features (optional – needed if config/features changed)
python -m noetic_ingest.cli features --dataset ds003490 --subject 001

# Summarize with robustness, ensembles, multiverse, QC
python -m noetic_ingest.cli summarize --dataset ds003490 --subject 001
```

Inspect for a test subject (e.g. `sub-001_ses-01`):

- `noetic_output/ds003490/mnps_ds003490_*/sub-001_ses-01/summary.json`  
  - `ensemble_robustness` (groups_realised, mean/var per subcoord).  
  - `robust_summary` (axes + subcoords: point, CI95, split-half).  
  - `multiverse_psd` (primary/secondary PSD methods + stability per subcoord).  
  - `entropy_qc` (provisional flags for `e_e`, `e_s`, `e_m`).  
- `noetic_output/ds003490/mnps_ds003490_*/sub-001_ses-01/qc_summary.json`  
  - `coverage` (epochs, seconds).  
  - `artifacts` (artifact methods, bad EEG channels).  
  - `ensemble` (ensemble variance stats).  
  - `reliability` (axes + subcoords split-half).  
  - `entropy_provisional` (subset of axes to treat cautiously downstream).

### Expected Behaviors

- **CSV merge is idempotent**: Running features twice yields identical `features.csv`
- **Failure path**: One bad file goes to `failed_files.txt`; continue resumes and skips it
- **Memory monitoring**: Emits warnings near configured budget without degrading throughput
- **Determinism**: Bootstrap results identical across runs with same seed and worker layout

## Documented Features

- ✅ README.md: Error handling section added
- ✅ Command_cheat_sheet.md: Failed files and temp cleanup documented
- ✅ Both documents explain `failed_files.txt` behavior
- ✅ Both documents explain resume handling of leftover temps

## Production-Ready

The implementation is now **production-ready** with:
- Safe concurrent writes (no CSV corruption)
- Robust error handling (continue past failures)
- Memory-aware (monitoring + warnings)
- Deterministic (per-worker seeds)
- Windows-compatible (spawn backend)
- Idempotent (repeatable results)
- Well-documented (README + cheat sheet)


## Implementatörens ord

Den parallelliserade pipeline-versionen är implementerad enligt ovanstående specifikation och validerad i praktiken på ds003490 med subject-filtrering. Vi använder process-baserad parallellism på filnivå, säkra temporära CSV-skrivningar per fil med sammanslagning i huvudprocessen, och en single-writer design för HDF5. Varje worker initieras med BLAS-trådar satta till 1 och deterministiska seeds (bas_seed + worker_id), vilket gör resultaten reproducerbara över körningar.

Fel i enstaka filer isoleras och loggas till `failed_files.txt`, och `continue`-kommandot återupptar körningar utan att göra om slutförda steg. Minnesövervakning (psutil) varnar vid 80% av budget och visar aktuellt kontext i loggar. Kvarvarande temp-filer städas på resume så att körningar är idempotenta.

Kort sagt: lösningen är robust för Windows-laptops, minnesmedveten, återstartbar och deterministisk. Använd `--n-jobs` och `--mem-budget-gb` för att finjustera prestanda; börja gärna med `--subject 001` för snabba valideringar.