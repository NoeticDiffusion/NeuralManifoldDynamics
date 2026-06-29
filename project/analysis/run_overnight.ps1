<#
.SYNOPSIS
    Overnight MEG validation run for ds003645.

.DESCRIPTION
    Executes the MEG-EEG NMD validation pipeline in sequence:
      1. Re-run ds003645_meg_eeg_comparison.ipynb  (A3 bug fix applied)
      2. Run  ds003645_meg_validation_package.ipynb (A0/A1/A4/B1/B3/C1/C2/C3/C4/E2/E3/F3)
      3. Run  mndm pipeline with 4s window config
      4. Run  mndm pipeline with 2s window config

    Logs are written to project/analysis/logs/ with timestamps.
    Each step waits for completion before the next starts.

.NOTES
    Run from any directory; all paths are absolute.
    Expected runtime: 6-8 hours.
#>

$ErrorActionPreference = "Continue"
$REPO = "H:\SourceRepo2\NeuralManifoldDynamics"
$PYTHON = "$REPO\.venv\Scripts\python.exe"
$CONFIG_DIR = "$REPO\mndm\config"
$NB_DIR = "$REPO\project\analysis"
$LOG_DIR = "$NB_DIR\logs"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

function Write-Step {
    param([string]$msg)
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host ""
    Write-Host "[$ts] === $msg ===" -ForegroundColor Cyan
}

function Run-Step {
    param(
        [string]$label,
        [string[]]$cmd,
        [string]$logfile,
        [string]$workdir = $REPO
    )
    $ts_start = Get-Date
    Write-Step $label
    Write-Host "CMD: $($cmd -join ' ')"
    Write-Host "LOG: $logfile"
    & $cmd[0] $cmd[1..($cmd.Length-1)] 2>&1 | Tee-Object -FilePath $logfile
    $exit_code = $LASTEXITCODE
    $elapsed = [math]::Round(((Get-Date) - $ts_start).TotalMinutes, 1)
    if ($exit_code -eq 0) {
        Write-Host "[$label] DONE in ${elapsed}m (exit 0)" -ForegroundColor Green
    } else {
        Write-Host "[$label] FINISHED WITH EXIT CODE $exit_code in ${elapsed}m" -ForegroundColor Yellow
        Write-Host "  Check $logfile for details"
    }
    return $exit_code
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$REPO\mndm\src;$REPO\core\src"

# ─── Step 1: Re-run comparison notebook (A3 bug fix) ───────────────────────
$NB1 = "$NB_DIR\ds003645_meg_eeg_comparison.ipynb"
$LOG1 = "$LOG_DIR\step1_comparison_nb_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Step "Step 1 / 4: Re-run comparison notebook (A3 bug fix in Section 7b)"
Run-Step `
    "comparison_notebook" `
    @($PYTHON, "-m", "jupyter", "nbconvert",
      "--to", "notebook",
      "--execute",
      "--inplace",
      "--ExecutePreprocessor.timeout=7200",
      "--ExecutePreprocessor.kernel_name=python3",
      $NB1) `
    $LOG1

# ─── Step 2: Run validation notebook ────────────────────────────────────────
$NB2 = "$NB_DIR\ds003645_meg_validation_package.ipynb"
$LOG2 = "$LOG_DIR\step2_validation_nb_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Step "Step 2 / 4: Run MEG validation notebook (A0/A1/A4/B1/B3/C1/C2/C3/C4/E2/E3/F3)"
Run-Step `
    "validation_notebook" `
    @($PYTHON, "-m", "jupyter", "nbconvert",
      "--to", "notebook",
      "--execute",
      "--inplace",
      "--ExecutePreprocessor.timeout=7200",
      "--ExecutePreprocessor.kernel_name=python3",
      $NB2) `
    $LOG2

# ─── Step 3: 4s window pipeline ──────────────────────────────────────────────
$CFG_4S = "$CONFIG_DIR\config_ingest_ds003645_4s.yaml"
$LOG3   = "$LOG_DIR\step3_pipeline_4s_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Step "Step 3 / 4: 4s window pipeline (config_ingest_ds003645_4s.yaml)"
Write-Host "Output -> E:\Science_Datasets\openneuro\processed_4s\ds003645"
Run-Step `
    "pipeline_4s" `
    @($PYTHON, "-m", "mndm.cli", "all",
      "--config", $CFG_4S,
      "--n-jobs", "2",
      "--mem-budget-gb", "20") `
    $LOG3

# ─── Step 4: 2s window pipeline ──────────────────────────────────────────────
$CFG_2S = "$CONFIG_DIR\config_ingest_ds003645_2s.yaml"
$LOG4   = "$LOG_DIR\step4_pipeline_2s_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Step "Step 4 / 4: 2s window pipeline (config_ingest_ds003645_2s.yaml)"
Write-Host "Output -> E:\Science_Datasets\openneuro\processed_2s\ds003645"
Run-Step `
    "pipeline_2s" `
    @($PYTHON, "-m", "mndm.cli", "all",
      "--config", $CFG_2S,
      "--n-jobs", "2",
      "--mem-budget-gb", "20") `
    $LOG4

# ─── Summary ──────────────────────────────────────────────────────────────────
Write-Step "All steps completed"
Write-Host "Logs in: $LOG_DIR"
Write-Host ""
Write-Host "Next morning: check meg_readiness_score.json in"
Write-Host "  E:\Science_Datasets\openneuro\processed\ds003645\meg_eeg_comparison\"
Write-Host ""
Write-Host "Then add window_robustness score from 4s/2s pipeline results."
