# peer_review_batch.ps1
# =====================
# Runs all three peer-review follow-up tasks for ds006848.
#
# Task 1 + 2 run immediately from existing data.
# Task 3 starts a new 4s-window MNDM pipeline run, then post-processes it.
#
# Usage:
#   cd H:\SourceRepo2\NeuralManifoldDynamics
#   .\project\analysis\peer_review_batch.ps1 [-SkipPipeline] [-TasksOnly 1,2]
#
# Params:
#   -SkipPipeline   Skip the 4s pipeline run (use existing run dir via -RunDir4s)
#   -RunDir4s       Explicit 4s run directory (only needed with -SkipPipeline)
#   -TasksOnly      Comma-separated list of tasks to run (default: 1,2,3)

param(
    [switch]$SkipPipeline,
    [string]$RunDir4s = "",
    [string]$TasksOnly = "1,2,3"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = "H:\SourceRepo2\NeuralManifoldDynamics"
$VenvPython = "H:\SourceRepo2\NeuralManifoldDynamics\.venv\Scripts\python.exe"
$PythonExe = "python"
$OutDir = "J:\repos\NoeticDiffusion\articles\embodied_anchoring_follow_up\results\peer_review_followup"
$ProcessedDir = "J:\processed\openneuro\ds006848"

# Required PYTHONPATH for mndm CLI
$env:PYTHONPATH = "H:\SourceRepo2\NeuralManifoldDynamics\mndm\src;H:\SourceRepo2\NeuralManifoldDynamics\core\src"

$Tasks = $TasksOnly -split ","

function Invoke-Mndm($Command, $ExtraArgs) {
    # Invoke mndm CLI via the project venv using module-level entrypoint
    $allArgs = @($Command) + $ExtraArgs
    $argStr = ($allArgs | ForEach-Object { "'$_'" }) -join ","
    $expr = "import sys; sys.argv=['mndm',$argStr]; from mndm.cli import main; main()"
    & $VenvPython -c $expr
}

function Write-Step($msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host $msg -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

Set-Location $RepoRoot

# ── Task 1: EEG artifact audit ──────────────────────────────────────────────
if ($Tasks -contains "1") {
    Write-Step "Task 1: EEG artifact audit"
    & $PythonExe project\scripts\pr01_eeg_artifact_audit.py
    if ($LASTEXITCODE -ne 0) { Write-Warning "Task 1 exited with code $LASTEXITCODE" }
    else { Write-Host "[OK] Task 1 complete" -ForegroundColor Green }
}

# ── Task 2: Trial-index fatigue analysis ─────────────────────────────────────
if ($Tasks -contains "2") {
    Write-Step "Task 2: Trial-index fatigue analysis"
    & $PythonExe project\scripts\pr02_fatigue_trial_index.py
    if ($LASTEXITCODE -ne 0) { Write-Warning "Task 2 exited with code $LASTEXITCODE" }
    else { Write-Host "[OK] Task 2 complete" -ForegroundColor Green }
}

# ── Task 3: 4s window pipeline + robustness check ────────────────────────────
if ($Tasks -contains "3") {
    Write-Step "Task 3: 4s-window MNDM pipeline + F2/F3 robustness check"

    if (-not $SkipPipeline) {
        Write-Host "Starting 4s-window features run... (this will take ~1-3 hours)" -ForegroundColor Yellow
        $FeatStart = Get-Date

        Invoke-Mndm "summarize" @(
            "--config", "mndm/config/config_ingest_ds006848_4s.yaml",
            "--dataset", "ds006848",
            "--out-dir", $ProcessedDir,
            "--n-jobs", "4"
        )

        if ($LASTEXITCODE -ne 0) {
            Write-Error "summarize step failed (exit $LASTEXITCODE). Aborting Task 3."
            exit 1
        }

        if ($LASTEXITCODE -ne 0) {
            Write-Error "summarize step failed (exit $LASTEXITCODE). Aborting Task 3."
            exit 1
        }

        $FeatElapsed = (Get-Date) - $FeatStart
        Write-Host "[OK] Pipeline done in $($FeatElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green

        # Discover the new run directory (latest ds006848 run)
        $LatestRun = Get-ChildItem $ProcessedDir -Directory |
            Where-Object { $_.Name -match "neuralmanifolddynamics_ds006848_" } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($null -eq $LatestRun) {
            Write-Error "Could not find 4s run directory in $ProcessedDir"
            exit 1
        }
        $RunDir4s = $LatestRun.FullName
        Write-Host "4s run directory: $RunDir4s"
    } else {
        if (-not $RunDir4s) {
            Write-Error "-SkipPipeline requires -RunDir4s to be set."
            exit 1
        }
        Write-Host "Using existing 4s run dir: $RunDir4s"
    }

    Write-Host "Running pr03 comparison analysis..."
    & $PythonExe project\scripts\pr03_shortwindow_robustness.py `
        --run-dir-4s $RunDir4s `
        --out-dir $OutDir

    if ($LASTEXITCODE -ne 0) { Write-Warning "pr03 exited with code $LASTEXITCODE" }
    else { Write-Host "[OK] Task 3 complete" -ForegroundColor Green }
}

# ── Summary ──────────────────────────────────────────────────────────────────
Write-Step "Peer-review batch complete"
Write-Host "Outputs in: $OutDir"
Get-ChildItem $OutDir | Where-Object { -not $_.PSIsContainer } | Select-Object Name, Length, LastWriteTime
