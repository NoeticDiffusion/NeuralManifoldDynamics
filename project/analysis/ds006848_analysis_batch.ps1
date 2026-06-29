# ds006848_analysis_batch.ps1
# Full ds006848 verbal-WM analysis batch -- 04b through 04f.
# Run this from the NeuralManifoldDynamics repo root.
#
# Produces a dated output package under $OutRoot, then copies a manifest
# and all CSV/TXT summary files to $HandoffDir for transfer to the
# analysis repo (NoeticDiffusion or equivalent).
#
# Usage:
#   .\project\analysis\ds006848_analysis_batch.ps1
#   .\project\analysis\ds006848_analysis_batch.ps1 -RunDir "J:/processed/.../neuralmanifolddynamics_..."
#   .\project\analysis\ds006848_analysis_batch.ps1 -DryRun

param(
    [string]$RunDir      = "J:/processed/openneuro/ds006848/neuralmanifolddynamics_ds006848_20260626_114620",
    [string]$BidsDir     = "K:/ExternalReceivedDatasets/openneuro/received/ds006848",
    [string]$OutRoot     = "J:/processed/openneuro/ds006848",
    [string]$HandoffDir  = "J:/processed/openneuro/ds006848/handoff_analysrepo",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Python = "H:\SourceRepo2\NeuralManifoldDynamics\.venv\Scripts\python.exe"
$Scripts = "H:\SourceRepo2\NeuralManifoldDynamics\project\scripts"
$Stamp   = (Get-Date -Format "yyyyMMdd_HHmmss")

function Run-Step {
    param([string]$Label, [string]$Script, [string[]]$Args)
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    if ($DryRun) {
        Write-Host "  [DRY RUN] Would run: $Python $Script $Args"
        return
    }
    & $Python $Script @Args
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$Label failed with exit code $LASTEXITCODE"
    }
}

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
$Out04b = "$OutRoot/04b_encoding_phase"
$Out04c = "$OutRoot/04c_purity_audit"
$Out04d = "$OutRoot/04d_behavioral"
$Out04e = "$OutRoot/04e_robustness"
$Out04f = "$OutRoot/04f_eeg_comparator"

# ---------------------------------------------------------------------------
# Run analyses
# ---------------------------------------------------------------------------

Run-Step "04b -- Encoding-phase MNPS analysis" `
    "$Scripts\04b_encoding_phase_analysis.py" `
    @("--run-dir", $RunDir, "--bids-dir", $BidsDir, "--out-dir", $Out04b)

Run-Step "04c -- Window-overlap purity audit" `
    "$Scripts\04c_window_overlap_purity_audit.py" `
    @("--run-dir", $RunDir, "--bids-dir", $BidsDir, "--out-dir", $Out04c)

Run-Step "04d -- Behavioral condition review" `
    "$Scripts\04d_behavioral_condition_review.py" `
    @("--bids-dir", $BidsDir, "--04b-dir", $Out04b, "--out-dir", $Out04d)

Run-Step "04e -- Subject robustness (LOO, bootstrap)" `
    "$Scripts\04e_subject_robustness.py" `
    @("--04b-dir", $Out04b, "--04c-dir", $Out04c, "--out-dir", $Out04e)

Run-Step "04f -- Classical EEG comparator" `
    "$Scripts\04f_classical_eeg_comparator.py" `
    @("--features-parquet", "$OutRoot/features.parquet",
      "--bids-dir", $BidsDir, "--out-dir", $Out04f)

# ---------------------------------------------------------------------------
# Collect handoff package
# ---------------------------------------------------------------------------
Write-Host "`n=== Assembling handoff package ===" -ForegroundColor Cyan

$PkgDir = "$HandoffDir/$Stamp"
if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $PkgDir | Out-Null }

# Copy all summary CSVs and TXTs
$SummaryDirs = @($Out04b, $Out04c, $Out04d, $Out04e, $Out04f)
foreach ($dir in $SummaryDirs) {
    $tag = Split-Path $dir -Leaf
    $dest = "$PkgDir/$tag"
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        Get-ChildItem $dir -Recurse -Include "*.csv","*.txt" |
            Copy-Item -Destination { "$dest/$($_.Name)" } -Force
    } else {
        Write-Host "  [DRY RUN] Would copy $dir CSVs/TXTs to $dest"
    }
}

# Copy claim ledger
$ClaimsFile = "H:\SourceRepo2\NeuralManifoldDynamics\project\claims\ds006848_verbal_wm_claims.md"
if (-not $DryRun) {
    Copy-Item $ClaimsFile -Destination "$PkgDir\ds006848_verbal_wm_claims.md" -Force
}

# Write package manifest
$Manifest = @'
ds006848 verbal-WM analysis package

Contents
--------
04b_encoding_phase/      Full encoding episode, common-duration, normalised bins, retrieval-by-mode
04c_purity_audit/        Window-overlap purity table, filter comparison, overlap distributions
04d_behavioral/          Accuracy, partial score, serial position, MNPS-behavior correlations
04e_robustness/          LOO Friedman, bootstrap CIs, pairwise effect sizes, rank consistency
04f_eeg_comparator/      Frontal theta/alpha/complexity condition comparison

Claim ledger: ds006848_verbal_wm_claims.md

Key validated findings
  V1: Rapid item-updating (Fast, FastDelay) higher MNPS m and d than Simultaneous/Slow during encoding.
      Friedman m p=0.0023, d p=0.0003. Survives purity filters F0/F1/F4.
  V2: Maintenance-window MNPS null across prior presentation modes.
  V3: ECG polarity correction applied (92.7% inverted QRS corrected).
  V4: WM-phase HRV gated (87.7% window contamination confirmed).

Caveat: 8 s MNPS windows. F2/F3 purity filters geometrically inapplicable
for 2.8 s Fast/Simultaneous encoding. Shorter-window rerun pending decision.
'@
$Header = "ds006848 verbal-WM analysis package`nGenerated: $Stamp`nRun dir: $RunDir`n`n"
$ManifestFull = $Header + $Manifest

if (-not $DryRun) {
    $ManifestFull | Out-File "$PkgDir\MANIFEST.txt" -Encoding UTF8
    Write-Host "Handoff package ready: $PkgDir" -ForegroundColor Green
} else {
    Write-Host "[DRY RUN] Package would be at: $PkgDir"
}
