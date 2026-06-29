# run_p2_p5_rerun.ps1
# Produces fresh H5s (row_source + features_projection_z), then runs validation notebook.
# Usage: powershell -ExecutionPolicy Bypass -File run_p2_p5_rerun.ps1

$LogDir  = "H:\SourceRepo2\NeuralManifoldDynamics\project\analysis\logs"
$OutDir  = "E:\Science_Datasets\openneuro\processed"
$DataDir = "K:\ds003645"
$Config  = "H:\SourceRepo2\NeuralManifoldDynamics\mndm\config\config_ingest_ds003645.yaml"
$NbPath  = "H:\SourceRepo2\NeuralManifoldDynamics\project\analysis\ds003645_meg_validation_package.ipynb"
$NbOut   = "H:\SourceRepo2\NeuralManifoldDynamics\project\analysis\ds003645_meg_validation_package_executed.ipynb"
$Repo    = "H:\SourceRepo2\NeuralManifoldDynamics"
$Python  = "$Repo\.venv\Scripts\python.exe"
$Jupyter = "$Repo\.venv\Scripts\jupyter.exe"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logMain = "$LogDir\run_p2_p5_$ts.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $logMain -Value $line -Encoding UTF8
}

$env:PYTHONIOENCODING = "utf-8"
Set-Location $Repo

Log "=== P2-P5 re-run started ==="
Log "Log: $logMain"

# Step 1: summarize
Log "--- Step 1: mndm.cli summarize ---"
$log1 = "$LogDir\summarize_$ts.log"
& $Python -m mndm.cli summarize `
    --dataset ds003645 `
    --config $Config `
    --data-dir $DataDir `
    --out-dir $OutDir 2>&1 | Tee-Object -FilePath $log1
$exit1 = $LASTEXITCODE
Log "Summarize exit: $exit1"
if ($exit1 -ne 0) {
    Log "ERROR: summarize failed -- see $log1"
    exit 1
}

# Step 2: verify new H5 structure
Log "--- Step 2: Verify H5 structure ---"
& $Python "H:\SourceRepo2\NeuralManifoldDynamics\project\analysis\verify_h5_structure.py" 2>&1 |
    Tee-Object -Append -FilePath $logMain

# Step 3: run validation notebook
Log "--- Step 3: Validation notebook ---"
$log3 = "$LogDir\validation_nb_$ts.log"
& $Jupyter nbconvert --to notebook --execute `
    --ExecutePreprocessor.timeout=1200 `
    --output "$NbOut" `
    "$NbPath" 2>&1 | Tee-Object -FilePath $log3
$exit3 = $LASTEXITCODE
Log "Notebook exit: $exit3"
if ($exit3 -ne 0) {
    Log "WARNING: notebook had errors -- see $log3"
} else {
    Log "Notebook OK: $NbOut"
}

# Step 4: print readiness score
Log "--- Step 4: Readiness score ---"
& $Python "H:\SourceRepo2\NeuralManifoldDynamics\project\analysis\print_readiness_score.py" 2>&1 |
    Tee-Object -Append -FilePath $logMain

Log "=== All steps done. Log: $logMain ==="
