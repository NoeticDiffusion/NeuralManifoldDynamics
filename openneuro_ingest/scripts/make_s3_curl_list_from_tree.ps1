param(
    [Parameter(Mandatory=$true)]
    [string]$DatasetId,

    # A local dataset tree to use as the file manifest (e.g. a datalad/git-annex checkout
    # containing placeholder files). We only use relative paths; contents are downloaded
    # from S3.
    [Parameter(Mandatory=$true)]
    [string]$SourceTree,

    # Output script file containing `curl ... -o <relative-path>` lines.
    [Parameter(Mandatory=$true)]
    [string]$OutScript,

    # Base URL for OpenNeuro public bucket (default works for public datasets).
    [string]$S3Base = "https://s3.amazonaws.com/openneuro.org",

    # Exclude derivatives/ (default: true)
    [switch]$IncludeDerivatives,

    # Optional include filter (regex on POSIX-style relative path). If provided, only
    # matching paths are included.
    [string]$IncludeRegex = "",

    # Additional exclude filter (regex on POSIX-style relative path).
    [string]$ExcludeRegex = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $SourceTree)) {
    throw "SourceTree not found: $SourceTree"
}

$sourceRoot = (Resolve-Path -LiteralPath $SourceTree).Path.TrimEnd("\")
$outPath = (Resolve-Path -LiteralPath (Split-Path -Parent $OutScript)).Path
$outFile = Join-Path $outPath (Split-Path -Leaf $OutScript)

Write-Host "DatasetId: $DatasetId"
Write-Host "SourceTree: $sourceRoot"
Write-Host "OutScript: $outFile"
Write-Host "S3Base: $S3Base"

# Collect files
$files = Get-ChildItem -LiteralPath $sourceRoot -Recurse -File -Force |
    Where-Object {
        # Drop internal git/datalad dirs
        $_.FullName -notmatch "\\\.git(\\|$)" -and
        $_.FullName -notmatch "\\\.datalad(\\|$)"
    } |
    ForEach-Object {
        $relWin = $_.FullName.Substring($sourceRoot.Length + 1)
        $relPosix = ($relWin -replace "\\","/")
        $relPosix
    }

if (-not $IncludeDerivatives) {
    $files = $files | Where-Object { $_ -notmatch "(^|/)derivatives(/|$)" }
}

if ($IncludeRegex -and $IncludeRegex.Trim().Length -gt 0) {
    $rxInc = [regex]$IncludeRegex
    $files = $files | Where-Object { $rxInc.IsMatch($_) }
}

if ($ExcludeRegex -and $ExcludeRegex.Trim().Length -gt 0) {
    $rxExc = [regex]$ExcludeRegex
    $files = $files | Where-Object { -not $rxExc.IsMatch($_) }
}

$files = $files | Sort-Object -Unique

if (-not $files -or $files.Count -eq 0) {
    throw "No files matched. Check SourceTree / filters."
}

Write-Host ("Matched files: {0}" -f $files.Count)

# Write curl lines. Keep output paths dataset-relative (no dsXXXXXX-<ver>/ prefix),
# so presigned_fallback.py will write into <data-dir>/<datasetId>/<relative-path>.
$lines = @()
foreach ($rel in $files) {
    $url = "$S3Base/$DatasetId/$rel"
    # Use simple curl lines that presigned_fallback can parse:
    $lines += ("curl --create-dirs {0} -o {1}" -f $url, $rel)
}

Set-Content -LiteralPath $outFile -Value $lines -Encoding utf8
Write-Host "Wrote curl list: $outFile"


