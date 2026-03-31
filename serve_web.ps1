param(
    [string]$Config = "configs/experiments/tfidf_tuned_v2_final.yaml",
    [string]$ServerHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$PrintOnly
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    throw "Project virtual environment was not found at '$pythonPath'."
}

$resolvedConfig = if ([System.IO.Path]::IsPathRooted($Config)) {
    $Config
} else {
    Join-Path $repoRoot $Config
}

$env:PYTHONPATH = Join-Path $repoRoot "src"
$arguments = @(
    "-m"
    "imdb_sentiment.cli"
    "serve-web"
    "--config"
    $resolvedConfig
    "--host"
    $ServerHost
    "--port"
    $Port
)

if ($PrintOnly) {
    Write-Output "PYTHONPATH=$env:PYTHONPATH"
    Write-Output "$pythonPath $($arguments -join ' ')"
    return
}

& $pythonPath @arguments
