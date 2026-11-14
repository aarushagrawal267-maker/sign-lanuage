<#
.SYNOPSIS
  One-time setup for Sign Intelligence Studio on Windows.

.DESCRIPTION
  * Creates a Python virtual environment (.venv) with Python 3.12 or 3.11
  * Installs PyTorch (CUDA build if an NVIDIA GPU is detected, otherwise CPU)
  * Installs the backend requirements and the frontend Node packages
  * Downloads the MediaPipe Holistic model (~13 MB) and initialises the SQLite database

  It does NOT download the WLASL dataset and does NOT start training.

.PARAMETER Cpu
  Force the CPU-only PyTorch build even if an NVIDIA GPU is present.

.PARAMETER Dev
  Also install test dependencies (pytest, Playwright test runner).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup.ps1
#>
[CmdletBinding()]
param(
  [switch]$Cpu,
  [switch]$Dev
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
$OnWindows = ($PSVersionTable.PSEdition -eq "Desktop") -or $IsWindows

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

function Invoke-Checked {
  param([string]$Exe, [string[]]$Arguments)
  & $Exe @Arguments
  if ($LASTEXITCODE -ne 0) { Fail "command failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

# ------------------------------------------------------------------ Python
Step "Looking for Python 3.12 / 3.11"
$PyCmd = $null
$PyArgs = @()
$candidates = @()
if (Get-Command py -ErrorAction SilentlyContinue) { $candidates += , @("py", "-3.12"); $candidates += , @("py", "-3.11") }
foreach ($n in @("python3.12", "python3.11", "python", "python3")) {
  if (Get-Command $n -ErrorAction SilentlyContinue) { $candidates += , @($n) }
}
foreach ($c in $candidates) {
  $exe = $c[0]; $a = @($c | Select-Object -Skip 1)
  try {
    $v = & $exe @a -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -eq 0 -and ($v -eq "3.12" -or $v -eq "3.11")) { $PyCmd = $exe; $PyArgs = $a; Ok "using $exe $($a -join ' ') (Python $v)"; break }
  } catch { }
}
if (-not $PyCmd) {
  Fail "Python 3.12 or 3.11 is required (MediaPipe does not yet support every newer version).`n       Install it with:  winget install Python.Python.3.12`n       then re-run setup.ps1"
}

# ------------------------------------------------------------------ Node
Step "Checking Node.js"
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
  Fail "Node.js 20+ is required. Install it with:  winget install OpenJS.NodeJS.LTS"
}
$nodeMajor = [int]((node --version).TrimStart("v").Split(".")[0])
if ($nodeMajor -lt 20) { Fail "Node.js 20+ is required (found $(node --version))." }
Ok "Node $(node --version)"

# ------------------------------------------------------------------ venv
Step "Creating virtual environment (.venv)"
$VenvPy = if ($OnWindows) { Join-Path $Root ".venv\Scripts\python.exe" } else { Join-Path $Root ".venv/bin/python" }
if (-not (Test-Path $VenvPy)) {
  Invoke-Checked $PyCmd ($PyArgs + @("-m", "venv", ".venv"))
  Ok "created"
} else { Ok "already exists" }
& $VenvPy -m pip --version 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Invoke-Checked $VenvPy @("-m", "ensurepip", "--upgrade") }
Invoke-Checked $VenvPy @("-m", "pip", "install", "--upgrade", "pip", "--quiet")

# ------------------------------------------------------------------ PyTorch
Step "Installing PyTorch"
$hasTorch = $false
try { & $VenvPy -c "import torch" 2>$null; $hasTorch = ($LASTEXITCODE -eq 0) } catch { }
if ($hasTorch) {
  Ok "already installed: $(& $VenvPy -c 'import torch; print(torch.__version__, ''CUDA'' if torch.cuda.is_available() else ''CPU'')')"
} else {
  $gpu = $false
  if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    try { nvidia-smi -L | Out-Null; $gpu = ($LASTEXITCODE -eq 0) } catch { }
  }
  if ($gpu) {
    Ok "NVIDIA GPU detected - installing the CUDA 12.6 build"
    & $VenvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { Warn "CUDA build failed; falling back to CPU build"; $gpu = $false }
  }
  if (-not $gpu) {
    Ok "installing the CPU build"
    Invoke-Checked $VenvPy @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
  }
}

# ------------------------------------------------------------------ backend deps
Step "Installing backend requirements"
$req = if ($Dev) { "backend/requirements-dev.txt" } else { "backend/requirements.txt" }
Invoke-Checked $VenvPy @("-m", "pip", "install", "-r", $req, "--quiet")
Ok "done"

# ------------------------------------------------------------------ frontend deps
Step "Installing frontend packages (npm)"
Push-Location (Join-Path $Root "frontend")
try {
  $npm = if ($OnWindows) { "npm.cmd" } else { "npm" }
  if (Test-Path "package-lock.json") { Invoke-Checked $npm @("ci", "--no-fund", "--no-audit") }
  else { Invoke-Checked $npm @("install", "--no-fund", "--no-audit") }
  Invoke-Checked "node" @("scripts/copy-mediapipe-wasm.mjs")
} finally { Pop-Location }
Ok "done"

# ------------------------------------------------------------------ MediaPipe model
Step "Downloading MediaPipe Holistic model"
Invoke-Checked $VenvPy @("scripts/fetch_models.py")

# ------------------------------------------------------------------ database
Step "Initialising the local database"
Push-Location (Join-Path $Root "backend")
try { Invoke-Checked $VenvPy @("-c", "from app.db.session import get_engine; get_engine(); print('    data/app.db ready')") }
finally { Pop-Location }

# ------------------------------------------------------------------ optional tools
Step "Optional tools"
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { Ok "ffmpeg found (needed to convert ASLPro .swf videos)" }
else { Warn "ffmpeg not found - ASLPro videos (162 of 2,038) cannot be converted. Install with: winget install Gyan.FFmpeg" }

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Start the application with:  powershell -ExecutionPolicy Bypass -File .\start.ps1"
Write-Host "Nothing has been downloaded from WLASL yet - load annotations and videos from the Dataset page."
