<#
.SYNOPSIS
  Starts the backend (http://localhost:8000) and frontend (http://localhost:5173).

.PARAMETER NoBrowser
  Do not open the browser automatically.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\start.ps1
#>
[CmdletBinding()]
param([switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Run = Join-Path $Root ".run"
$OnWindows = ($PSVersionTable.PSEdition -eq "Desktop") -or $IsWindows
New-Item -ItemType Directory -Force -Path $Run | Out-Null

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

$VenvPy = if ($OnWindows) { Join-Path $Root ".venv\Scripts\python.exe" } else { Join-Path $Root ".venv/bin/python" }
$Vite = Join-Path $Root "frontend/node_modules/vite/bin/vite.js"
if (-not (Test-Path $VenvPy)) { Fail "Python environment not found. Run setup.ps1 first." }
if (-not (Test-Path $Vite)) { Fail "Frontend packages not installed. Run setup.ps1 first." }
if (-not (Test-Path (Join-Path $Root "assets/mediapipe/holistic_landmarker.task"))) {
  Write-Host "MediaPipe model missing - downloading..." -ForegroundColor Yellow
  & $VenvPy (Join-Path $Root "scripts/fetch_models.py")
}

function Test-PortBusy([int]$Port) {
  try {
    $c = New-Object System.Net.Sockets.TcpClient
    $iar = $c.BeginConnect("127.0.0.1", $Port, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne(300) -and $c.Connected
    $c.Close()
    return $ok
  } catch { return $false }
}

function Test-Running([string]$Name) {
  $pidFile = Join-Path $Run "$Name.pid"
  if (-not (Test-Path $pidFile)) { return $false }
  $procId = [int](Get-Content $pidFile)
  return [bool](Get-Process -Id $procId -ErrorAction SilentlyContinue)
}

if ((Test-Running "backend") -or (Test-Running "frontend")) {
  Fail "The application already appears to be running. Use stop.ps1 first."
}
foreach ($p in 8000, 5173) {
  if (Test-PortBusy $p) { Fail "Port $p is already in use by another program. Close it (or run stop.ps1) and try again." }
}

function Start-Bg([string]$Name, [string]$Exe, [string[]]$ArgList, [string]$Dir) {
  $out = Join-Path $Run "$Name.log"
  $err = Join-Path $Run "$Name.err.log"
  # Windows PowerShell 5.1 does not quote arguments: wrap any that contain spaces.
  $quoted = $ArgList | ForEach-Object { if ($_ -match "\s") { '"' + $_ + '"' } else { $_ } }
  $params = @{ FilePath = $Exe; ArgumentList = $quoted; WorkingDirectory = $Dir; RedirectStandardOutput = $out; RedirectStandardError = $err; PassThru = $true }
  if ($OnWindows) { $params.WindowStyle = "Hidden" }
  $p = Start-Process @params
  Set-Content -Path (Join-Path $Run "$Name.pid") -Value $p.Id
  Write-Host ("  {0,-9} pid {1,-6} log .run/{0}.log" -f $Name, $p.Id)
  return $p
}

Write-Host "Starting Sign Intelligence Studio..." -ForegroundColor Cyan
$backend = Start-Bg "backend" $VenvPy @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") (Join-Path $Root "backend")
$frontend = Start-Bg "frontend" "node" @($Vite, "--port", "5173", "--strictPort") (Join-Path $Root "frontend")

function Wait-Url([string]$Url, [int]$Seconds) {
  $deadline = (Get-Date).AddSeconds($Seconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
      if ($r.StatusCode -eq 200) { return $true }
    } catch { }
    Start-Sleep -Milliseconds 700
  }
  return $false
}

if (-not (Wait-Url "http://127.0.0.1:8000/api/health" 90)) {
  Write-Host "Backend did not become ready. Last log lines:" -ForegroundColor Red
  Get-Content (Join-Path $Run "backend.err.log") -Tail 25 -ErrorAction SilentlyContinue
  & (Join-Path $Root "stop.ps1")
  exit 1
}
if (-not (Wait-Url "http://localhost:5173/" 60)) {
  Write-Host "Frontend did not become ready. Last log lines:" -ForegroundColor Red
  Get-Content (Join-Path $Run "frontend.err.log") -Tail 25 -ErrorAction SilentlyContinue
  Get-Content (Join-Path $Run "frontend.log") -Tail 25 -ErrorAction SilentlyContinue
  & (Join-Path $Root "stop.ps1")
  exit 1
}

Write-Host ""
Write-Host "  Frontend   http://localhost:5173" -ForegroundColor Green
Write-Host "  Backend    http://localhost:8000"
Write-Host "  API docs   http://localhost:8000/docs"
Write-Host ""
Write-Host "Stop with:  powershell -ExecutionPolicy Bypass -File .\stop.ps1"
if (-not $NoBrowser -and $OnWindows) { Start-Process "http://localhost:5173" }
