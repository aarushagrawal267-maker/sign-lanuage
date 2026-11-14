<#
.SYNOPSIS
  Stops the backend and frontend started by start.ps1.

.DESCRIPTION
  Only the processes recorded in .run/*.pid (and their child processes) are
  stopped, after checking they are still the expected python/node processes.
#>
[CmdletBinding()]
param()

$Root = $PSScriptRoot
$Run = Join-Path $Root ".run"
$OnWindows = ($PSVersionTable.PSEdition -eq "Desktop") -or $IsWindows
$stopped = 0

function Stop-Tree([int]$ProcId) {
  if ($OnWindows) {
    & taskkill.exe /PID $ProcId /T /F 2>&1 | Out-Null
  } else {
    & pkill -TERM -P $ProcId 2>$null
    Stop-Process -Id $ProcId -ErrorAction SilentlyContinue
  }
}

foreach ($name in @("frontend", "backend")) {
  $pidFile = Join-Path $Run "$name.pid"
  if (-not (Test-Path $pidFile)) { continue }
  $procId = [int](Get-Content $pidFile)
  $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
  if ($proc) {
    if ($proc.ProcessName -match "^(python|python3|python3\.\d+|node|uvicorn)(\.exe)?$") {
      Stop-Tree $procId
      Write-Host "stopped $name (pid $procId)"
      $stopped++
    } else {
      Write-Host "pid $procId now belongs to '$($proc.ProcessName)', not $name - leaving it alone" -ForegroundColor Yellow
    }
  } else {
    Write-Host "$name was not running"
  }
  Remove-Item $pidFile -ErrorAction SilentlyContinue
}

if ($stopped -eq 0) { Write-Host "Nothing to stop." }
else { Write-Host "Sign Intelligence Studio stopped." -ForegroundColor Green }
