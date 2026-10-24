$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) { & ./setup.ps1 }
& ./.venv/Scripts/python.exe -E run.py --remote
