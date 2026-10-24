$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required. Install it and retry.' }
}
& ./.venv/Scripts/python.exe -E -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
New-Item -ItemType Directory -Force -Path tools | Out-Null
if (-not (Test-Path -LiteralPath 'tools/cloudflared.exe')) {
    Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile 'tools/cloudflared.exe'
}
foreach ($weight in @('yolo11s.pt', 'yolo11n.pt', 'yolo11n-pose.pt')) {
    if (-not (Test-Path -LiteralPath $weight)) {
        Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/$weight" -OutFile $weight
    }
}
& ./.venv/Scripts/python.exe -E -m pip check
& ./.venv/Scripts/python.exe -E scripts/optimize_model.py
if ($LASTEXITCODE -ne 0) { throw 'Model optimization failed. PyTorch fallback remains available.' }
Write-Host 'Ready. Run: .\.venv\Scripts\python.exe -E run.py --remote'
