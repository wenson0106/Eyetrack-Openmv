$ErrorActionPreference = "Stop"

$venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.11 -m venv (Join-Path $PSScriptRoot "..\.venv")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python -c "import sys; assert sys.version_info[:2] == (3, 11), 'Python 3.11 is required'"
    python -m venv (Join-Path $PSScriptRoot "..\.venv")
} else {
    throw "Python 3.11 was not found. Install it from https://www.python.org/downloads/"
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r (Join-Path $PSScriptRoot "..\requirements.txt")
& $venvPython (Join-Path $PSScriptRoot "download_models.py")

Write-Host "Setup complete. Start with: .\.venv\Scripts\python.exe eye_server.py"
