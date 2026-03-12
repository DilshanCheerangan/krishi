<#
setup_and_run_windows.ps1

Windows helper script to (1) ensure Python is installed (via winget if available),
(2) create a virtual environment, (3) install pip requirements, and
(4) show/run commands to start the backend and serve the frontend.

Run in PowerShell (not elevated unless prompted by installer):
.
PS> .\setup_and_run_windows.ps1
#>

Write-Host "KRISHI – Knowledge-driven Real-time Intelligent System for Harvest Improvement Windows setup helper" -ForegroundColor Cyan

function Abort($msg) {
    Write-Host $msg -ForegroundColor Red
    exit 1
}

# 1) Ensure python exists
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found in PATH." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "winget found — installing latest Python 3..." -ForegroundColor Green
        winget install --id Python.Python.3 -e --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            Abort "winget failed to install Python. Please install Python manually from https://www.python.org/downloads/"
        }
        Write-Host "Python installed. You may need to close and reopen PowerShell to refresh PATH." -ForegroundColor Green
    }
    else {
        Abort "Neither Python nor winget available. Please install Python from https://www.python.org/downloads/ and re-run this script."
    }
}

# 2) create virtual environment
if (-not (Test-Path -Path .\.venv)) {
    Write-Host "Creating virtual environment in .\.venv" -ForegroundColor Cyan
    python -m venv .\.venv
    if ($LASTEXITCODE -ne 0) { Abort "Failed to create virtual environment." }
}
else {
    Write-Host "Virtual environment .\.venv already exists." -ForegroundColor Yellow
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
if (Test-Path -Path .\.venv\Scripts\Activate.ps1) {
    # Use the current session to activate so pip installs are visible here
    . .\.venv\Scripts\Activate.ps1
} else {
    Abort "Activation script not found. Ensure Python venv was created successfully."
}

Write-Host "Upgrading pip and installing requirements..." -ForegroundColor Cyan
python -m pip install --upgrade pip
if (Test-Path -Path .\requirements.txt) {
    python -m pip install -r .\requirements.txt
} else {
    Write-Host "requirements.txt not found — skipping dependency install." -ForegroundColor Yellow
}

Write-Host "\nSetup complete. Next steps:" -ForegroundColor Green
Write-Host "1) To run the backend API (development mode):" -ForegroundColor White
Write-Host "   python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor Gray
Write-Host "2) To serve the frontend from port 8080 (separate terminal):" -ForegroundColor White
Write-Host "   python -m http.server 8080 --directory frontend" -ForegroundColor Gray
Write-Host "3) Open the frontend in your browser: http://localhost:8080" -ForegroundColor White

Write-Host "\nOption: start the backend now in this window? (Y/N)" -ForegroundColor Cyan
$ans = Read-Host
if ($ans -and $ans.Trim().ToUpper() -eq 'Y') {
    Write-Host "Starting backend (uvicorn) — press Ctrl+C to stop" -ForegroundColor Green
    python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
}

Write-Host "Script finished." -ForegroundColor Cyan
