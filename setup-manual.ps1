# Manual Setup - Step by Step
# Jalankan command ini satu per satu di PowerShell

Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "Setup Development Environment - Manual Mode" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/5] Setting up Temporal Service..." -ForegroundColor Yellow
if (-not (Test-Path "services\temporal\venv")) {
    python -m venv services\temporal\venv
    .\services\temporal\venv\Scripts\python.exe -m pip install --upgrade pip
    .\services\temporal\venv\Scripts\pip.exe install -r services\temporal\requirements.txt
    Write-Host "DONE: Temporal" -ForegroundColor Green
} else {
    Write-Host "SKIP: Temporal already exists" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[2/5] Setting up Liveness Service..." -ForegroundColor Yellow
if (-not (Test-Path "services\liveness\venv")) {
    python -m venv services\liveness\venv
    .\services\liveness\venv\Scripts\python.exe -m pip install --upgrade pip
    .\services\liveness\venv\Scripts\pip.exe install -r services\liveness\requirements.txt
    Write-Host "DONE: Liveness" -ForegroundColor Green
} else {
    Write-Host "SKIP: Liveness already exists" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[3/5] Setting up Hybrid Service..." -ForegroundColor Yellow
if (-not (Test-Path "services\hybrid\venv")) {
    python -m venv services\hybrid\venv
    .\services\hybrid\venv\Scripts\python.exe -m pip install --upgrade pip
    .\services\hybrid\venv\Scripts\pip.exe install -r services\hybrid\requirements.txt
    Write-Host "DONE: Hybrid" -ForegroundColor Green
} else {
    Write-Host "SKIP: Hybrid already exists" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[4/5] Setting up Spatial Service..." -ForegroundColor Yellow
if (-not (Test-Path "services\spatial\venv")) {
    python -m venv services\spatial\venv
    .\services\spatial\venv\Scripts\python.exe -m pip install --upgrade pip
    .\services\spatial\venv\Scripts\pip.exe install -r services\spatial\requirements.txt
    Write-Host "DONE: Spatial" -ForegroundColor Green
} else {
    Write-Host "SKIP: Spatial already exists" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[5/5] Setting up Gateway..." -ForegroundColor Yellow
if (-not (Test-Path "gateway\venv")) {
    python -m venv gateway\venv
    .\gateway\venv\Scripts\python.exe -m pip install --upgrade pip
    .\gateway\venv\Scripts\pip.exe install -r gateway\requirements.txt
    Write-Host "DONE: Gateway" -ForegroundColor Green
} else {
    Write-Host "SKIP: Gateway already exists" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Creating symlinks for weights..." -ForegroundColor Yellow
$ErrorActionPreference = 'SilentlyContinue'
New-Item -ItemType SymbolicLink -Path "services\temporal\weights" -Target "..\..\backend\weights" | Out-Null
New-Item -ItemType SymbolicLink -Path "services\liveness\weights" -Target "..\..\backend\weights" | Out-Null
New-Item -ItemType SymbolicLink -Path "services\hybrid\weights" -Target "..\..\backend\weights" | Out-Null
New-Item -ItemType SymbolicLink -Path "services\spatial\weights" -Target "..\..\backend\weights" | Out-Null
$ErrorActionPreference = 'Continue'

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next: Run services with:" -ForegroundColor Cyan
Write-Host "   .\run-dev-simple.ps1" -ForegroundColor White
Write-Host ""
