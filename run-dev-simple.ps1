# Run All Services - Simple Version (No Emoji)

Write-Host "Starting All Services in Development Mode..." -ForegroundColor Cyan
Write-Host ""

# Check if setup sudah dilakukan
if (-not (Test-Path "backend\services\temporal\venv")) {
    Write-Host "ERROR: Setup belum dilakukan!" -ForegroundColor Red
    Write-Host "Jalankan dulu setup manual:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "python -m venv backend\services\temporal\venv" -ForegroundColor Gray
    Write-Host ".\backend\services\temporal\venv\Scripts\pip.exe install -r backend\services\temporal\requirements.txt" -ForegroundColor Gray
    Write-Host ""
    Write-Host "(Ulangi untuk liveness, hybrid, spatial, gateway)" -ForegroundColor Gray
    exit 1
}

Write-Host "Starting services..." -ForegroundColor Yellow
Write-Host ""

# Start Temporal Service
Write-Host "Starting Temporal Service on port 8001..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\backend\services\temporal'; .\venv\Scripts\python.exe main.py"

Start-Sleep -Seconds 2

# Start Liveness Service
Write-Host "Starting Liveness Service on port 8002..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\backend\services\liveness'; .\venv\Scripts\python.exe main.py"

Start-Sleep -Seconds 2

# Start Hybrid Service
Write-Host "Starting Hybrid Service on port 8003..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\backend\services\hybrid'; .\venv\Scripts\python.exe main.py"

Start-Sleep -Seconds 2

# Start Spatial Service
Write-Host "Starting Spatial Service on port 8004..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\backend\services\spatial'; .\venv\Scripts\python.exe main.py"

Start-Sleep -Seconds 2

# Start Gateway
Write-Host "Starting Gateway on port 8000..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\backend\gateway'; .\venv\Scripts\python.exe main.py"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "All services started!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Endpoints:" -ForegroundColor Cyan
Write-Host "   Gateway:  http://localhost:8000" -ForegroundColor White
Write-Host "   Temporal: http://localhost:8001" -ForegroundColor White
Write-Host "   Liveness: http://localhost:8002" -ForegroundColor White
Write-Host "   Hybrid:   http://localhost:8003" -ForegroundColor White
Write-Host "   Spatial:  http://localhost:8004" -ForegroundColor White
Write-Host ""
Write-Host "Test:" -ForegroundColor Cyan
Write-Host "   curl http://localhost:8000/health" -ForegroundColor White
Write-Host ""
