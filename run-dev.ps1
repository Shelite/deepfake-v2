# Run All Services in Development Mode
# Otomatis membuka 5 terminal dan menjalankan semua services

Write-Host "🚀 Starting All Services in Development Mode..." -ForegroundColor Cyan
Write-Host ""

# Check if setup sudah dilakukan
if (-not (Test-Path "services\temporal\venv")) {
    Write-Host "❌ Setup belum dilakukan!" -ForegroundColor Red
    Write-Host "Jalankan dulu: .\setup-dev.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "📍 Starting services..." -ForegroundColor Yellow
Write-Host ""

# Start Temporal Service
Write-Host "Starting Temporal Service (port 8001)..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\services\temporal'; .\venv\Scripts\python.exe main.py"

Start-Sleep -Seconds 2

# Start Liveness Service
Write-Host "Starting Liveness Service (port 8002)..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\services\liveness'; .\venv\Scripts\python.exe main.py"

Start-Sleep -Seconds 2

# Start Hybrid Service
Write-Host "Starting Hybrid Service (port 8003)..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\services\hybrid'; .\venv\Scripts\python.exe main.py"

Start-Sleep -Seconds 2

# Start Spatial Service
Write-Host "Starting Spatial Service (port 8004)..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\services\spatial'; .\venv\Scripts\python.exe main.py"

Start-Sleep -Seconds 2

# Start Gateway
Write-Host "Starting Gateway (port 8000)..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\gateway'; .\venv\Scripts\python.exe main.py"

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "✅ All services started!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green
Write-Host ""
Write-Host "📍 Endpoints:" -ForegroundColor Cyan
Write-Host "   Gateway:  http://localhost:8000" -ForegroundColor White
Write-Host "   Temporal: http://localhost:8001" -ForegroundColor White
Write-Host "   Liveness: http://localhost:8002" -ForegroundColor White
Write-Host "   Hybrid:   http://localhost:8003" -ForegroundColor White
Write-Host "   Spatial:  http://localhost:8004" -ForegroundColor White
Write-Host ""
Write-Host "🧪 Test dengan:" -ForegroundColor Cyan
Write-Host "   curl http://localhost:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "🛑 Untuk stop semua services:" -ForegroundColor Cyan
Write-Host "   Tutup semua terminal PowerShell yang terbuka" -ForegroundColor White
Write-Host "   Atau jalankan: .\stop-dev.ps1" -ForegroundColor White
Write-Host ""
