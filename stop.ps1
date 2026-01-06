# Stop Deepfake Detection System

Write-Host "🛑 Stopping Deepfake Detection System..." -ForegroundColor Yellow
Write-Host ""

docker-compose down

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ All services stopped" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Failed to stop services" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "💡 To start again, run: .\start.ps1" -ForegroundColor Cyan
Write-Host ""
