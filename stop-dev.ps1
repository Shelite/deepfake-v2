# Stop All Development Services

Write-Host "🛑 Stopping all Python services..." -ForegroundColor Yellow
Write-Host ""

# Kill all Python processes running uvicorn/main.py
$processes = Get-Process | Where-Object {
    $_.ProcessName -like "*python*" -and 
    $_.CommandLine -like "*main.py*"
}

if ($processes) {
    Write-Host "Found $($processes.Count) service(s) running" -ForegroundColor Gray
    $processes | ForEach-Object {
        Write-Host "Stopping: $($_.ProcessName) (PID: $($_.Id))" -ForegroundColor Gray
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host ""
    Write-Host "✅ All services stopped" -ForegroundColor Green
} else {
    Write-Host "No services running" -ForegroundColor Gray
}

Write-Host ""
Write-Host "💡 To start again: .\run-dev.ps1" -ForegroundColor Cyan
Write-Host ""
