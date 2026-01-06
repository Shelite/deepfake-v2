# Deepfake Detection System - Quick Start Script
# Run this script to start all services

Write-Host "🚀 Starting Deepfake Detection System..." -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "🐳 Checking Docker..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check if model weights exist
Write-Host ""
Write-Host "📦 Checking model weights..." -ForegroundColor Yellow
$weights = @(
    "backend\weights\model_v10_optimized.h5",
    "backend\weights\rppg_model_lite.pth",
    "backend\weights\model_modul4_generalized.h5",
    "backend\weights\xception_modul1_final2.keras"
)

$missing = @()
foreach ($weight in $weights) {
    if (Test-Path $weight) {
        Write-Host "✅ Found: $weight" -ForegroundColor Green
    } else {
        Write-Host "❌ Missing: $weight" -ForegroundColor Red
        $missing += $weight
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "⚠️  Missing model weights. Please download them first." -ForegroundColor Yellow
    Write-Host "Put them in backend/weights/ directory" -ForegroundColor Yellow
    exit 1
}

# Build and start services
Write-Host ""
Write-Host "🔨 Building Docker images..." -ForegroundColor Yellow
docker-compose build

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🚀 Starting services..." -ForegroundColor Yellow
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start services" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check health
Write-Host ""
Write-Host "🏥 Checking service health..." -ForegroundColor Yellow

$services = @(
    @{Name="Gateway"; Port=8000},
    @{Name="Temporal"; Port=8001},
    @{Name="Liveness"; Port=8002},
    @{Name="Hybrid"; Port=8003},
    @{Name="Spatial"; Port=8004}
)

foreach ($service in $services) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$($service.Port)/health" -Method GET -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ $($service.Name) (port $($service.Port)) - Healthy" -ForegroundColor Green
        } else {
            Write-Host "⚠️  $($service.Name) (port $($service.Port)) - Unhealthy" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ $($service.Name) (port $($service.Port)) - Not responding" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🎉 Deepfake Detection System is running!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""
Write-Host "📍 Endpoints:" -ForegroundColor Cyan
Write-Host "   Gateway:  http://localhost:8000" -ForegroundColor White
Write-Host "   Ensemble: http://localhost:8000/predict/ensemble" -ForegroundColor White
Write-Host ""
Write-Host "📊 View logs:" -ForegroundColor Cyan
Write-Host "   docker-compose logs -f" -ForegroundColor White
Write-Host ""
Write-Host "🛑 Stop services:" -ForegroundColor Cyan
Write-Host "   docker-compose down" -ForegroundColor White
Write-Host ""
