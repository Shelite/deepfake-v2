# Development Mode - Tanpa Docker
# Run semua services secara manual di terminal terpisah

Write-Host "🚀 Deepfake Detection - Development Mode Setup" -ForegroundColor Cyan
Write-Host ""

Write-Host "📋 Instruksi:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1️⃣  Buat Virtual Environment untuk setiap service (SEKALI SAJA)" -ForegroundColor White
Write-Host "2️⃣  Install dependencies di setiap venv" -ForegroundColor White
Write-Host "3️⃣  Jalankan setiap service di terminal terpisah" -ForegroundColor White
Write-Host ""

$continue = Read-Host "Lanjutkan setup? (y/n)"
if ($continue -ne "y") {
    exit
}

# Setup Temporal Service
Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🔧 Setting up Temporal Service..." -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan

if (-not (Test-Path "services\temporal\venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Gray
    python -m venv services\temporal\venv
    
    Write-Host "Installing dependencies..." -ForegroundColor Gray
    & services\temporal\venv\Scripts\python.exe -m pip install --upgrade pip
    & services\temporal\venv\Scripts\pip.exe install -r services\temporal\requirements.txt
    
    Write-Host "✅ Temporal service setup complete" -ForegroundColor Green
} else {
    Write-Host "⏭️  Temporal venv already exists" -ForegroundColor Gray
}

# Setup Liveness Service
Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🔧 Setting up Liveness Service..." -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan

if (-not (Test-Path "services\liveness\venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Gray
    python -m venv services\liveness\venv
    
    Write-Host "Installing dependencies..." -ForegroundColor Gray
    & services\liveness\venv\Scripts\python.exe -m pip install --upgrade pip
    & services\liveness\venv\Scripts\pip.exe install -r services\liveness\requirements.txt
    
    Write-Host "✅ Liveness service setup complete" -ForegroundColor Green
} else {
    Write-Host "⏭️  Liveness venv already exists" -ForegroundColor Gray
}

# Setup Hybrid Service
Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🔧 Setting up Hybrid Service..." -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan

if (-not (Test-Path "services\hybrid\venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Gray
    python -m venv services\hybrid\venv
    
    Write-Host "Installing dependencies..." -ForegroundColor Gray
    & services\hybrid\venv\Scripts\python.exe -m pip install --upgrade pip
    & services\hybrid\venv\Scripts\pip.exe install -r services\hybrid\requirements.txt
    
    Write-Host "✅ Hybrid service setup complete" -ForegroundColor Green
} else {
    Write-Host "⏭️  Hybrid venv already exists" -ForegroundColor Gray
}

# Setup Spatial Service
Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🔧 Setting up Spatial Service..." -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan

if (-not (Test-Path "services\spatial\venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Gray
    python -m venv services\spatial\venv
    
    Write-Host "Installing dependencies..." -ForegroundColor Gray
    & services\spatial\venv\Scripts\python.exe -m pip install --upgrade pip
    & services\spatial\venv\Scripts\pip.exe install -r services\spatial\requirements.txt
    
    Write-Host "✅ Spatial service setup complete" -ForegroundColor Green
} else {
    Write-Host "⏭️  Spatial venv already exists" -ForegroundColor Gray
}

# Setup Gateway
Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🔧 Setting up Gateway..." -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan

if (-not (Test-Path "gateway\venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Gray
    python -m venv gateway\venv
    
    Write-Host "Installing dependencies..." -ForegroundColor Gray
    & gateway\venv\Scripts\python.exe -m pip install --upgrade pip
    & gateway\venv\Scripts\pip.exe install -r gateway\requirements.txt
    
    Write-Host "✅ Gateway setup complete" -ForegroundColor Green
} else {
    Write-Host "⏭️  Gateway venv already exists" -ForegroundColor Gray
}

# Create symlinks for weights
Write-Host ""
Write-Host "🔗 Creating symlinks for model weights..." -ForegroundColor Yellow

$services = @("temporal", "liveness", "hybrid", "spatial")
foreach ($service in $services) {
    $target = "services\$service\weights"
    if (-not (Test-Path $target)) {
        New-Item -ItemType SymbolicLink -Path $target -Target "..\..\backend\weights" -ErrorAction SilentlyContinue | Out-Null
        if (Test-Path $target) {
            Write-Host "✅ Created symlink: $target" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Failed to create symlink for $service. Copy weights manually." -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green
Write-Host ""
Write-Host "📝 Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Buka 5 terminal PowerShell terpisah" -ForegroundColor White
Write-Host ""
Write-Host "2. Di setiap terminal, jalankan:" -ForegroundColor White
Write-Host ""
Write-Host "   Terminal 1 - Temporal Service:" -ForegroundColor Yellow
Write-Host "   cd e:\deepfake-v2\services\temporal" -ForegroundColor Gray
Write-Host "   .\venv\Scripts\python.exe main.py" -ForegroundColor Gray
Write-Host ""
Write-Host "   Terminal 2 - Liveness Service:" -ForegroundColor Yellow
Write-Host "   cd e:\deepfake-v2\services\liveness" -ForegroundColor Gray
Write-Host "   .\venv\Scripts\python.exe main.py" -ForegroundColor Gray
Write-Host ""
Write-Host "   Terminal 3 - Hybrid Service:" -ForegroundColor Yellow
Write-Host "   cd e:\deepfake-v2\services\hybrid" -ForegroundColor Gray
Write-Host "   .\venv\Scripts\python.exe main.py" -ForegroundColor Gray
Write-Host ""
Write-Host "   Terminal 4 - Spatial Service:" -ForegroundColor Yellow
Write-Host "   cd e:\deepfake-v2\services\spatial" -ForegroundColor Gray
Write-Host "   .\venv\Scripts\python.exe main.py" -ForegroundColor Gray
Write-Host ""
Write-Host "   Terminal 5 - Gateway:" -ForegroundColor Yellow
Write-Host "   cd e:\deepfake-v2\gateway" -ForegroundColor Gray
Write-Host "   .\venv\Scripts\python.exe main.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Test API:" -ForegroundColor White
Write-Host "   curl http://localhost:8000/health" -ForegroundColor Gray
Write-Host ""
Write-Host "💡 Atau gunakan: .\run-dev.ps1 (otomatis start semua)" -ForegroundColor Cyan
Write-Host ""
