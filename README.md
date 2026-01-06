# Deepfake Detection System

Sistem deteksi deepfake berbasis microservices dengan 4 model deteksi: Temporal, Liveness, Hybrid, dan Spatial.

## 📋 Prerequisites

- **Docker** & **Docker Compose** (recommended)
- **Git** & **Git LFS**
- **Python 3.9+** (untuk development tanpa Docker)
- **Node.js 18+** & **pnpm** (untuk frontend)

## 🚀 Quick Start (Docker - Recommended)

### 1. Clone Repository

```bash
git clone https://github.com/Shelite/deepfake-v2.git
cd deepfake-v2
```

### 2. Download Model Weights (Git LFS)

```bash
# Install Git LFS (jika belum)
git lfs install

# Pull model weights
git lfs pull
```

**Verify model files di `backend/weights/`:**
- `best_model_2k_FINAL_V10.h5` (~158 MB)
- `model_modul4_final.keras` (~261 MB)
- `xception_modul1_final2.keras` (~286 MB)
- `rppg_model_90%.pth` (~1.3 MB)

### 3. Setup Environment

```bash
# Copy environment example (opsional)
cp .env.example .env
```

### 4. Build & Run dengan Docker

```bash
# Build semua services
docker-compose build

# Start semua services
docker-compose up -d

# Check status
docker-compose ps
```

**Services akan berjalan di:**
- Frontend: http://localhost:3000
- Gateway API: http://localhost:8000
- Temporal Service: http://localhost:8001
- Liveness Service: http://localhost:8002
- Hybrid Service: http://localhost:8003
- Spatial Service: http://localhost:8004

### 5. Test Health Check

```bash
curl http://localhost:8000/health
```

## 🛠️ Development Setup (Without Docker)

### Backend Services

```bash
cd backend

# Install dependencies untuk semua services
pip install -r requirements.txt

# Atau install per service
cd services/temporal
pip install -r requirements.txt
python main.py

# Ulangi untuk liveness, hybrid, spatial
```

### Frontend

```bash
cd frontend

# Install dependencies
pnpm install

# Run development server
pnpm dev
```

Frontend akan berjalan di http://localhost:3000

## 📁 Project Structure

```
deepfake-v2/
├── backend/
│   ├── gateway/              # API Gateway
│   ├── services/
│   │   ├── temporal/         # Temporal detection
│   │   ├── liveness/         # Liveness detection
│   │   ├── hybrid/           # Hybrid detection
│   │   └── spatial/          # Spatial detection
│   └── weights/              # Model weights (Git LFS)
├── frontend/                 # Next.js frontend
├── docker-compose.yml        # Docker orchestration
└── README.md
```

## 🔧 Configuration

### Environment Variables

Edit `.env` untuk konfigurasi (opsional):

```env
# Backend
TEMPORAL_URL=http://localhost:8001
LIVENESS_URL=http://localhost:8002
HYBRID_URL=http://localhost:8003
SPATIAL_URL=http://localhost:8004

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 📝 Available Scripts (PowerShell)

```powershell
# Development
.\setup-dev.ps1          # Setup development environment
.\run-dev.ps1            # Run dengan Docker Compose
.\run-dev-simple.ps1     # Run sederhana

# Production
.\start.ps1              # Start services
.\stop.ps1               # Stop services
.\stop-dev.ps1           # Stop development services
```

## 🧪 Testing

```bash
# Test individual service
curl -X POST http://localhost:8001/detect \
  -F "file=@test_video.mp4"

# Test via gateway
curl -X POST http://localhost:8000/detect/temporal \
  -F "file=@test_video.mp4"
```

## 📊 Model Details

| Service  | Model File                      | Size    | Purpose           |
|----------|---------------------------------|---------|-------------------|
| Temporal | best_model_2k_FINAL_V10.h5      | 158 MB  | Frame sequence    |
| Liveness | rppg_model_90%.pth              | 1.3 MB  | PPG signal        |
| Hybrid   | model_modul4_final.keras        | 261 MB  | Combined features |
| Spatial  | xception_modul1_final2.keras    | 286 MB  | Single frame      |

## 🐛 Troubleshooting

### Model files tidak ter-download
```bash
git lfs install
git lfs pull
```

### Docker build error
```bash
# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### Port sudah digunakan
Edit `docker-compose.yml` dan ubah port mapping sesuai kebutuhan.

## 📖 Documentation

- [Setup Instructions](SETUP.md) - Panduan lengkap setup
- [API Documentation](backend/gateway/README.md) - API endpoints
- [Model Training](refrensi/) - Jupyter notebooks untuk training

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is for academic purposes.

## 👤 Author

**Shelite**

---

**Note:** Model weights disimpan menggunakan Git LFS. Pastikan Git LFS sudah terinstall sebelum clone repository.
