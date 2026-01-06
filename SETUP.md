# Setup Instructions for Microservices Architecture

## ✅ Checklist Setup

### 1. Copy Model Weights
```bash
# Pastikan semua model weights ada di backend/weights/
cd e:\deepfake-v2
ls backend\weights\

# Expected files:
# - model_v10_optimized.h5 (Temporal)
# - rppg_model_lite.pth (Liveness)
# - model_modul4_generalized.h5 (Hybrid)
# - xception_modul1_final2.keras (Spatial)
```

### 2. Build Docker Images
```bash
cd e:\deepfake-v2

# Build semua services
docker-compose build

# Atau build satu-satu:
docker-compose build temporal
docker-compose build liveness
docker-compose build hybrid
docker-compose build spatial
docker-compose build gateway
```

### 3. Start Services
```bash
# Start semua services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Test Health
```bash
# Test gateway
curl http://localhost:8000/health

# Test individual services
curl http://localhost:8001/health  # Temporal
curl http://localhost:8002/health  # Liveness
curl http://localhost:8003/health  # Hybrid
curl http://localhost:8004/health  # Spatial
```

### 5. Test Prediction
```bash
# Test ensemble (recommended)
curl -X POST http://localhost:8000/predict/ensemble \
  -F "video=@test_video.mp4"

# Test single model
curl -X POST http://localhost:8000/predict/temporal \
  -F "video=@test_video.mp4"
```

## 🔧 Troubleshooting

### Service tidak start
```bash
# Check logs
docker-compose logs temporal

# Rebuild
docker-compose build --no-cache temporal
docker-compose up -d temporal
```

### Model tidak load
```bash
# Check volume mount
docker-compose exec temporal ls -la /app/weights/

# Jika kosong, pastikan backend/weights/ ada file model
```

### Port conflict
```bash
# Stop existing processes
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process -Force

# Atau ubah port di docker-compose.yml
```

## 📝 Next Steps

1. Update frontend API endpoint:
   - Ubah `/api/detect/video?model=temporal` 
   - Menjadi `/predict/ensemble` atau `/predict/{model_name}`

2. Deploy ke VPS:
   - Install Docker di VPS Linux
   - Clone project
   - Copy weights
   - Run `docker-compose up -d`
   - Setup Nginx reverse proxy
   - Setup SSL dengan Let's Encrypt

## 🎯 Ensemble API Response Format

```json
{
  "prediction": "FAKE",
  "confidence": 0.87,
  "temporal": {
    "prediction": "FAKE",
    "confidence": 0.85,
    "available": true
  },
  "liveness": {
    "prediction": "FAKE",
    "confidence": 0.92,
    "available": true
  },
  "hybrid": {
    "prediction": "FAKE",
    "confidence": 0.78,
    "available": true
  },
  "spatial": {
    "prediction": "FAKE",
    "confidence": 0.88,
    "available": true
  },
  "voting_result": "FAKE",
  "models_available": 4,
  "ensemble_method": "weighted_average"
}
```

## Frontend Integration

Update fetch call di `frontend/app/upload/[model]/page.tsx`:

```typescript
// Old (single model)
const endpoint = `/api/detect/video?model=${modelId}`

// New (ensemble)
const endpoint = modelId === 'ensemble' 
  ? `/predict/ensemble`
  : `/predict/${modelId}`

const fullUrl = `${API_URL}${endpoint}`
```

Parse response untuk ensemble:

```typescript
if (modelId === 'ensemble') {
  setResult({
    prediction: data.prediction,
    confidence: data.confidence,
    models: {
      temporal: data.temporal,
      liveness: data.liveness,
      hybrid: data.hybrid,
      spatial: data.spatial
    },
    voting: data.voting_result,
    ensemble_method: data.ensemble_method
  })
}
```
