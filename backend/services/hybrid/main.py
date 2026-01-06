"""
Hybrid Detection Service
FastAPI microservice untuk Deep-Hybrid model (Xception + InsightFace RetinaFace)
Port: 8003
"""

import os
# Set environment variables BEFORE importing tensorflow/keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from hybrid_detector import HybridDetector

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "weights", "model_modul4_final.keras"))
PORT = int(os.getenv("PORT", 8003))
UPLOAD_DIR = "temp_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global detector instance
detector = None

# FastAPI App
app = FastAPI(
    title="Hybrid Detection Service",
    description="Deep-Hybrid (Xception) deepfake detection microservice",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    reason: str
    frames_analyzed: int
    fake_votes: int
    real_votes: int
    avg_score: float
    median_score: float


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global detector
    try:
        print(f"Loading Hybrid model from: {MODEL_PATH}")
        print(f"Model file exists: {os.path.exists(MODEL_PATH)}")
        print("=" * 80)
        
        detector = HybridDetector(MODEL_PATH)
        
        print("=" * 80)
        print(f"[OK] Hybrid Detection Service ready on port {PORT}")
        print(f"[OK] Detector loaded: {detector is not None}")
        print(f"[OK] Model loaded: {detector.model is not None if detector else False}")
        print(f"[OK] Face cascade loaded: {detector.face_cascade is not None if detector else False}")
        print("=" * 80)
    except Exception as e:
        import traceback
        print("=" * 80)
        print(f"[ERROR] Could not load model")
        print(f"Error: {e}")
        print(traceback.format_exc())
        print("=" * 80)
        detector = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "hybrid",
        "model_loaded": detector is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(video: UploadFile = File(...)):
    """
    Predict if video is REAL or FAKE using Hybrid model (Xception + RetinaFace)
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    temp_path = None
    try:
        # Validate file type
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported: mp4, avi, mov, mkv"
            )
        
        # Save uploaded file
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{video.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Run prediction
        result = detector.detect_video(temp_path)
        
        # Format response
        prediction = "FAKE" if result['is_fake'] else "REAL"
        confidence = result['confidence'] / 100.0  # Convert to 0-1 range
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reason": result['message'],
            "frames_analyzed": result['frames_analyzed'],
            "fake_votes": result['fake_votes'],
            "real_votes": result['real_votes'],
            "avg_score": result['avg_score'],
            "median_score": result['median_score']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ ERROR in /predict endpoint:\n{error_detail}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False
    )
