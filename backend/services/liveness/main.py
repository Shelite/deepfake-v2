"""
Liveness Detection Service
FastAPI microservice untuk Deep-Liveness model (rPPG)
Port: 8002
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import uvicorn

from liveness_detector import predict_video, load_liveness_model

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "weights", "rppg_model_90%.pth"))
PORT = int(os.getenv("PORT", 8002))
UPLOAD_DIR = "temp_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# FastAPI App
app = FastAPI(
    title="Liveness Detection Service",
    description="Deep-Liveness (rPPG) deepfake detection microservice",
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
    mediapipe_available: bool


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    reason: str
    frames_processed: int
    blinks_detected: int


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        print(f"Loading model from: {MODEL_PATH}")
        load_liveness_model(MODEL_PATH)
        print(f"✅ Liveness service started on port {PORT}")
    except Exception as e:
        import traceback
        print(f"⚠️  ERROR: Could not load model")
        print(f"Error: {e}")
        print(traceback.format_exc())


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from liveness_detector import _liveness_model, MEDIAPIPE_AVAILABLE
    return {
        "status": "healthy",
        "service": "liveness",
        "model_loaded": _liveness_model is not None,
        "mediapipe_available": MEDIAPIPE_AVAILABLE
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(video: UploadFile = File(...)):
    """
    Predict if video is REAL or FAKE using Liveness model
    """
    temp_path = None
    try:
        # Save uploaded file
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{video.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Run prediction
        result = predict_video(temp_path, MODEL_PATH)
        
        # Check if successful
        if not result.get("success", False):
            raise HTTPException(
                status_code=422,
                detail=result.get("error", "Failed to process video")
            )
        
        # Convert format
        is_deepfake = result.get("is_deepfake", False)
        prediction = "FAKE" if is_deepfake else "REAL"
        
        # Parse confidence "85.3%" -> 0.853
        confidence_str = result.get("confidence", "50.0%")
        confidence = float(confidence_str.rstrip("%")) / 100.0
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "reason": result.get("reason", ""),
            "frames_processed": result.get("frames_analyzed", 0),
            "blinks_detected": result.get("blinks_detected", 0)
        }
    
    except HTTPException:
        raise
    except Exception as e:
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
