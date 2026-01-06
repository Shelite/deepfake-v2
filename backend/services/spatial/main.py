"""
Spatial Detection Service
FastAPI microservice untuk Deep-Spatial model (Xception)
Port: 8004
Uses: TensorFlow 2.20 + tf-keras for Keras 3.x compatibility
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from spatial_detector import predict_video, load_spatial_model

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "weights", "xception_modul1_final2.keras"))
PORT = int(os.getenv("PORT", 8004))
UPLOAD_DIR = "temp_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# FastAPI App
app = FastAPI(
    title="Spatial Detection Service",
    description="Deep-Spatial (Xception) deepfake detection microservice - Keras 3.x",
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
    faces_analyzed: int


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        print(f"Loading model from: {MODEL_PATH}")
        load_spatial_model(MODEL_PATH)
        print(f"✅ Spatial service started on port {PORT}")
    except Exception as e:
        import traceback
        print(f"⚠️  ERROR: Could not load model")
        print(f"Error: {e}")
        print(traceback.format_exc())


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from spatial_detector import _model
    return {
        "status": "healthy",
        "service": "spatial",
        "model_loaded": _model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(video: UploadFile = File(...)):
    """
    Predict if video is REAL or FAKE using Spatial model
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
            "faces_analyzed": result.get("faces_analyzed", 0)
        }
    
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
