"""
API Gateway for Deepfake Detection System
Ensemble prediction dengan 4 model: Temporal, Liveness, Hybrid, Spatial
Port: 8000
"""

import asyncio
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn

# Configuration
TEMPORAL_URL = os.getenv("TEMPORAL_URL", "http://localhost:8001")
LIVENESS_URL = os.getenv("LIVENESS_URL", "http://localhost:8002")
HYBRID_URL = os.getenv("HYBRID_URL", "http://localhost:8003")
SPATIAL_URL = os.getenv("SPATIAL_URL", "http://localhost:8004")

# Ensemble weights - total must = 1.0
# Hybrid diberi bobot lebih rendah karena model kurang akurat
WEIGHTS = {
    "temporal": 0.30,   # Temporal cukup reliable
    "liveness": 0.35,   # rPPG paling akurat untuk deteksi
    "hybrid": 0.10,     # Hybrid kurang akurat, bobot kecil
    "spatial": 0.25     # Spatial cukup reliable
}

# Reliability scores (0-1) - seberapa bisa dipercaya model ini
MODEL_RELIABILITY = {
    "temporal": 0.75,
    "liveness": 0.85,   # Most reliable
    "hybrid": 0.50,     # Less reliable
    "spatial": 0.70
}

# FastAPI App
app = FastAPI(
    title="Deepfake Detection Gateway",
    description="Smart ensemble detection menggunakan 4 model AI",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Models
class ServiceHealth(BaseModel):
    service: str
    status: str
    available: bool


class HealthResponse(BaseModel):
    status: str
    services: List[ServiceHealth]


class ModelResult(BaseModel):
    prediction: str
    confidence: float
    available: bool
    error: Optional[str] = None
    reason: Optional[str] = None  # ADD REASON FIELD
    frames_processed: Optional[int] = None  # ADD FRAMES FIELD
    # Additional fields for hybrid model
    fake_votes: Optional[int] = None
    real_votes: Optional[int] = None
    avg_score: Optional[float] = None
    median_score: Optional[float] = None


class EnsembleResponse(BaseModel):
    # Final ensemble result
    prediction: str  # "REAL" or "FAKE"
    confidence: float  # Weighted average confidence
    
    # Individual model results
    temporal: ModelResult
    liveness: ModelResult
    hybrid: ModelResult
    spatial: ModelResult
    
    # Metadata
    voting_result: str  # Majority voting result
    models_available: int  # How many models responded
    ensemble_method: str  # "weighted_average"


async def check_service_health(service_name: str, url: str) -> ServiceHealth:
    """Check if a service is healthy"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health")
            if response.status_code == 200:
                return ServiceHealth(
                    service=service_name,
                    status="healthy",
                    available=True
                )
    except Exception as e:
        pass
    
    return ServiceHealth(
        service=service_name,
        status="unhealthy",
        available=False
    )


async def predict_with_service(
    service_name: str, 
    url: str, 
    video_file: bytes, 
    filename: str
) -> ModelResult:
    """
    Send video to a specific service and get prediction
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:  # 2 min timeout
            files = {"video": (filename, video_file, "video/mp4")}
            response = await client.post(f"{url}/predict", files=files)
            
            if response.status_code == 200:
                data = response.json()
                return ModelResult(
                    prediction=data["prediction"],
                    confidence=data["confidence"],
                    available=True,
                    reason=data.get("reason", ""),  # EXTRACT REASON FROM RESPONSE
                    frames_processed=data.get("frames_processed", data.get("frames_analyzed", 0)),
                    # Extract hybrid-specific fields
                    fake_votes=data.get("fake_votes"),
                    real_votes=data.get("real_votes"),
                    avg_score=data.get("avg_score"),
                    median_score=data.get("median_score")
                )
            else:
                # Try to extract error detail from response
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", f"HTTP {response.status_code}")
                except:
                    error_detail = f"HTTP {response.status_code}"
                
                print(f"❌ Service {service_name} error: {error_detail}")
                
                return ModelResult(
                    prediction="UNKNOWN",
                    confidence=0.5,
                    available=False,
                    error=error_detail
                )
    
    except Exception as e:
        return ModelResult(
            prediction="UNKNOWN",
            confidence=0.5,
            available=False,
            error=str(e)
        )


def calculate_ensemble(results: Dict[str, ModelResult]) -> Dict:
    """
    Calculate ensemble prediction using SMART algorithm:
    1. Weighted voting berdasarkan reliability
    2. Konsistensi check antar model
    3. Confidence adjustment berdasarkan agreement
    4. Hybrid model tidak jadi tie-breaker
    """
    available_models = {k: v for k, v in results.items() if v.available}
    
    if not available_models:
        raise HTTPException(
            status_code=503,
            detail="No models available for prediction"
        )
    
    # === PHASE 1: Weighted Voting dengan Reliability ===
    weighted_votes = {"FAKE": 0.0, "REAL": 0.0}
    raw_votes = {"FAKE": 0, "REAL": 0}
    
    for model_name, model_result in available_models.items():
        if model_result.prediction in weighted_votes:
            reliability = MODEL_RELIABILITY.get(model_name, 0.5)
            weighted_votes[model_result.prediction] += reliability
            raw_votes[model_result.prediction] += 1
    
    # === PHASE 2: Exclude hybrid jika tidak konsisten dengan majority ===
    # Hitung majority tanpa hybrid
    non_hybrid_votes = {"FAKE": 0, "REAL": 0}
    for model_name, model_result in available_models.items():
        if model_name != "hybrid" and model_result.prediction in non_hybrid_votes:
            non_hybrid_votes[model_result.prediction] += 1
    
    non_hybrid_majority = "FAKE" if non_hybrid_votes["FAKE"] > non_hybrid_votes["REAL"] else "REAL"
    
    # Jika hybrid berbeda dari majority model lain, kurangi pengaruhnya lebih lanjut
    hybrid_penalty = 1.0
    if "hybrid" in available_models:
        if available_models["hybrid"].prediction != non_hybrid_majority:
            hybrid_penalty = 0.3  # Kurangi pengaruh hybrid 70%
    
    # === PHASE 3: Calculate Weighted Confidence ===
    total_weight = 0
    weighted_sum = 0
    agreement_count = 0
    
    for model_name, model_result in available_models.items():
        base_weight = WEIGHTS.get(model_name, 0)
        reliability = MODEL_RELIABILITY.get(model_name, 0.5)
        
        # Apply hybrid penalty
        if model_name == "hybrid":
            base_weight *= hybrid_penalty
        
        effective_weight = base_weight * reliability
        confidence = model_result.confidence
        
        # Convert confidence: jika prediksi REAL, fake_score = 1 - confidence
        if model_result.prediction == "REAL":
            fake_score = 1 - confidence
        else:
            fake_score = confidence
        
        weighted_sum += fake_score * effective_weight
        total_weight += effective_weight
    
    # Normalize
    final_fake_score = weighted_sum / total_weight if total_weight > 0 else 0.5
    
    # === PHASE 4: Agreement-based Adjustment ===
    # Jika semua model setuju, tingkatkan confidence
    # Jika tidak setuju, turunkan confidence
    total_models = len(available_models)
    max_votes = max(raw_votes.values())
    agreement_ratio = max_votes / total_models if total_models > 0 else 0.5
    
    # Adjust confidence based on agreement
    if agreement_ratio >= 0.75:  # 75%+ models agree
        confidence_boost = 1.1
    elif agreement_ratio >= 0.5:  # 50-75% agree
        confidence_boost = 1.0
    else:  # Less than 50% agree (tie or split)
        confidence_boost = 0.8
    
    # Apply adjustment but keep within bounds
    adjusted_score = final_fake_score * confidence_boost
    adjusted_score = max(0.1, min(0.9, adjusted_score))  # Keep between 0.1 and 0.9
    
    # === PHASE 5: Final Decision ===
    # Gunakan weighted voting result sebagai primary decision
    voting_result = "FAKE" if weighted_votes["FAKE"] > weighted_votes["REAL"] else "REAL"
    
    # Final prediction: prioritas weighted voting, tapi dengan confidence dari calculation
    final_prediction = voting_result
    
    # Final confidence: seberapa yakin dengan prediksi
    if final_prediction == "FAKE":
        final_confidence = adjusted_score
    else:
        final_confidence = 1 - adjusted_score
    
    return {
        "prediction": final_prediction,
        "confidence": float(final_confidence),
        "voting_result": voting_result,
        "models_available": len(available_models),
        "agreement_ratio": agreement_ratio,
        "hybrid_penalty_applied": hybrid_penalty < 1.0
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check health of all services
    """
    services = await asyncio.gather(
        check_service_health("temporal", TEMPORAL_URL),
        check_service_health("liveness", LIVENESS_URL),
        check_service_health("hybrid", HYBRID_URL),
        check_service_health("spatial", SPATIAL_URL),
    )
    
    all_healthy = all(s.available for s in services)
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        services=list(services)
    )


@app.post("/predict/ensemble", response_model=EnsembleResponse)
async def predict_ensemble(video: UploadFile = File(...)):
    """
    Smart Deepfake Detection - Ensemble dari 4 model
    
    Menggabungkan hasil dari:
    - Deep-Temporal (Video + Audio features)
    - Deep-Liveness (rPPG signal analysis)
    - Deep-Hybrid (Xception facial features)
    - Deep-Spatial (Xception spatial artifacts)
    
    Returns:
    - Final prediction (REAL/FAKE) dengan confidence score
    - Individual results dari tiap model
    - Voting result untuk transparansi
    """
    try:
        # Read video file once
        video_bytes = await video.read()
        filename = video.filename or "video.mp4"
        
        # Send to all services in parallel
        results = await asyncio.gather(
            predict_with_service("temporal", TEMPORAL_URL, video_bytes, filename),
            predict_with_service("liveness", LIVENESS_URL, video_bytes, filename),
            predict_with_service("hybrid", HYBRID_URL, video_bytes, filename),
            predict_with_service("spatial", SPATIAL_URL, video_bytes, filename),
        )
        
        # Map results
        model_results = {
            "temporal": results[0],
            "liveness": results[1],
            "hybrid": results[2],
            "spatial": results[3]
        }
        
        # Calculate ensemble
        ensemble = calculate_ensemble(model_results)
        
        return EnsembleResponse(
            prediction=ensemble["prediction"],
            confidence=ensemble["confidence"],
            temporal=model_results["temporal"],
            liveness=model_results["liveness"],
            hybrid=model_results["hybrid"],
            spatial=model_results["spatial"],
            voting_result=ensemble["voting_result"],
            models_available=ensemble["models_available"],
            ensemble_method="weighted_average"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/{model_name}")
async def predict_single_model(model_name: str, video: UploadFile = File(...)):
    """
    Predict menggunakan 1 model saja (backward compatibility)
    """
    url_map = {
        "temporal": TEMPORAL_URL,
        "liveness": LIVENESS_URL,
        "hybrid": HYBRID_URL,
        "spatial": SPATIAL_URL
    }
    
    if model_name not in url_map:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' tidak ditemukan. Available: {list(url_map.keys())}"
        )
    
    video_bytes = await video.read()
    result = await predict_with_service(
        model_name, 
        url_map[model_name], 
        video_bytes, 
        video.filename or "video.mp4"
    )
    
    if not result.available:
        raise HTTPException(
            status_code=503,
            detail=f"Service '{model_name}' unavailable: {result.error}"
        )
    
    return {
        "prediction": result.prediction,
        "confidence": result.confidence,
        "model": model_name
    }


@app.post("/api/detect/video")
async def detect_video_legacy(file: UploadFile = File(...), model: str = "ensemble"):
    """
    Legacy endpoint untuk backward compatibility dengan frontend lama
    Maps ke endpoint microservices yang baru
    """
    if model == "ensemble":
        # Use ensemble prediction
        video_bytes = await file.read()
        filename = file.filename or "video.mp4"
        
        results = await asyncio.gather(
            predict_with_service("temporal", TEMPORAL_URL, video_bytes, filename),
            predict_with_service("liveness", LIVENESS_URL, video_bytes, filename),
            predict_with_service("hybrid", HYBRID_URL, video_bytes, filename),
            predict_with_service("spatial", SPATIAL_URL, video_bytes, filename),
        )
        
        model_results = {
            "temporal": results[0],
            "liveness": results[1],
            "hybrid": results[2],
            "spatial": results[3]
        }
        
        ensemble = calculate_ensemble(model_results)
        
        # Format response sesuai format lama frontend
        return {
            "success": True,
            "is_deepfake": ensemble["prediction"] == "FAKE",
            "confidence": f"{ensemble['confidence'] * 100:.1f}%",
            "label": f"{ensemble['prediction']} (Smart Ensemble)",
            "prediction": ensemble["prediction"],
            "raw_score": ensemble["confidence"],
            "raw_confidence": ensemble["confidence"],
            "threshold": 0.5,
            "frames_analyzed": 20,  # Ensemble typically analyzes 20 frames
            "filename": file.filename,
            "status": "success",
            "message": "Analysis completed using smart ensemble of 4 models",
            "voting_result": ensemble["voting_result"],
            "ensemble_method": "smart_weighted_voting",
            "models_available": sum(1 for m in model_results.values() if m.available),
            "agreement_ratio": ensemble.get("agreement_ratio", 0),
            "hybrid_penalty_applied": ensemble.get("hybrid_penalty_applied", False),
            "reason": f"Smart ensemble: {ensemble['voting_result']} by weighted voting. Agreement: {ensemble.get('agreement_ratio', 0)*100:.0f}%. {sum(1 for m in model_results.values() if m.available)}/4 models used.",
            "model_scores": {
                "temporal": model_results["temporal"].confidence if model_results["temporal"].available else None,
                "liveness": model_results["liveness"].confidence if model_results["liveness"].available else None,
                "hybrid": model_results["hybrid"].confidence if model_results["hybrid"].available else None,
                "spatial": model_results["spatial"].confidence if model_results["spatial"].available else None
            },
            "models": {
                "temporal": {
                    "prediction": model_results["temporal"].prediction,
                    "confidence": model_results["temporal"].confidence,
                    "available": model_results["temporal"].available,
                    "reason": model_results["temporal"].reason,
                    "weight": WEIGHTS.get("temporal", 0),
                    "reliability": MODEL_RELIABILITY.get("temporal", 0)
                },
                "liveness": {
                    "prediction": model_results["liveness"].prediction,
                    "confidence": model_results["liveness"].confidence,
                    "available": model_results["liveness"].available,
                    "reason": model_results["liveness"].reason,
                    "weight": WEIGHTS.get("liveness", 0),
                    "reliability": MODEL_RELIABILITY.get("liveness", 0)
                },
                "hybrid": {
                    "prediction": model_results["hybrid"].prediction,
                    "confidence": model_results["hybrid"].confidence,
                    "available": model_results["hybrid"].available,
                    "reason": model_results["hybrid"].reason,
                    "weight": WEIGHTS.get("hybrid", 0),
                    "reliability": MODEL_RELIABILITY.get("hybrid", 0),
                    "penalty_applied": ensemble.get("hybrid_penalty_applied", False)
                },
                "spatial": {
                    "prediction": model_results["spatial"].prediction,
                    "confidence": model_results["spatial"].confidence,
                    "available": model_results["spatial"].available,
                    "reason": model_results["spatial"].reason,
                    "weight": WEIGHTS.get("spatial", 0),
                    "reliability": MODEL_RELIABILITY.get("spatial", 0)
                }
            }
        }
    else:
        # Single model prediction
        url_map = {
            "temporal": TEMPORAL_URL,
            "liveness": LIVENESS_URL,
            "hybrid": HYBRID_URL,
            "spatial": SPATIAL_URL
        }
        
        if model not in url_map:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model}' tidak ditemukan"
            )
        
        video_bytes = await file.read()
        result = await predict_with_service(
            model, 
            url_map[model], 
            video_bytes, 
            file.filename or "video.mp4"
        )
        
        if not result.available:
            raise HTTPException(
                status_code=503,
                detail=f"Service '{model}' unavailable: {result.error}"
            )
        
        # Format sesuai response format lama - INCLUDE ALL NEEDED FIELDS
        response_data = {
            "success": True,
            "is_deepfake": result.prediction == "FAKE",
            "confidence": f"{result.confidence * 100:.1f}%",
            "label": f"{result.prediction} ({model.title()})",
            "prediction": result.prediction,
            "raw_score": result.confidence,
            "raw_confidence": result.confidence,
            "threshold": 0.5,  # Default threshold
            "frames_analyzed": result.frames_processed or 20,  # GET FROM SERVICE
            "filename": file.filename,
            "status": "success",
            "message": f"Analysis completed using {model} model",
            "reason": result.reason if result.reason else f"Analyzed by {model} model",
            "model": model.title()
        }
        
        # Add hybrid-specific fields if available
        if model == "hybrid":
            if result.fake_votes is not None:
                response_data["fake_votes"] = result.fake_votes
            if result.real_votes is not None:
                response_data["real_votes"] = result.real_votes
            if result.avg_score is not None:
                response_data["avg_score"] = result.avg_score
            if result.median_score is not None:
                response_data["median_score"] = result.median_score
        
        return response_data


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
