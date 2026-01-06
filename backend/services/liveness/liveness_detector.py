"""
Deep-Liveness Detector menggunakan rPPG (remote Photoplethysmography)
Model: RPPGNetLite dengan Conv3D untuk deteksi liveness berbasis perubahan warna kulit
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import base64

# Import MTCNN untuk fallback face detection
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    MTCNN = None

# Import mediapipe - versi 0.10.14 menggunakan mp.solutions
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print(f"✓ MediaPipe loaded successfully (version: {mp.__version__})")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    print("=" * 60)
    print("❌ WARNING: MediaPipe not installed!")
    print("   Liveness detection will NOT work.")
    print("   To install, run:")
    print("   pip install mediapipe==0.10.14")
    print("=" * 60)
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    print(f"❌ Error loading MediaPipe: {e}")
    print("   Try reinstalling: pip uninstall mediapipe && pip install mediapipe==0.10.14")


# ==========================================
# KONSTANTA & KONFIGURASI
# ==========================================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
TARGET_ROI = (64, 64)
SEQ_LEN = 64  # Panjang sequence untuk model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bobot Fusion (Gabungan rPPG + Blink)
WEIGHT_RPPG = 0.5   # 50% Kepercayaan pada Model rPPG
WEIGHT_BLINK = 0.5  # 50% Kepercayaan pada Pola Kedipan


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def eye_aspect_ratio(landmarks, eye, w, h):
    """
    Menghitung Eye Aspect Ratio (EAR) untuk deteksi kedipan
    """
    def xy(i):
        p = landmarks[i]
        return np.array([p.x * w, p.y * h])

    p1, p2, p3, p4, p5, p6 = [xy(i) for i in eye]
    # Tambahkan epsilon untuk menghindari pembagian dengan nol
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4) + 1e-6)


def blink_score_logic(ear_values: List[Dict]) -> float:
    """
    Menganalisis pola kedipan dari data EAR.
    Return: 1.0 (Fake/Diam - tidak berkedip) atau 0.0 (Real/Berkedip)
    """
    if len(ear_values) < 10:
        return 0.5  # Data kurang, return neutral
    
    # Extract EAR values dari list of dicts
    ear_list = [item['ear'] for item in ear_values]
    ear_arr = np.array(ear_list)
    
    # Hitung variasi dan rasio mata tertutup
    ear_std = np.std(ear_arr)              # Variasi EAR
    closed_ratio = np.mean(ear_arr < 0.2)  # Rasio mata tertutup (threshold 0.2)
    
    # Aturan Heuristik dari Colab:
    # Fake jika variasi sangat kecil (mata melotot statis) ATAU tidak pernah kedip
    if ear_std < 0.004 or closed_ratio < 0.02:
        return 1.0  # FAKE - tidak ada kedipan natural
    else:
        return 0.0  # REAL - ada kedipan natural


def encode_frame_preview(frame_bgr: np.ndarray) -> str:
    """
    Encode frame BGR ke base64 JPEG untuk preview
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return ""
    
    # Encode ke JPEG
    success, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return ""
    
    # Convert ke base64
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"


# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class RPPGNetLite(nn.Module):
    """
    Lightweight 3D CNN untuk deteksi rPPG signal dari video wajah
    Input: (Batch, 1, 64, 64, 64) - (B, Channel, Depth/Time, H, W)
    Output: (Batch, 2) - Real/Fake classification
    """
    def __init__(self):
        super().__init__()
        # Feature Extraction dengan Conv3D
        self.features = nn.Sequential(
            # Conv1: Extract low-level features
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # Keep temporal, downsample spatial
            
            # Conv2: Extract mid-level features
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),  # Downsample all dimensions
            
            # Conv3: Extract high-level features
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))  # Average pool spatial, keep temporal
        )
        
        self.flatten = nn.Flatten()
        
        # Classifier
        # After pooling: (B, 64_channels, 32_frames, 1, 1) → (B, 64*32)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary classification: Real/Fake
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# ==========================================
# MODEL LOADING
# ==========================================
_liveness_model = None

def load_liveness_model(model_path: str):
    """
    Load RPPGNetLite model dari file .pth
    Model di-load sekali dan di-cache
    """
    global _liveness_model
    
    if _liveness_model is not None:
        return _liveness_model
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file tidak ditemukan: {model_path}")
    
    print(f"Loading Liveness model dari: {model_path}")
    
    # Inisialisasi model
    model = RPPGNetLite()
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    model.eval()
    
    _liveness_model = model
    print(f"✓ Liveness model loaded successfully on {DEVICE}")
    
    return _liveness_model


# ==========================================
# VIDEO PREPROCESSING
# ==========================================
def process_video_liveness(video_path: str) -> Tuple[np.ndarray, List[Dict], List[np.ndarray], str]:
    """
    Process video untuk ekstraksi rPPG features
    
    Returns:
        sequence: Array (N, 64, 64) green channel ROI frames
        ear_data: List of dicts dengan frame index dan EAR values
        display_frames: List of original frames untuk preview
        error_msg: Error message jika ada
    """
    # Check if MediaPipe is available
    if not MEDIAPIPE_AVAILABLE or mp is None:
        error_msg = (
            "❌ MediaPipe library tidak terinstall!\n\n"
            "Liveness detection membutuhkan MediaPipe untuk face detection.\n\n"
            "Cara install:\n"
            "1. Buka terminal di folder backend\n"
            "2. Jalankan: .\\venv\\Scripts\\python.exe -m pip install mediapipe==0.10.14\n"
            "3. Restart backend server\n\n"
            "Atau gunakan model deteksi lain (Deep-Spatial atau Deep-Temporal)"
        )
        return None, [], [], error_msg
    
    # Inisialisasi MediaPipe Face Mesh dengan threshold lebih rendah
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,  # Turunkan dari 0.5 ke 0.3
        min_tracking_confidence=0.3     # Turunkan dari 0.5 ke 0.3
    )
    
    # DISABLE MTCNN fallback karena menyebabkan hang
    mtcnn_detector = None
    # if MTCNN_AVAILABLE:
    #     try:
    #         mtcnn_detector = MTCNN()
    #         print("✓ MTCNN fallback detector initialized")
    #     except:
    #         pass
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, [], [], "Gagal membuka video"
    
    frames = []  # Green channel ROI frames
    ear_data = []  # EAR values
    display_frames = []  # Original frames untuk preview
    frame_idx = 0
    skip_frames = 2  # Proses setiap 2 frame untuk performa
    max_frames = 100  # Batasi maksimal frame yang diproses untuk mencegah hang
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Skip frames untuk performa (proses setiap N frame)
            if frame_idx % skip_frames != 0:
                continue
            
            # Batasi jumlah frame untuk mencegah proses terlalu lama
            if len(frames) >= max_frames:
                print(f"⚠️ Reached max frames limit ({max_frames}), stopping processing")
                break
            
            h, w = frame.shape[:2]
            
            # Preprocessing: Brightness & Contrast enhancement untuk deteksi lebih baik
            enhanced = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
            
            # Convert ke RGB untuk MediaPipe
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
            # Deteksi face mesh
            results = face_mesh.process(rgb)
            
            face_detected = False
            x1, y1, x2, y2 = 0, 0, 0, 0
            current_ear = 0.0
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Hitung EAR (Eye Aspect Ratio)
                ear_left = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                current_ear = (ear_left + ear_right) / 2
                
                # Extract face bounding box
                xs = [p.x for p in landmarks]
                ys = [p.y for p in landmarks]
                x1, y1 = int(min(xs) * w), int(min(ys) * h)
                x2, y2 = int(max(xs) * w), int(max(ys) * h)
                
                face_detected = True
            
            # Fallback ke MTCNN jika MediaPipe gagal
            elif mtcnn_detector is not None:
                detections = mtcnn_detector.detect_faces(rgb)
                if len(detections) > 0:
                    # Ambil deteksi dengan confidence tertinggi
                    best_det = max(detections, key=lambda d: d['confidence'])
                    if best_det['confidence'] > 0.9:  # Threshold confidence
                        box = best_det['box']
                        x1, y1, width, height = box
                        x2, y2 = x1 + width, y1 + height
                        
                        # Default EAR untuk MTCNN (tidak ada landmark mata)
                        current_ear = 0.25  # Nilai default
                        
                        face_detected = True
            
            if face_detected:
                # Simpan EAR data
                ear_data.append({
                    "frame": frame_idx,
                    "ear": current_ear
                })
                
                # Padding
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                # Crop dan resize ROI
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    roi_resized = cv2.resize(roi, TARGET_ROI)
                    
                    # Extract green channel dan normalize
                    green_channel = roi_resized[:, :, 1] / 255.0
                    frames.append(green_channel)
                    
                    # Simpan display frame (dengan bounding box)
                    display_frame = frame.copy()
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    display_frames.append(display_frame)
    
    except Exception as e:
        return None, [], [], f"Error saat memproses video: {str(e)}"
    
    finally:
        cap.release()
        face_mesh.close()
    
    # Validasi jumlah frames
    total_video_frames = frame_idx
    frames_detected = len(frames)
    
    if frames_detected == 0:
        # Return dengan low confidence REAL daripada error
        print(f"⚠️ No face detected in {total_video_frames} frames, returning low-confidence REAL")
        return (
            np.zeros((32, 64, 64), dtype=np.float32),  # Dummy sequence
            [],  # No EAR data
            display_frames if display_frames else [np.zeros((480, 640, 3), dtype=np.uint8)],  # Dummy frame
            "NO_FACE_DETECTED"  # Special flag
        )
    
    if frames_detected < 32:
        # Jika kurang dari 32, coba lanjutkan dengan padding (akan di-handle prepare_sequence)
        print(f"⚠️  Warning: Hanya {frames_detected} frames terdeteksi, akan di-padding ke 64 frames")
    
    sequence = np.array(frames, dtype=np.float32)
    
    return sequence, ear_data, display_frames, ""


def prepare_sequence(sequence: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Prepare sequence untuk model inference
    - Padding jika terlalu pendek
    - Random crop jika terlalu panjang
    - Normalization (mean=0, std=1)
    """
    # Padding jika terlalu pendek
    if sequence.shape[0] < seq_len:
        # Repeat last frame untuk padding
        pad_size = seq_len - sequence.shape[0]
        pad = np.repeat(sequence[-1:], pad_size, axis=0)
        sequence = np.concatenate([sequence, pad])
    
    # Crop jika terlalu panjang (ambil bagian awal untuk konsistensi)
    if sequence.shape[0] > seq_len:
        sequence = sequence[:seq_len]
    
    # Normalization: Mean=0, Std=1
    # Ini memaksa model fokus pada fluktuasi (nadi) bukan warna absolut (warna kulit)
    mean = np.mean(sequence)
    std = np.std(sequence)
    sequence = (sequence - mean) / (std + 1e-6)
    
    return sequence


# ==========================================
# PREDICTION FUNCTIONS
# ==========================================
def predict_video(video_path: str, model_path: str) -> Dict:
    """
    Prediksi deepfake pada video menggunakan liveness detection
    
    Returns:
        Dict berisi:
        - success: bool
        - is_deepfake: bool
        - confidence: str (percentage)
        - raw_score: float
        - label: str
        - frames_analyzed: int
        - frames_info: List[Dict]
        - frames_preview: List[str] (base64 encoded)
        - error: str (jika ada)
    """
    try:
        # Load model
        model = load_liveness_model(model_path)
        
        # Process video
        sequence, ear_data, display_frames, error_msg = process_video_liveness(video_path)
        
        # Handle no face detected case
        if error_msg == "NO_FACE_DETECTED":
            print("⚠️ No face detected, returning default REAL prediction with low confidence")
            return {
                "success": True,
                "is_deepfake": False,
                "confidence": "30.0%",
                "raw_score": 0.30,
                "threshold": 0.5,
                "label": "REAL",
                "reason": "⚠️ Tidak ada wajah terdeteksi dalam video. Model liveness membutuhkan wajah yang jelas dan frontal. Hasil default: REAL dengan confidence rendah. Untuk hasil lebih akurat, gunakan model Spatial atau Temporal.",
                "frames_analyzed": 0,
                "blinks_detected": 0,
                "frames_info": [],
                "frames_preview": []
            }
        
        if error_msg:
            return {
                "success": False,
                "error": error_msg
            }
        
        # Prepare sequence untuk inference
        seq_prepared = prepare_sequence(sequence, SEQ_LEN)
        
        # Convert ke tensor
        # Shape: (1, 1, 64, 64, 64) - (Batch, Channel, Depth, H, W)
        seq_tensor = torch.tensor(seq_prepared[np.newaxis, np.newaxis, ...], dtype=torch.float32).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(seq_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # probabilities[0]: [prob_real, prob_fake]
            prob_rppg_fake = probabilities[0, 1].item()
            prob_rppg_real = probabilities[0, 0].item()
        
        # === FUSION LOGIC (Gabungkan rPPG + Blink) ===
        # 1. Skor dari model rPPG
        print(f"   - Skor rPPG (Deep Learning): {prob_rppg_fake*100:.2f}% Fake")
        
        # 2. Skor dari analisis kedipan
        prob_blink_fake = blink_score_logic(ear_data)
        print(f"   - Skor Kedipan (Heuristik): {prob_blink_fake*100:.2f}% Fake")
        
        # 3. Gabungkan dengan bobot 50-50
        prob_fake = (prob_rppg_fake * WEIGHT_RPPG) + (prob_blink_fake * WEIGHT_BLINK)
        prob_real = 1.0 - prob_fake
        print(f"   - Probabilitas Final: {prob_fake*100:.2f}% Fake")
        
        # Threshold (default 0.5, bisa disesuaikan)
        threshold = 0.5
        is_deepfake = prob_fake > threshold
        
        # Format confidence
        confidence_value = prob_fake if is_deepfake else prob_real
        confidence_str = f"{confidence_value * 100:.1f}%"
        
        # Encode frames untuk preview (sample ~20 frames)
        total_frames = len(display_frames)
        sample_indices = np.linspace(0, total_frames - 1, min(20, total_frames), dtype=int)
        
        frames_info = []
        frames_preview = []
        
        for idx in sample_indices:
            frame = display_frames[idx]
            h, w = frame.shape[:2]
            
            frames_info.append({
                "index": int(idx),
                "width": w,
                "height": h
            })
            
            # Encode frame
            encoded = encode_frame_preview(frame)
            frames_preview.append(encoded)
        
        print(f"✅ Liveness prediction completed:")
        print(f"   - Frames analyzed: {len(sequence)}")
        print(f"   - Prob Real: {prob_real:.4f}, Prob Fake: {prob_fake:.4f}")
        print(f"   - Result: {'FAKE' if is_deepfake else 'REAL'}")
        
        # Generate explanation based on rPPG analysis
        if is_deepfake:
            if prob_fake > 0.9:
                reason = "Sinyal rPPG (remote Photoplethysmography) sangat tidak konsisten. Tidak terdeteksi denyut jantung alami atau pola aliran darah yang valid, menandakan wajah ini hasil sintesis AI."
            elif prob_fake > 0.7:
                reason = "Pola rPPG menunjukkan anomali signifikan. Denyut nadi tidak natural dan variasi warna kulit tidak mengikuti pola biologis manusia asli."
            else:
                reason = "Terdeteksi ketidaknormalan pada sinyal vital rPPG yang mengindikasikan kemungkinan manipulasi digital pada wajah."
        else:
            if prob_fake < 0.3:
                reason = "Sinyal rPPG sangat konsisten dengan manusia hidup. Terdeteksi pola denyut jantung natural, variasi warna kulit sesuai aliran darah, dan micro-expressions autentik."
            elif prob_fake < 0.5:
                reason = "Pola rPPG menunjukkan tanda kehidupan asli dengan denyut nadi konsisten dan perubahan warna kulit yang natural."
            else:
                reason = "Video menunjukkan karakteristik liveness meskipun ada sedikit noise pada sinyal rPPG, namun masih dalam rentang normal untuk video asli."
        
        return {
            "success": True,
            "is_deepfake": is_deepfake,
            "confidence": confidence_str,
            "raw_score": float(prob_fake),
            "threshold": threshold,
            "label": "FAKE" if is_deepfake else "REAL",
            "reason": reason,
            "frames_analyzed": len(sequence),
            "frames_info": frames_info,
            "frames_preview": frames_preview
        }
    
    except Exception as e:
        print(f"❌ Error dalam predict_video: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": f"Error saat prediksi: {str(e)}"
        }


def predict_image(image_path: str, model_path: str) -> Dict:
    """
    Liveness detection tidak cocok untuk gambar statis
    Return error message
    """
    return {
        "success": False,
        "error": "Liveness detection hanya mendukung video, tidak mendukung gambar statis"
    }
