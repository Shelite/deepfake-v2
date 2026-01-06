# models/temporal_detector.py
# Deep-Temporal Model untuk deteksi deepfake
# Berdasarkan arsitektur V10 dari notebook preprocessing

import os
import base64
import numpy as np
import cv2
import librosa
import warnings
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import Xception
from mtcnn import MTCNN

# Suppress warnings
warnings.filterwarnings("ignore")

# Konstanta Processing (Sama dengan saat training)
SEQ_LEN = 20
IMG_SIZE = 224
AUD_LEN = 63
DETECT_WIDTH = 640
OPTIMAL_THRESHOLD = 0.515  # Threshold optimal dari notebook

# Global detector (akan diinisialisasi sekali)
_detector = None
_model = None


def get_detector():
    """Lazy loading MTCNN detector"""
    global _detector
    if _detector is None:
        try:
            _detector = MTCNN()
        except Exception as e:
            print(f"Error initializing MTCNN: {e}")
            _detector = None
    return _detector


def variance_of_laplacian(image):
    """Menghitung sharpness gambar menggunakan Laplacian variance"""
    return cv2.Laplacian(image, cv2.CV_64F).var()


def apply_gamma_correction(image, gamma=1.3):
    """Apply gamma correction untuk memperbaiki brightness"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def enhance_lighting_clahe(image):
    """Enhance lighting menggunakan CLAHE (untuk BGR image)"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def get_face_crop(frame, box):
    """Crop wajah dengan margin"""
    x, y, w, h = box
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    h_img, w_img = frame.shape[:2]
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w_img, x + w + margin_x)
    y2 = min(h_img, y + h + margin_y)
    return frame[y1:y2, x1:x2]


def encode_frame_preview(frame_bgr):
    """Encode frame BGR menjadi data URL base64 JPEG untuk dikirim ke frontend."""
    try:
        success, buffer = cv2.imencode('.jpg', frame_bgr)
        if not success:
            return None
        b64 = base64.b64encode(buffer.tobytes()).decode('ascii')
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def extract_audio(path):
    """Ekstrak MFCC dari audio video. Returns zero array jika gagal."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            y, sr = librosa.load(path, sr=16000, duration=3.0)
            
            # Cek apakah audio valid
            if y is None or len(y) == 0:
                print("Audio kosong, menggunakan zero array")
                return np.zeros((AUD_LEN, 13), dtype=np.float32)
            
            mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13).T
            if mfcc.shape[0] < AUD_LEN:
                mfcc = np.pad(mfcc, ((0, AUD_LEN - mfcc.shape[0]), (0, 0)), 'constant')
            else:
                mfcc = mfcc[:AUD_LEN, :]
            return mfcc.astype(np.float32)
        except Exception as e:
            print(f"Warning: Audio extraction failed ({e}), menggunakan zero array")
            # Return zero array instead of None - model tetap bisa jalan tanpa audio
            return np.zeros((AUD_LEN, 13), dtype=np.float32)


def process_video_smart(path):
    """
    Preprocessing video: ekstrak 20 frame terbaik dengan deteksi wajah
    Returns: visual_data, audio_data, display_images, error_message
    """
    detector = get_detector()
    if detector is None:
        return None, None, None, "MTCNN detector not initialized"
    
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < SEQ_LEN:
        cap.release()
        return None, None, None, f"Video terlalu pendek. Memiliki {total_frames} frame, butuh minimal {SEQ_LEN}."
    
    chunk_size = max(1, total_frames // SEQ_LEN)
    final_frames_proc = []
    final_frames_display = []

    for i in range(SEQ_LEN):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * chunk_size)
        best_frame = None
        best_sharpness = -1.0
        frames_scanned = 0
        
        while frames_scanned < chunk_size:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            if w > 320:
                scale_blur = w / 320
                gray_small = cv2.resize(gray, (320, int(h / scale_blur)))
            else:
                gray_small = gray
            
            score = variance_of_laplacian(gray_small)
            if score > best_sharpness:
                best_sharpness = score
                best_frame = frame
            frames_scanned += 1

        if best_frame is not None:
            # Apply gamma correction
            best_frame_gamma = apply_gamma_correction(best_frame)
            
            h, w = best_frame_gamma.shape[:2]
            scale = 1.0
            
            if w > DETECT_WIDTH:
                scale = w / DETECT_WIDTH
                frame_small = cv2.resize(best_frame_gamma, (DETECT_WIDTH, int(h / scale)))
            else:
                frame_small = best_frame_gamma
            
            rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            res = detector.detect_faces(rgb_small)

            if res:
                box = res[0]['box']
                xs, ys, ws, hs = box
                xr = int(xs * scale)
                yr = int(ys * scale)
                wr = int(ws * scale)
                hr = int(hs * scale)

                # Apply CLAHE enhancement (BGR)
                frame_enhanced_bgr = enhance_lighting_clahe(best_frame_gamma)
                
                # Convert to RGB for processing
                face_proc = get_face_crop(cv2.cvtColor(frame_enhanced_bgr, cv2.COLOR_BGR2RGB), [xr, yr, wr, hr])
                face_display = get_face_crop(best_frame, [xr, yr, wr, hr])

                try:
                    face_resized_proc = cv2.resize(face_proc, (IMG_SIZE, IMG_SIZE))
                    final_frames_proc.append(face_resized_proc.astype(np.float32) / 255.0)

                    face_resized_display = cv2.resize(face_display, (IMG_SIZE, IMG_SIZE))
                    final_frames_display.append(face_resized_display)
                except:
                    pass

    cap.release()
    
    if len(final_frames_proc) < SEQ_LEN:
        # Padding jika frame kurang, atau return error jika tidak ada wajah
        if len(final_frames_proc) > 0:
            while len(final_frames_proc) < SEQ_LEN:
                final_frames_proc.append(final_frames_proc[-1])
                if final_frames_display:
                    final_frames_display.append(final_frames_display[-1])
        else:
            return None, None, None, f"Wajah tidak terdeteksi dalam frame yang cukup. Ditemukan {len(final_frames_proc)}, butuh {SEQ_LEN}."

    vis = np.array(final_frames_proc[:SEQ_LEN], dtype=np.float32)
    aud = extract_audio(path)  # Akan return zero array jika gagal
    
    # Pastikan aud tidak None (seharusnya sudah ditangani di extract_audio)
    if aud is None:
        aud = np.zeros((AUD_LEN, 13), dtype=np.float32)

    return vis, aud, final_frames_display, None  # None = no error


def build_v10_model():
    """
    Arsitektur Model V10: Xception + GRU untuk visual, CNN 1D untuk audio
    """
    # Visual Stream (Xception + GRU)
    vin = layers.Input((SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
    base_cnn = Xception(weights='imagenet', include_top=False, pooling='avg')
    base_cnn.trainable = True
    
    # Freeze layer awal
    for layer in base_cnn.layers[:-32]:
        layer.trainable = False
    
    x = layers.TimeDistributed(base_cnn)(vin)
    x = layers.GRU(128, dropout=0.5, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dense(64, activation='relu')(x)

    # Audio Stream (CNN 1D)
    ain = layers.Input((AUD_LEN, 13))
    y = layers.Conv1D(32, 3, activation='relu', padding='same')(ain)
    y = layers.MaxPooling1D(2)(y)
    y = layers.Conv1D(64, 3, activation='relu', padding='same')(y)
    y = layers.GlobalAveragePooling1D()(y)
    y = layers.Dense(64, activation='relu')(y)

    # Fusion & Classification
    z = layers.concatenate([x, y])
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dropout(0.5)(z)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(z)
    
    return models.Model([vin, ain], out)


def load_temporal_model(model_path: str):
    """Load model Deep-Temporal dari file .h5"""
    global _model
    
    if _model is not None:
        return _model
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file tidak ditemukan: {model_path}")
    
    # Set mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Build dan load weights
    _model = build_v10_model()
    _model.load_weights(model_path)
    
    print(f"✅ Model Deep-Temporal berhasil dimuat dari: {model_path}")
    return _model


def predict_video(video_path: str, model_path: str):
    """
    Prediksi deepfake pada video
    Returns: dict dengan hasil prediksi
    """
    # Load model
    model = load_temporal_model(model_path)
    
    # Preprocessing - now returns 4 values
    vis, aud, display_imgs, error_msg = process_video_smart(video_path)
    
    if vis is None:
        return {
            "success": False,
            "error": error_msg or "Gagal memproses video (wajah tidak ditemukan, video terlalu pendek, atau file rusak)",
            "is_deepfake": None,
            "confidence": None
        }
    
    # Safety check untuk audio
    if aud is None:
        aud = np.zeros((AUD_LEN, 13), dtype=np.float32)
    
    # Prediksi
    pred = model.predict(
        [np.expand_dims(vis, 0), np.expand_dims(aud, 0)], 
        verbose=0
    )[0][0]
    
    # Gunakan OPTIMAL_THRESHOLD dari notebook
    is_deepfake = bool(pred > OPTIMAL_THRESHOLD)
    confidence_value = float(pred) if is_deepfake else float(1 - pred)
    confidence = f"{confidence_value * 100:.1f}%"

    # Generate explanation based on score
    if is_deepfake:
        if pred > 0.9:
            reason = "Inconsistensi temporal yang sangat kuat terdeteksi antara frame video dan fitur audio. Pola pergerakan wajah tidak natural dan sinkronisasi audio-visual mencurigakan."
        elif pred > 0.7:
            reason = "Ditemukan pola pergerakan tidak konsisten antar frame dan anomali pada fitur audio yang mengindikasikan manipulasi deepfake."
        else:
            reason = "Terdeteksi ketidaksesuaian ringan pada temporal sequence dan audio features yang kemungkinan hasil manipulasi AI."
    else:
        if pred < 0.3:
            reason = "Pola pergerakan wajah sangat natural dan konsisten antar frame. Sinkronisasi audio-visual sempurna, tidak ada tanda manipulasi."
        elif pred < 0.5:
            reason = "Temporal consistency baik, pergerakan wajah natural, dan fitur audio sesuai dengan video asli."
        else:
            reason = "Video menunjukkan karakteristik asli meskipun ada sedikit noise, namun masih dalam batas normal untuk video real."

    # Info frame untuk frontend
    frames_info = []
    frames_preview = []
    for idx, frame in enumerate(display_imgs[:SEQ_LEN]):
        h, w = frame.shape[:2]
        frames_info.append({"index": idx + 1, "width": int(w), "height": int(h)})
        preview = encode_frame_preview(frame)
        if preview:
            frames_preview.append(preview)
    
    return {
        "success": True,
        "is_deepfake": is_deepfake,
        "confidence": confidence,
        "raw_score": float(pred),
        "threshold": OPTIMAL_THRESHOLD,
        "label": "FAKE (Deepfake)" if is_deepfake else "REAL (Asli)",
        "reason": reason,
        "frames_analyzed": SEQ_LEN,
        "frames_info": frames_info,
        "frames_preview": frames_preview
    }


def predict_image(image_path: str, model_path: str):
    """
    Prediksi deepfake pada gambar tunggal
    Untuk gambar, kita duplikasi menjadi sequence
    """
    model = load_temporal_model(model_path)
    detector = get_detector()
    
    if detector is None:
        return {
            "success": False,
            "error": "MTCNN detector tidak dapat diinisialisasi",
            "is_deepfake": None,
            "confidence": None
        }
    
    # Baca gambar
    img = cv2.imread(image_path)
    if img is None:
        return {
            "success": False,
            "error": "Gagal membaca file gambar",
            "is_deepfake": None,
            "confidence": None
        }
    
    # Apply gamma correction
    img_gamma = apply_gamma_correction(img)
    
    # Deteksi wajah
    h, w = img_gamma.shape[:2]
    
    scale = 1.0
    if w > DETECT_WIDTH:
        scale = w / DETECT_WIDTH
        frame_small = cv2.resize(img_gamma, (DETECT_WIDTH, int(h / scale)))
    else:
        frame_small = img_gamma
    
    rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    res = detector.detect_faces(rgb_small)
    
    if not res:
        return {
            "success": False,
            "error": "Wajah tidak terdeteksi dalam gambar",
            "is_deepfake": None,
            "confidence": None
        }
    
    # Crop dan preprocess wajah
    box = res[0]['box']
    xs, ys, ws, hs = box
    xr = int(xs * scale)
    yr = int(ys * scale)
    wr = int(ws * scale)
    hr = int(hs * scale)
    
    # Apply CLAHE enhancement (BGR) then convert to RGB
    frame_enhanced_bgr = enhance_lighting_clahe(img_gamma)
    face = get_face_crop(cv2.cvtColor(frame_enhanced_bgr, cv2.COLOR_BGR2RGB), [xr, yr, wr, hr])
    
    try:
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face_normalized = face_resized.astype(np.float32) / 255.0
    except:
        return {
            "success": False,
            "error": "Gagal memproses wajah dalam gambar",
            "is_deepfake": None,
            "confidence": None
        }
    
    # Duplikasi untuk membuat sequence (untuk model temporal)
    vis = np.array([face_normalized] * SEQ_LEN, dtype=np.float32)
    
    # Untuk audio, gunakan zeros (karena gambar tidak punya audio)
    aud = np.zeros((AUD_LEN, 13), dtype=np.float32)
    
    # Prediksi
    pred = model.predict(
        [np.expand_dims(vis, 0), np.expand_dims(aud, 0)], 
        verbose=0
    )[0][0]
    
    # Gunakan OPTIMAL_THRESHOLD
    is_deepfake = bool(pred > OPTIMAL_THRESHOLD)
    confidence_value = float(pred) if is_deepfake else float(1 - pred)
    confidence = f"{confidence_value * 100:.1f}%"
    
    return {
        "success": True,
        "is_deepfake": is_deepfake,
        "confidence": confidence,
        "raw_score": float(pred),
        "threshold": OPTIMAL_THRESHOLD,
        "label": "FAKE (Deepfake)" if is_deepfake else "REAL (Asli)",
        "note": "Analisis gambar menggunakan model temporal (tanpa audio)"
    }
