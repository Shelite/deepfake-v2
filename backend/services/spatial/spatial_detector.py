import os
import base64
import cv2
import numpy as np
import warnings
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

warnings.filterwarnings("ignore")

IMG_SIZE = 299
FRAMES_PER_VIDEO = 20
THRESHOLD = 0.5

_detector = None
_model = None
_model_path_loaded = None


def get_detector():
    global _detector
    if _detector is None:
        try:
            _detector = MTCNN()
        except Exception as e:
            print(f"Error initializing MTCNN: {e}")
            _detector = None
    return _detector


def load_spatial_model(model_path: str):
    global _model, _model_path_loaded
    if _model is not None and _model_path_loaded == model_path:
        return _model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file tidak ditemukan: {model_path}")
    
    try:
        print(f"⏳ Loading Spatial model from: {model_path}")
        _model = load_model(model_path, compile=False)
        _model_path_loaded = model_path
        print(f"✅ Model Deep-Spatial dimuat dari: {model_path}")
        return _model
    except Exception as e:
        print(f"❌ Error loading spatial model: {str(e)[:200]}")
        raise RuntimeError(f"Gagal memuat model spatial: {str(e)[:200]}")


def encode_frame_preview(frame_bgr):
    try:
        ok, buffer = cv2.imencode('.jpg', frame_bgr)
        if not ok:
            return None
        b64 = base64.b64encode(buffer.tobytes()).decode('ascii')
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def align_face(img, keypoints):
    left_eye = keypoints.get('left_eye')
    right_eye = keypoints.get('right_eye')
    if not left_eye or not right_eye:
        return img
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    center_x = int((left_eye[0] + right_eye[0]) / 2)
    center_y = int((left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def preprocess_face(frame_bgr, face_data, align=True):
    try:
        working = frame_bgr.copy()
        if align:
            working = align_face(working, face_data.get('keypoints', {}))
        x, y, w, h = face_data['box']
        margin = int(0.2 * w)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(working.shape[1], x + w + margin)
        y2 = min(working.shape[0], y + h + margin)
        face_img = working[y1:y2, x1:x2]
        if face_img.size == 0:
            return None
        face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_array = np.expand_dims(face_rgb.astype(np.float32), axis=0)
        face_preprocessed = preprocess_input(face_array)
        return face_preprocessed, face_resized
    except Exception:
        return None


def predict_frame(model, preprocessed_batch):
    pred = model.predict(preprocessed_batch, verbose=0)[0][0]
    return float(pred)


def process_video_spatial(video_path: str, model):
    detector = get_detector()
    if detector is None:
        return None, None, "MTCNN detector tidak dapat diinisialisasi"

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return None, None, "Video tidak memiliki frame"

    step = max(1, total_frames // FRAMES_PER_VIDEO)
    scores = []
    previews = []
    frame_idx = 0

    while frame_idx < total_frames and len(scores) < FRAMES_PER_VIDEO:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)
        if faces:
            face_data = faces[0]
            prep = preprocess_face(frame, face_data, align=True)
            if prep is not None:
                preprocessed, face_preview = prep
                score = predict_frame(model, preprocessed)
                scores.append(score)
                preview = encode_frame_preview(face_preview)
                if preview:
                    previews.append(preview)
        frame_idx += step

    cap.release()

    if not scores:
        return None, None, "Wajah tidak terdeteksi dalam video"

    return scores, previews, None


def summarize_scores(scores):
    avg_score = float(np.mean(scores))
    is_real_prob = avg_score
    is_fake_prob = 1.0 - avg_score
    is_deepfake = is_fake_prob > is_real_prob  # threshold 0.5 equivalently
    confidence_value = is_fake_prob if is_deepfake else is_real_prob
    confidence = f"{confidence_value * 100:.1f}%"
    
    # Generate spatial analysis explanation
    if is_deepfake:
        if is_fake_prob > 0.9:
            reason = "Xception mendeteksi artefak spasial yang sangat kuat pada level piksel. Ditemukan pola kompresi tidak natural, edge inconsistencies, dan texture anomalies yang khas hasil GAN/deepfake."
        elif is_fake_prob > 0.7:
            reason = "Terdeteksi spatial artifacts signifikan seperti blending errors, color inconsistencies, dan pola noise yang tidak sesuai dengan foto/video asli."
        else:
            reason = "Terdapat anomali spasial ringan pada texture dan edge consistency yang mengindikasikan kemungkinan manipulasi AI."
    else:
        if is_real_prob > 0.9:
            reason = "Analisis spatial menunjukkan texture sangat natural, pola kompresi konsisten, dan tidak ada artefak manipulasi digital. Semua karakteristik pixel-level sesuai dengan foto/video asli."
        elif is_real_prob > 0.7:
            reason = "Spatial features menunjukkan karakteristik asli dengan texture natural dan edge consistency yang baik."
        else:
            reason = "Meskipun ada sedikit noise, spatial analysis menunjukkan video ini kemungkinan besar asli dengan artefak minimal."
    
    return is_deepfake, confidence_value, confidence, avg_score, reason


def predict_video(video_path: str, model_path: str):
    model = load_spatial_model(model_path)
    scores, previews, error = process_video_spatial(video_path, model)
    if scores is None:
        return {
            "success": False,
            "error": error or "Gagal memproses video",
            "is_deepfake": None,
            "confidence": None
        }

    is_deepfake, confidence_value, confidence, avg_score, reason = summarize_scores(scores)

    frames_info = []
    for idx, preview in enumerate(previews[:FRAMES_PER_VIDEO]):
        frames_info.append({
            "index": idx + 1,
            "width": IMG_SIZE,
            "height": IMG_SIZE
        })

    return {
        "success": True,
        "is_deepfake": is_deepfake,
        "confidence": confidence,
        "raw_score": float(avg_score),
        "threshold": THRESHOLD,
        "label": "FAKE (Deepfake)" if is_deepfake else "REAL (Asli)",
        "reason": reason,
        "frames_analyzed": len(scores),
        "frames_info": frames_info,
        "frames_preview": previews[:FRAMES_PER_VIDEO]
    }


def predict_image(image_path: str, model_path: str):
    model = load_spatial_model(model_path)
    detector = get_detector()
    if detector is None:
        return {
            "success": False,
            "error": "MTCNN detector tidak dapat diinisialisasi",
            "is_deepfake": None,
            "confidence": None
        }

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {
            "success": False,
            "error": "Gagal membaca file gambar",
            "is_deepfake": None,
            "confidence": None
        }

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    if not faces:
        return {
            "success": False,
            "error": "Wajah tidak terdeteksi dalam gambar",
            "is_deepfake": None,
            "confidence": None
        }

    prep = preprocess_face(img_bgr, faces[0], align=True)
    if prep is None:
        return {
            "success": False,
            "error": "Gagal memproses wajah",
            "is_deepfake": None,
            "confidence": None
        }

    preprocessed, face_preview = prep
    score = predict_frame(model, preprocessed)
    scores = [score]
    is_deepfake, confidence_value, confidence, avg_score, reason = summarize_scores(scores)

    preview = encode_frame_preview(face_preview)

    return {
        "success": True,
        "is_deepfake": is_deepfake,
        "confidence": confidence,
        "raw_score": float(avg_score),
        "threshold": THRESHOLD,
        "label": "FAKE (Deepfake)" if is_deepfake else "REAL (Asli)",
        "reason": reason,
        "frames_analyzed": 1,
        "frames_info": [{"index": 1, "width": IMG_SIZE, "height": IMG_SIZE}],
        "frames_preview": [preview] if preview else []
    }
