import cv2
import numpy as np
import os

# Use standalone Keras for Keras 3 model format
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import load_model
import tensorflow as tf

class HybridDetector:
    def __init__(self, model_path):
        """
        Initialize Hybrid Detector dengan Xception
        
        Args:
            model_path: Path ke model Xception (.h5 atau .keras)
        """
        print("[HYBRID] Loading Hybrid Detection Model...")
        print(f"   Model path: {model_path}")
        print(f"   Model exists: {os.path.exists(model_path)}")
        
        # Load model Xception - use keras.models for Keras 3 format
        self.model = load_model(model_path, compile=False)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("[OK] Xception model loaded successfully!")
        
        # Konfigurasi sesuai training
        self.img_size = (299, 299)  # Xception input size
        self.frame_skip = 5  # Ambil lebih banyak frame (1 tiap 5 frame)
        self.max_frames = 30  # Frame yang dianalisis
        self.fake_threshold = 0.5  # Threshold untuk klasifikasi
        
        # Initialize face detector - use Haar Cascade (always available)
        print("[HYBRID] Loading Haar Cascade Face Detector...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("[WARN] Haar Cascade failed to load - will use center crop")
            self.use_face_detection = False
        else:
            print("[OK] Haar Cascade face detector loaded!")
            self.use_face_detection = True
        
        print("[OK] Hybrid Detector Ready!")
    
    def extract_face(self, frame):
        """
        Extract wajah dari frame menggunakan face detector
        Menggunakan multiple scale dan parameter yang lebih toleran
        
        Args:
            frame: Frame video (BGR format dari OpenCV)
            
        Returns:
            Cropped face atau center crop jika tidak ada wajah
        """
        try:
            if self.use_face_detection:
                # Try Haar Cascade detection with multiple parameters
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Try different scale factors for better detection
                for scale_factor in [1.05, 1.1, 1.2]:
                    for min_neighbors in [3, 4, 5]:
                        faces = self.face_cascade.detectMultiScale(
                            gray, 
                            scaleFactor=scale_factor, 
                            minNeighbors=min_neighbors, 
                            minSize=(50, 50),  # Minimum face size
                            maxSize=(frame.shape[1]//2, frame.shape[0]//2)  # Max half frame
                        )
                        
                        if len(faces) > 0:
                            # Ambil wajah terbesar
                            face = max(faces, key=lambda f: f[2] * f[3])
                            x, y, w, h = face
                            
                            # Add more padding for better context (30%)
                            padding = int(max(w, h) * 0.3)
                            x = max(0, x - padding)
                            y = max(0, y - padding)
                            w = min(frame.shape[1] - x, w + 2 * padding)
                            h = min(frame.shape[0] - y, h + 2 * padding)
                            
                            crop = frame[y:y+h, x:x+w]
                            crop = cv2.resize(crop, self.img_size)
                            return crop
            
            # Fallback: Smart center crop assuming face is in upper-center
            h, w = frame.shape[:2]
            
            # Crop upper 2/3 of frame (face usually in upper portion)
            crop_h = int(h * 0.7)
            crop_w = int(w * 0.7)
            start_x = (w - crop_w) // 2
            start_y = int(h * 0.1)  # Start 10% from top
            
            crop = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
            crop = cv2.resize(crop, self.img_size)
            return crop
            
        except Exception as e:
            print(f"[WARN] Error in face extraction: {e}")
            # Ultimate fallback: resize full frame
            return cv2.resize(frame, self.img_size)
    
    def predict_frame(self, face_crop):
        """
        Prediksi single frame
        
        Args:
            face_crop: Cropped face image (sudah di-resize)
            
        Returns:
            Prediction score (0-1, dimana >0.5 = FAKE)
        """
        # Normalisasi (sama dengan training)
        img = face_crop.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Prediksi
        pred = self.model.predict(img, verbose=0)[0][0]
        return float(pred)
    
    def detect_video(self, video_path):
        """
        Deteksi deepfake pada video dengan voting mechanism
        
        Args:
            video_path: Path ke file video
            
        Returns:
            Dictionary berisi hasil deteksi
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        scores = []
        frame_id = 0
        processed_frames = 0
        
        print(f"[VIDEO] Processing video: {os.path.basename(video_path)}")
        
        while cap.isOpened() and processed_frames < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames sesuai konfigurasi
            if frame_id % self.frame_skip == 0:
                # Extract face
                face_crop = self.extract_face(frame)
                
                if face_crop is not None:
                    # Prediksi
                    score = self.predict_frame(face_crop)
                    scores.append(score)
                    processed_frames += 1
            
            frame_id += 1
        
        cap.release()
        
        # Analisis hasil
        if len(scores) == 0:
            return {
                'is_fake': False,
                'confidence': 0.0,
                'message': 'No face detected in video',
                'frames_analyzed': 0,
                'fake_votes': 0,
                'real_votes': 0,
                'avg_score': 0.0,
                'median_score': 0.0
            }
        
        scores = np.array(scores)
        
        # Voting
        fake_votes = np.sum(scores >= self.fake_threshold)
        real_votes = np.sum(scores < self.fake_threshold)
        total = len(scores)
        
        # Statistik
        avg_score = float(np.mean(scores))
        median_score = float(np.median(scores))
        std_score = float(np.std(scores))
        
        # Keputusan final menggunakan MEDIAN (lebih robust terhadap outlier)
        # dan majority voting
        vote_ratio = fake_votes / total
        
        # Gunakan kombinasi: median score DAN voting majority
        # Jika median >= threshold DAN mayoritas vote fake -> FAKE
        # Jika median < threshold DAN mayoritas vote real -> REAL
        # Jika tidak konsisten, gunakan median sebagai tie-breaker
        
        if median_score >= self.fake_threshold and vote_ratio >= 0.5:
            is_fake = True
            confidence = vote_ratio * 100
        elif median_score < self.fake_threshold and vote_ratio < 0.5:
            is_fake = False
            confidence = (1 - vote_ratio) * 100
        else:
            # Tidak konsisten, gunakan median sebagai final decision
            is_fake = median_score >= self.fake_threshold
            # Confidence lebih rendah karena tidak konsisten
            confidence = abs(median_score - 0.5) * 2 * 100 * 0.7  # Maksimal 70% jika tidak konsisten
        
        result = {
            'is_fake': bool(is_fake),
            'confidence': float(min(confidence, 99.9)),  # Cap at 99.9%
            'message': 'FAKE (Deepfake Detected)' if is_fake else 'REAL (Authentic Video)',
            'frames_analyzed': int(total),
            'fake_votes': int(fake_votes),
            'real_votes': int(real_votes),
            'avg_score': float(avg_score),
            'median_score': float(median_score)
        }
        
        print(f"[RESULT] Analysis complete:")
        print(f"   Frames analyzed: {total}")
        print(f"   Fake votes: {fake_votes} ({vote_ratio*100:.1f}%)")
        print(f"   Real votes: {real_votes} ({(1-vote_ratio)*100:.1f}%)")
        print(f"   Average score: {avg_score:.4f}")
        print(f"   Median score: {median_score:.4f}")
        print(f"   Std deviation: {std_score:.4f}")
        print(f"   Result: {result['message']} (confidence: {result['confidence']:.1f}%)")
        
        return result
    
    def detect_image(self, image_path):
        """
        Deteksi deepfake pada single image
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            Dictionary berisi hasil deteksi
        """
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Extract face
        face_crop = self.extract_face(frame)
        
        if face_crop is None:
            return {
                'is_fake': False,
                'confidence': 0.0,
                'message': 'No face detected in image',
                'score': 0.0
            }
        
        # Prediksi
        score = self.predict_frame(face_crop)
        is_fake = score >= self.fake_threshold
        confidence = score * 100 if is_fake else (1 - score) * 100
        
        result = {
            'is_fake': bool(is_fake),
            'confidence': float(confidence),
            'message': 'FAKE (Deepfake Detected)' if is_fake else 'REAL (Authentic Image)',
            'score': float(score)
        }
        
        print(f"[RESULT] Image analysis:")
        print(f"   Score: {score:.4f}")
        print(f"   Result: {result['message']}")
        
        return result
