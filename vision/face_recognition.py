import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace
import config

class FaceIdentifier:
    def __init__(self):
        self.known_embeddings = {} # {'Name': [vector], ...}
        self.load_embeddings()

    def load_embeddings(self):
        """
        Loads saved embeddings or generates them from the 'authorized_faces' folder.
        """
        # 1. Try to load existing pickle file
        if os.path.exists(config.EMBEDDINGS_PATH):
            try:
                with open(config.EMBEDDINGS_PATH, "rb") as f:
                    self.known_embeddings = pickle.load(f)
                print(f"Loaded {len(self.known_embeddings)} face embeddings.")
                return
            except Exception as e:
                print(f"Error loading embeddings: {e}. Regenerating...")

        # 2. If no pickle, scan the folder
        print("Generating Face Embeddings... This may take a moment.")
        if not os.path.exists(config.FACE_DIR):
            os.makedirs(config.FACE_DIR)
            
        files = os.listdir(config.FACE_DIR)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(file)[0]
                img_path = os.path.join(config.FACE_DIR, file)
                
                try:
                    # Generate embedding
                    # We use enforcement=True here to ensure the reference image actually has a face
                    embedding_objs = DeepFace.represent(
                        img_path = img_path,
                        model_name = config.FACE_MODEL,
                        enforce_detection = True
                    )
                    embedding = embedding_objs[0]["embedding"]
                    self.known_embeddings[name] = embedding
                    print(f" - Encoded: {name}")
                except Exception as e:
                    print(f" - Could not process {file}: {e}")

        # 3. Save to pickle for next time
        if not os.path.exists(os.path.dirname(config.EMBEDDINGS_PATH)):
            os.makedirs(os.path.dirname(config.EMBEDDINGS_PATH))
            
        with open(config.EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(self.known_embeddings, f)
        print("Embeddings saved.")

    def extract_face_crop(self, frame, keypoints):
        """
        Uses YOLO Keypoints (Nose, Eyes, Ears) to crop the face area.
        Returns the cropped image or None.
        """
        pts = keypoints.xy[0].cpu().numpy()
        conf = keypoints.conf[0].cpu().numpy()
        
        # Indices: 0=Nose, 1=LEye, 2=REye, 3=LEar, 4=REar
        face_indices = [0, 1, 2, 3, 4]
        
        valid_x = []
        valid_y = []
        
        for i in face_indices:
            if conf[i] > 0.5: # Only use confident keypoints
                valid_x.append(pts[i][0])
                valid_y.append(pts[i][1])
                
        if len(valid_x) < 3: # Need at least 3 points to define a face
            return None
            
        # Calculate Bounding Box
        min_x, max_x = min(valid_x), max(valid_x)
        min_y, max_y = min(valid_y), max(valid_y)
        
        # Add Padding (Face is bigger than just eyes+nose)
        width = max_x - min_x
        height = max_y - min_y
        
        pad_x = int(width * 0.5)
        pad_y = int(height * 0.6)
        
        x1 = max(0, int(min_x - pad_x))
        y1 = max(0, int(min_y - pad_y))
        x2 = min(frame.shape[1], int(max_x + pad_x))
        y2 = min(frame.shape[0], int(max_y + pad_y))
        
        if (x2 - x1) < 20 or (y2 - y1) < 20: # Too small
            return None
            
        return frame[y1:y2, x1:x2]

    def identify_person(self, face_crop):
        """
        Generates embedding for the crop and compares with known faces.
        """
        try:
            # We use detector_backend="skip" because we already cropped the face with YOLO.
            # This makes it MUCH faster.
            embedding_objs = DeepFace.represent(
                img_path = face_crop,
                model_name = config.FACE_MODEL,
                enforce_detection = False, 
                detector_backend = "skip" 
            )
            current_emb = embedding_objs[0]["embedding"]
            
            best_match_name = None
            min_dist = 100.0
            
            # Compare with all known faces
            for name, known_emb in self.known_embeddings.items():
                # Cosine Distance Calculation
                a = np.array(current_emb)
                b = np.array(known_emb)
                
                # Manual Cosine Distance (Optimized)
                dot = np.dot(a, b)
                norma = np.linalg.norm(a)
                normb = np.linalg.norm(b)
                dist = 1 - (dot / (norma * normb))
                
                if dist < min_dist:
                    min_dist = dist
                    best_match_name = name
            
            if min_dist < config.FACE_MATCH_THRESHOLD:
                return best_match_name
            
            return None # Unknown
            
        except Exception as e:
            # DeepFace might fail if the crop is empty or weird
            return None