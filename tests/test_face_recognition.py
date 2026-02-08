import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import cv2
from app.detection.yolo_detector import YOLODetector
from app.recognition.facenet_recognizer import FaceRecognizer
import numpy as np
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pt")

# Init models
detector = YOLODetector(MODEL_PATH)
recognizer = FaceRecognizer()

# Dummy DB embeddings (pour tester)
db = {
    "Mathieu": None  # on remplira plus tard avec embedding réel
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect_faces(frame)

    for (x1, y1, x2, y2) in faces:
        face_crop = frame[y1:y2, x1:x2]
        emb = recognizer.get_embedding(face_crop)

        name = "UNKNOWN"
        # comparaison test avec la "DB"
        for person, db_emb in db.items():
            if db_emb is not None:
                match, dist = recognizer.compare_embeddings(emb, db_emb)
                if match:
                    name = person
                    break

        # draw rectangle + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, name, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Face Recognition Live", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC pour quitter
        break

cap.release()
cv2.destroyAllWindows()
