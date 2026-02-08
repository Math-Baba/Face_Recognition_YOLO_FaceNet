import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from detection.yolo_detector import YOLODetector
from recognition.facenet_recognizer import FaceRecognizer
from database.postgres import Database
from unknown_faces import UnknownFaceManager

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.pt"

# Config
MATCH_THRESHOLD = 0.4           # Seuil de reconnaissance (visage connu)
UNKNOWN_DUP_THRESHOLD = 0.25    # Seuil pour éviter les doublons d’inconnus
SAMPLES_TO_SAVE = 3             # Nombre d’échantillons nécessaires avant d’enregistrer un inconnu


def main():
    print("[INFO] Initialisation...")

    # Initialisation des composants principaux
    detector = YOLODetector(str(MODEL_PATH))    # Détection des visages
    recognizer = FaceRecognizer()               # Extraction & comparaison des embeddings
    db = Database()                             # Connexion PostgreSQL
    unknown_manager = UnknownFaceManager()      # Gestion des visages inconnus

    # Chargement des personnes connues depuis la base de données
    persons_db = db.get_all_persons()
    print(f"[INFO] {len(persons_db)} personnes chargées depuis PostgreSQL")

    # Initialisation de la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam introuvable")
        return

    # Buffer temporaire pour accumuler les embeddings des inconnus
    unknown_buffer = {}  

    # Identifiant interne pour suivre les inconnus
    track_id = 0

    print("[INFO] Démarrage reconnaissance faciale (ESC pour quitter)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Détection des visages avec YOLO
        faces = detector.detect_faces(frame, conf=0.6)

        for (x1, y1, x2, y2) in faces:

            # Découpage du visage
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            
            # Extraction de l’embedding facial
            embedding = recognizer.get_embedding(face_crop)
            if embedding is None:
                continue

            # Recherche de la meilleure correspondance en base
            name, distance = recognizer.find_best_match(
                embedding, persons_db, threshold=MATCH_THRESHOLD
            )

            # ---------------- VISAGE CONNU ----------------
            if name:
                label = f"{name} ({distance:.2f})"
                color = (0, 255, 0) # vert

            # ---------------- VISAGE INCONNU ----------------
            else:
                label = "UNKNOWN"
                color = (0, 165, 255) # orange

                # Initialisation du buffer pour cet inconnu
                if track_id not in unknown_buffer:
                    unknown_buffer[track_id] = []

                # Ajout de l’embedding courant
                unknown_buffer[track_id].append(embedding)

                # Une fois assez d’échantillons collectés
                if len(unknown_buffer[track_id]) >= SAMPLES_TO_SAVE:
                    # Moyenne des derniers embeddings
                    avg_emb = np.mean(
                        np.stack(unknown_buffer[track_id][-SAMPLES_TO_SAVE:]),
                        axis=0
                    )

                    # Vérification des doublons d’inconnus
                    is_duplicate = False
                    for _, u_emb in unknown_manager.get_all_unknown_embeddings().items():
                        _, d = recognizer.compare_embeddings(avg_emb, u_emb, UNKNOWN_DUP_THRESHOLD)
                        if d < UNKNOWN_DUP_THRESHOLD:
                            is_duplicate = True
                            break

                    # Enregistrement si c’est un nouvel inconnu
                    if not is_duplicate:
                        unknown_manager.add_unknown_face(avg_emb, face_crop)

                    # Nettoyage du buffer
                    unknown_buffer.pop(track_id, None)
                    track_id += 1

            # Affichage du rectangle et du label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )

        cv2.imshow("Face Recognition", frame)

        # Quitter avec ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()
    db.close()
    print("[INFO] Arrêt du système")


if __name__ == "__main__":
    main()
