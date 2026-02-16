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
MATCH_THRESHOLD = 0.4            # Seuil de reconnaissance (visage connu)
CONFIDENCE_MARGIN = 0             # Desactive : 0.02 rejetait trop. Re-activer si confusions
DEBUG_DISTANCES = False          # Mettre True pour diagnostiquer
UNKNOWN_DUP_THRESHOLD = 0.25    # Seuil pour eviter les doublons d inconnus
SAMPLES_TO_SAVE = 5
TRACK_MAX_DIST = 80
RELOAD_DB_EVERY_N_FRAMES = 90   # Recharger persons_db periodiquement

def main():
    print("[INFO] Initialisation...")

    # Initialisation des composants principaux
    detector = YOLODetector(str(MODEL_PATH))
    recognizer = FaceRecognizer()
    db = Database()
    unknown_manager = UnknownFaceManager()

    # Chargement des personnes connues depuis la base de donnees
    persons_db = db.get_all_persons()
    print(f"[INFO] {len(persons_db)} personnes chargees depuis PostgreSQL")

    # Initialisation de la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam introuvable")
        return

    # Pistes de visages inconnus : suivi spatial par position
    face_tracks = {}
    next_track_id = 0
    frame_count = 0

    print("[INFO] Demarrage reconnaissance faciale (ESC pour quitter)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Rechargement periodique des personnes (inclut les nouvelles inscriptions)
        if frame_count % RELOAD_DB_EVERY_N_FRAMES == 0:
            persons_db = db.get_all_persons()

        # Detection des visages avec YOLO
        faces = detector.detect_faces(frame, conf=0.6)
        used_tracks_this_frame = set()  # Evite que 2 visages de la meme frame partagent une piste

        for (x1, y1, x2, y2) in faces:

            # Decoupage du visage
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Extraction de l embedding facial
            embedding = recognizer.get_embedding(face_crop)
            if embedding is None:
                continue

            # Recherche de la meilleure correspondance en base (avec marge de confiance)
            name, distance = recognizer.find_best_match(
                embedding, persons_db, threshold=MATCH_THRESHOLD, confidence_margin=CONFIDENCE_MARGIN
            )

            # Debug : afficher les distances (toutes les 15 frames par visage)
            if DEBUG_DISTANCES and frame_count % 15 == 0:
                all_d = [(n, recognizer.cosine_distance(embedding, e)) for n, e in persons_db if e is not None]
                all_d.sort(key=lambda x: x[1])
                best_n, best_d = all_d[0] if all_d else ("?", float("inf"))
                second_d = all_d[1][1] if len(all_d) > 1 else float("inf")
                margin = second_d - best_d
                reason = "OK" if name else ("seuil" if best_d >= MATCH_THRESHOLD else "marge")
                print(f"  [DEBUG] meilleur: {best_n}={best_d:.3f}, 2e={second_d:.3f}, marge={margin:.3f} -> {reason}")

            # ---------------- VISAGE CONNU ----------------
            if name:
                label = f"{name} ({distance:.2f})"
                color = (0, 255, 0)  # vert

            # ---------------- VISAGE INCONNU ----------------
            else:
                label = "UNKNOWN"
                color = (0, 165, 255)  # orange

                # Suivi spatial : associer a une piste existante NON deja utilisee ce frame
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tid = None
                best_d = float("inf")
                for t_id, t in face_tracks.items():
                    if t_id in used_tracks_this_frame:
                        continue
                    tc = t["center"]
                    d = np.sqrt((cx - tc[0]) ** 2 + (cy - tc[1]) ** 2)
                    if d < TRACK_MAX_DIST and d < best_d:
                        best_d = d
                        tid = t_id
                if tid is None:
                    tid = next_track_id
                    next_track_id += 1
                used_tracks_this_frame.add(tid)

                # Mise a jour ou creation de la piste
                if tid not in face_tracks:
                    face_tracks[tid] = {"center": (cx, cy), "buffer": [], "face_crop": face_crop}
                face_tracks[tid]["center"] = (cx, cy)
                face_tracks[tid]["buffer"].append(embedding)
                face_tracks[tid]["face_crop"] = face_crop

                buf = face_tracks[tid]["buffer"]
                if len(buf) >= SAMPLES_TO_SAVE:
                    avg_emb = np.mean(np.stack(buf[-SAMPLES_TO_SAVE:]), axis=0)
                    is_duplicate = False
                    for _, u_emb in unknown_manager.get_all_unknown_embeddings().items():
                        _, d = recognizer.compare_embeddings(avg_emb, u_emb, UNKNOWN_DUP_THRESHOLD)
                        if d < UNKNOWN_DUP_THRESHOLD:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unknown_manager.add_unknown_face(avg_emb, face_tracks[tid]["face_crop"])
                    face_tracks.pop(tid, None)

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
    print("[INFO] Arret du systeme")


if __name__ == "__main__":
    main()
