import pickle
from pathlib import Path
from datetime import datetime
import cv2


class UnknownFaceManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent / "data" / "unknown_faces" # Dossier de stockage des visages inconnus et embeddings
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.file = self.base_dir / "unknown_embeddings.pkl"  # Fichier de persistance des embeddings (pickle)
        self.data = self._load() # Chargement des données existantes

    # Charge les embeddings inconnus depuis le fichier pickle.
    def _load(self):
        if self.file.exists():
            with open(self.file, "rb") as f:
                return pickle.load(f)
        return {}

    # Sauvegarde l’état actuel des visages inconnus sur disque.
    def _save(self):
        with open(self.file, "wb") as f:
            pickle.dump(self.data, f)

    # Ajoute un nouveau visage inconnu avec son embedding
    # Sauvegarde également l’image du visage si fournie
    def add_unknown_face(self, embedding, frame=None):
        uid = f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.data[uid] = {
            "embedding": embedding,
            "name": None
        }

        if frame is not None:
            cv2.imwrite(str(self.base_dir / f"{uid}.jpg"), frame)

        self._save()
        return uid

    # Retourne uniquement les embeddings des visages encore non enregistrés (name == None).
    def get_all_unknown_embeddings(self):
        return {
            uid: d["embedding"]
            for uid, d in self.data.items()
            if d["embedding"] is not None and d["name"] is None
        }

    # Retourne tous les visages inconnus non encore associés à un nom.
    def get_unregistered_faces(self):
        return {
            uid: d
            for uid, d in self.data.items()
            if d["name"] is None
        }

    # Associe un nom à un visage inconnu après validation utilisateur.
    def register_unknown_face(self, face_id, name):
        if face_id in self.data:
            self.data[face_id]["name"] = name
            self._save()

    # Supprime définitivement un visage inconnu (embedding + image associée)
    def delete_unknown_face(self, face_id):
        if face_id in self.data:
            face_path = self.base_dir / f"{face_id}.jpg"
            if face_path.exists():
                face_path.unlink()
            del self.data[face_id]
            self._save()
