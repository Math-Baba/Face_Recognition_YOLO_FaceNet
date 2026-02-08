import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import cv2

"""
Classe responsable de l'extraction des embeddings faciaux
et de la comparaison des visages à l'aide de FaceNet.
"""
class FaceRecognizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Chargement du modèle FaceNet pré-entraîné (VGGFace2)
        self.model = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()

        # Pipeline de prétraitement des images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),   # Conversion OpenCV → PIL
            transforms.Resize((160, 160)), # Taille requise par FaceNet
            transforms.ToTensor(),     # PIL → Tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisation des pixels
        ])

        # Cache léger pour embeddings récents (non persistant)
        self._embedding_cache = {}

    # Normalise un embedding (L2 normalization) pour permettre une comparaison fiable
    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        if emb is None:
            return None
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    # Extrait un embedding facial normalisé à partir d'une image de visage
    def get_embedding(self, face_img):
        if face_img is None or face_img.size == 0:
            return None

        # Ignore les visages trop petits (peu exploitables)
        if face_img.shape[0] < 50 or face_img.shape[1] < 50:
            return None

        try:
            # Conversion BGR → RGB
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_img

            # Prétraitement et passage dans le modèle
            tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(tensor)
            emb = embedding.cpu().numpy()[0]
            return self._normalize(emb)
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

    # Calcule la distance cosinus entre deux embeddings
    # Plus la valeur est faible, plus les visages sont similaires
    def cosine_distance(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return float('inf')
        e1 = self._normalize(np.array(emb1, dtype=float))
        e2 = self._normalize(np.array(emb2, dtype=float))
        return 1.0 - float(np.clip(np.dot(e1, e2), -1.0, 1.0))

    # Compare deux embeddings faciaux (retourne match -> bool, distance cosinus)
    def compare_embeddings(self, emb1, emb2, threshold=0.4):
        if emb1 is None or emb2 is None:
            return False, float('inf')
        dist = self.cosine_distance(emb1, emb2)
        return dist < threshold, dist

    # Recherche la meilleure correspondance dans la base de données
    def find_best_match(self, embedding, persons_db, threshold=0.4):
        if not persons_db or embedding is None:
            return None, float('inf')

        best_match = None
        best_distance = float('inf')

        for person_name, person_emb in persons_db:
            if person_emb is None:
                continue
            _, dist = self.compare_embeddings(embedding, person_emb, threshold)
            if dist < best_distance:
                best_distance = dist
                best_match = person_name if dist < threshold else None

        return best_match, best_distance
