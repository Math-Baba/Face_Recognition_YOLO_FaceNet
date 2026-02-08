import os
import torch
from ultralytics import YOLO

# Chemin racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Chemin vers le modèle YOLO entraîné
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pt")

"""
Classe responsable de la détection des visages
à l'aide d'un modèle YOLO.
"""
class YOLODetector:
    def __init__(self, model_path=None):

        # Utilise le modèle par défaut si aucun chemin n'est fourni
        if model_path is None:
            model_path = MODEL_PATH
        
        # Séléction automatique du GPU si disponible, sinon CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Chargement du modèle YOLO
        self.model = YOLO(model_path)
        self.model.to(self.device)

    # Détecte les visages dans une image
    def detect_faces(self, frame, conf=0.6):

        # Lancement de l'inférence YOLO
        results = self.model(frame, conf=conf, imgsz=640, verbose=False, device=self.device)
        faces = []
        
        # Vérification de l'existence de visages détectés
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes.xyxy:

                # Coordonnées de la bounding box
                x1, y1, x2, y2 = [int(v) for v in box]

                # Calcul de la taille du visage
                height = y2 - y1
                width = x2 - x1

                # Ajout d'une marge autour du visage
                padding = int(max(height, width) * 0.1)
                
                # Limitations des coordonnées aux dimensions de l'image
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                
                faces.append((x1, y1, x2, y2))
        
        return faces
