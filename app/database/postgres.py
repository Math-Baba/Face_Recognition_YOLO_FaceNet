import psycopg2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

"""
Classe de gestion de la base de données PostgreSQL 
pour le stockage et la récupération des visages connus
"""
class Database:
    def __init__(self, host=None, dbname=None, user=None, password=None):
        # Récupération des paramètres de connexion depuis .env
        host = host or os.getenv("DB_HOST")
        dbname = dbname or os.getenv("DB_NAME")
        user = user or os.getenv("DB_USER")
        password = password or os.getenv("DB_PASSWORD")

        # Connexion à la base de données PostgreSQL
        self.conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password
        )
        # Curseur pour exécuter les requêtes SQL
        self.cur = self.conn.cursor()

    # Insère une nouvelle personne dans la db (prénom + embedding -> vecteur facial)
    def insert_person(self, name, embedding):

        # Conversion du numpy array en liste
        embedding_list = embedding.tolist()  

        # Insertion dans la db
        self.cur.execute(
            "INSERT INTO persons (first_name, embedding) VALUES (%s, %s)",
            (name, embedding_list)
        )
        self.conn.commit()

    # Récupération de tous les visages dans la db
    def get_all_persons(self):
        self.cur.execute("SELECT first_name, embedding FROM persons")
        rows = self.cur.fetchall()

        # Reconversion des embeddings en numpy array (assure forme 1D, 512 dims)
        result = []
        for name, embedding in rows:
            if embedding is None:
                continue
            arr = np.array(embedding, dtype=float).flatten()
            if len(arr) != 512:
                print(f"[WARN] Embedding pour {name}: mauvaise taille {len(arr)}, attendu 512")
                continue
            result.append((name, arr))
        return result

    # Ferme proprement la connexion à la db
    def close(self):
        self.cur.close()
        self.conn.close()
