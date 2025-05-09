import pickle
from sklearn.tree import DecisionTreeClassifier
import sys
import os

# Ajoute la racine du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pang.pang import pang_load_and_represent

# === Fichiers d’entrée
FILE_GRAPHS = "data/graphs/dracor_graphs.txt"
FILE_PATTERNS = "data/graphs/dracor_patterns.txt"
FILE_LABELS = "data/graphs/dracor_labels.txt"

# === Chargement des données
X_full, _, labels, _ = pang_load_and_represent(FILE_GRAPHS, FILE_PATTERNS, FILE_LABELS)

# === Entraînement du modèle

model = DecisionTreeClassifier(
    max_leaf_nodes=5,  # → max 4 splits = 4 motifs testés
    class_weight="balanced",
    random_state=42
)

model.fit(X_full, labels)

# === Sauvegarde du modèle
# Crée le dossier 'models' s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Sauvegarde du modèle
with open("models/pang_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modèle entraîné et sauvegardé dans models/pang_model.pkl")
