import streamlit as st
from pang.pang import pang_load_and_represent, load_titles, load_model

st.set_page_config(page_title="DramaPang", layout="wide")

# === Chargement silencieux des données (une seule fois) ===
if "dracor_data" not in st.session_state:
    FILE_GRAPHS = "data/graphs/dracor_graphs.txt"
    FILE_PATTERNS = "data/graphs/dracor_patterns.txt"
    FILE_LABELS = "data/graphs/dracor_labels.txt"
    FILE_TITLES = "data/graphs/dracor_titles.txt"
    FILE_MODEL = "models/pang_model.pkl"
    X_full, Graphes, Patterns, labels, noms = pang_load_and_represent(FILE_GRAPHS, FILE_PATTERNS, FILE_LABELS)
    titles = load_titles(FILE_TITLES)
    model = load_model(FILE_MODEL)
    st.session_state.dracor_data = (X_full, Graphes, Patterns, labels, titles, noms, model)

st.title("🎭 DramaPang")
st.markdown("""
Bienvenue sur **DramaPang**, une application interactive pour explorer les réseaux de personnages dans les pièces de théâtre françaises, et les classifier en **comédies** ou **tragédies**.

---

### Objectif
Ce projet utilise des graphes extraits de pièces issues de [DraCor](https://dracor.org/) pour prédire leur genre à partir des interactions entre personnages.

La classification est effectuée avec le framework [PANG](https://github.com/CompNet/Pang) (*Pattern-based Anomaly detection in Graphs*), un outil fondé sur l'extraction et la sélection de motifs discriminants.

---

### Données

Le dataset est composé de **400 graphes**, chacun représentant une pièce de théâtre française :

- **200 comédies** (label `0`)
- **200 tragédies** (label `1`)

Chaque graphe est construit à partir des interactions entre personnages :

- **Nœuds** : personnages
  - Label lié au `genre` :
    - `MALE` → représenté en bleu
    - `FEMALE` → représentée en rose
    - `UNKNOWN` → représenté(e) en gris

- **Arêtes** : co-présence de deux personnages dans un ou plusieurs actes
  - Pondération discrétisée en trois niveaux :
    - 1 seule co-présence → représentée en noir
    - 2 à 5 co-présences → représentées en gris
    - plus de 5 co-présences → représentées en rouge

---

### Navigation
Utilisez les pages sur la gauche pour :
- explorer les données et les graphes ;
- consulter les résultats de classification ;
- analyser une pièce spécifique.

---

### 🔗 Liens utiles
- Base de données DraCor : [dracor.org](https://dracor.org/)
- Framework PANG : [GitHub – DramaPang](https://github.com/CompNet/Pang)
- Code source du projet : [GitHub – DramaPang](https://github.com/LucasPotin98/DramaPang)
- Mon site web : [lucaspotin98.github.io](https://lucaspotin98.github.io/)

---
""")
