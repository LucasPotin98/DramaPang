import streamlit as st

st.set_page_config(page_title="DramaPang", layout="wide")

st.title("🎭 DramaPang")
st.markdown("""
Bienvenue sur **DramaPang**, une application interactive pour explorer les réseaux de personnages dans les pièces de théâtre françaises, et les classifier en **comédies** ou **tragédies**.

---

### 🧠 Objectif
Ce projet utilise des graphes extraits de pièces issues de [DraCor](https://dracor.org/) pour prédire leur genre à partir des interactions entre personnages.

La classification est effectuée avec le framework **PANG** (*Pattern-based Anomaly detection in Graphs*), un outil fondé sur l'extraction et la sélection de motifs discriminants.

---

### 📦 Données
- **400 graphes** issus de pièces françaises : 200 comédies, 200 tragédies
- Nœuds = personnages (avec genre)
- Arêtes = co-présence dans un ou plusieurs actes

---

### 📂 Navigation
Utilisez les pages sur la gauche pour :
- explorer les données et les graphes ;
- consulter les résultats de classification ;
- visualiser les motifs les plus discriminants ;
- analyser une pièce spécifique.

---

### 🔗 Liens utiles
- Base de données DraCor : [dracor.org](https://dracor.org/)
- Framework PANG : [GitHub – DramaPang](https://github.com/CompNet/Pang)
- Code source du projet : [GitHub – DramaPang](https://github.com/LucasPotin98/DramaPang)
- Mon site web : [lucaspotin98.github.io](https://lucaspotin98.github.io/)

---
""")
