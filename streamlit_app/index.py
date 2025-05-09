import streamlit as st

st.set_page_config(page_title="🎬 MovieGraph Analyzer", layout="wide")

st.title("🎬 MovieGraph Analyzer")
st.markdown("""
Bienvenue sur **MovieGraph Analyzer**, une application interactive qui explore les **structures conversationnelles** entre personnages de films à travers des **graphes de dialogue**.

---

🧠 **Objectif :** Détecter des motifs de structure liés au genre du film (*action*, *drame*, etc.), en combinant analyse graphe et contenu textuel.

📚 Données : issues du [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), enrichies avec :
- Le genre des personnages (H/F/UNK),
- Les relations de dialogue (quantité, tonalité, nature),
- La polarité des interactions.

📈 **Navigation :**
- 🧾 Dataset : comment les graphes ont été construits, stats sur les dialogues.
- 🤖 Classification : résultats sur les modèles graphe / texte.
- 🔍 Analyse de motifs : motifs récurrents et discriminants par genre.
""")
