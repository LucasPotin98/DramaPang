import streamlit as st
import matplotlib.pyplot as plt

from pang.db import (
    get_connection,
    get_dataset_id,
    get_graphs_for_dataset,
    get_pattern_dict_for_dataset,
    get_patterns_for_graph
)
from pang.utils import gspan_to_networkx, draw_graph
from pang.predict import load_model, predict, get_decision_path, constructRepresentation
from pang.classify import computePerfs

# === CONFIGURATION DE LA PAGE ===
st.set_page_config(page_title="Analyse par graphe", layout="wide")
st.header("🔍 Analyse d’une pièce")

# === Connexion base de données et chargement des ressources ===
conn = get_connection()
dataset_choice = "dracor"
dataset_id = get_dataset_id(conn, dataset_choice)
df_graphs = get_graphs_for_dataset(conn, dataset_id)
model = load_model(dataset_choice)
patternsDict = get_pattern_dict_for_dataset(conn, dataset_id)

# === Performances globales du modèle ===
st.markdown("### 📈 Performances globales")
mean, std = computePerfs(model, dataset_choice)
st.write(f"F1-score moyen (validation croisée) : **{mean:.2f}** ± {std:.2f}")

# === Sélection du graphe à analyser ===
st.markdown("### 🧪 Sélection de la pièce à analyser")
graph_options = df_graphs['graph_index'].tolist()
selected_graph = st.selectbox("Identifiant de la pièce :", graph_options)

# === Visualisation du graphe sélectionné ===
selected_gspan = df_graphs[df_graphs["graph_index"] == selected_graph]["gspan"].values[0]
G = gspan_to_networkx(selected_gspan)
fig = draw_graph(G, dataset_choice, figsize=(5, 5))
st.pyplot(fig)

# === Prédiction du modèle ===
st.markdown("### 🎯 Prédiction du modèle")
selected_patterns, total = get_patterns_for_graph(conn, dataset_id, selected_graph)
representation = constructRepresentation(selected_patterns, total)
prediction = predict(model, representation)
st.write(f"Le modèle prédit que cette pièce est une **{prediction[0].lower()}**.")

# === Interprétation : motifs discriminants ===
st.markdown("### 💡 Motifs discriminants utilisés pour la décision")
patterns = get_decision_path(model, representation)

if patterns:
    cols = st.columns(4)
    for i, (feature, value) in enumerate(patterns):
        with cols[i % 4]:
            st.markdown(f"**Motif {i+1}**")
            Gp = gspan_to_networkx(patternsDict[feature])
            fig = draw_graph(Gp, dataset_choice, figsize=(3, 3))
            st.pyplot(fig)
            st.markdown(
                f"<div style='text-align: center;'>{'Présent' if value == 1 else 'Absent'}</div>",
                unsafe_allow_html=True
            )
else:
    st.warning("Aucun motif discriminant identifié pour cette pièce.")
