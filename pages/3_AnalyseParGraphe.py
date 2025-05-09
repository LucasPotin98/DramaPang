import streamlit as st
import matplotlib.pyplot as plt
from app.visualization import plot_character_graph, plot_decision_tree_highlighted

# === Chargement des données ===
X_full, Graphes, labels, titles, noms, model = st.session_state.dracor_data

# === CHOIX DU GRAPHE ===
st.markdown("### 🎭 Sélection de la pièce")
selected_title = st.selectbox("Choisir une pièce :", titles)
index = titles.index(selected_title)
G = Graphes[index]

fig = plot_character_graph(G, title=selected_title, node_names=noms[index])
st.plotly_chart(fig, use_container_width=True)

# === PRÉDICTION ===
st.markdown("### 🎯 Prédiction")
representation = X_full[index]
prediction = model.predict([representation])

color = "#e8e8e8"
st.markdown(
    f"""
    <div style='background-color:{color}; padding: 1rem; border-radius: 8px; text-align: center;'>
        <h3 style='margin-bottom: 0.5rem;'>Le modèle prédit :</h3>
        <p style='font-size: 1.5rem; font-weight: bold;'>{"Comédie" if prediction[0]==1 else "Tragédie"}</p>
    </div>
    """, unsafe_allow_html=True
)

# === Affichage de l'arbre de décision ===
st.markdown("### 🌳 Arbre de décision utilisé")
# Récupération des noms de motifs utilisés (indices des features actives dans le modèle)
feature_indices = model.feature_names_in_ if hasattr(model, "feature_names_in_") else [f"Motif {i}" for i in range(X_full.shape[1])]
fig_tree = plot_decision_tree_highlighted(
    model,
    representation,
    feature_names=feature_indices,
    max_depth=4
)
st.pyplot(fig_tree)