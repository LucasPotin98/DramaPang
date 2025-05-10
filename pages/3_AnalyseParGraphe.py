import streamlit as st
import matplotlib.pyplot as plt
from app.visualization import plot_character_graph, plot_decision_tree_highlighted, plot_pattern
from app.loading import load_dracor_data
# === Chargement des données ===
X_full, Graphes, Patterns, labels, titles, noms, model = load_dracor_data()

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

prediction_text = "Comédie" if prediction[0] == 1 else "Tragédie"

st.markdown(
    f"""
    <div style='background-color:rgba(0, 0, 0, 0); padding: 1rem; border-radius: 8px; text-align: center;'>
        <h3 style='margin-bottom: 0.5rem;'>Le modèle prédit :</h3>
        <p style='font-size: 1.5rem; font-weight: bold;'>🎭 {prediction_text}</p>
    </div>
    """,
    unsafe_allow_html=True
)



# === Affichage de l'arbre de décision ===
st.markdown("### 🌳 Arbre de décision utilisé")
# Récupération des noms de motifs utilisés (indices des features actives dans le modèle)
feature_indices = model.feature_names_in_ if hasattr(model, "feature_names_in_") else [f"Motif {i}" for i in range(X_full.shape[1])]
fig_tree, patternsSelected = plot_decision_tree_highlighted(
    model,
    representation,
    feature_names=feature_indices,
    max_depth=4
)
st.pyplot(fig_tree)

# === 5. Interprétation des motifs
st.markdown("### 💡 Sous-graphes discriminants")

if patternsSelected:
    n = len(patternsSelected)
    n_cols = min(4, n)  # maximum 4 colonnes
    cols = st.columns(n_cols)

    for i, (feature, value) in enumerate(patternsSelected):
        with cols[i % n_cols]:
            st.markdown(f"**Motif {i + 1} – #{feature}**")
            G = Patterns[feature]
            fig = plot_pattern(G)  # version compacte
            st.pyplot(fig)
            presence = "Motif Présent" if value == 1 else "Motif Absent"
            st.markdown(f"<div style='text-align: center;'>{presence}</div>", unsafe_allow_html=True)
else:
    st.warning("Aucun motif discriminant trouvé.")

