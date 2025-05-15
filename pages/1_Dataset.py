import streamlit as st
import matplotlib.pyplot as plt
from app.visualization import plot_character_graph
from app.loading import load_dracor_data

st.header("Dataset : pièces et graphes")

# === Chargement des données ===
X_full, Graphes, Patterns, labels, titles, noms, model = load_dracor_data()

# === Statistiques globales ===
st.subheader("Statistiques générales")
n_graphs = len(Graphes)
n_comedies = sum(1 for label in labels if label == 1)
n_tragedies = sum(1 for label in labels if label == 0)
n_motifs = X_full.shape[1]

st.markdown(
    f"""
- **Nombre total de pièces** : {n_graphs}
- **Comédies** : {n_comedies}  
- **Tragédies** : {n_tragedies}  
- **Nombre de motifs extraits** : {n_motifs}
"""
)

# === Colonnes pour histogrammes ===
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Taille des graphes (personnages)**")
    sizes = [G.number_of_nodes() for G in Graphes]
    fig1, ax1 = plt.subplots()
    ax1.hist(sizes, bins=20, color="gray", edgecolor="black")
    ax1.set_xlabel("Nombre de personnages")
    ax1.set_ylabel("Nombre de pièces")
    st.pyplot(fig1)

with col2:
    st.markdown("**Connectivité (degré moyen)**")
    avg_degrees = [
        sum(dict(G.degree()).values()) / G.number_of_nodes() for G in Graphes
    ]
    fig2, ax2 = plt.subplots()
    ax2.hist(avg_degrees, bins=20, color="skyblue", edgecolor="black")
    ax2.set_xlabel("Degré moyen")
    ax2.set_ylabel("Nombre de pièces")
    st.pyplot(fig2)

# === Visualisation d’un graphe ===
st.subheader("🎭 Exemple de graphe")

selected_title = st.selectbox("Choisir une pièce :", titles)
index = titles.index(selected_title)
G = Graphes[index]

fig = plot_character_graph(G, title=selected_title, node_names=noms[index])
st.plotly_chart(fig, use_container_width=True)
