import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from pang.pang import (
    pang_load_and_represent,
    compute_scores,
    pang_classify_with_selection
)

# === PARAMS ===
FILE_GRAPHS = "data/graphs/dracor_graphs.txt"
FILE_PATTERNS = "data/graphs/dracor_patterns.txt"
FILE_LABELS = "data/graphs/dracor_labels.txt"

st.header("🤖 Classification des pièces")

# === Chargement et représentation ===
st.markdown("Chargement des données et construction de la matrice binaire complète...")
X_full, Graphes, labels = pang_load_and_represent(FILE_GRAPHS, FILE_PATTERNS, FILE_LABELS)

# === Choix de la mesure ===
measure = st.selectbox(
    "Choisir une mesure de qualité pour scorer les motifs :",
    ["Sup", "AbsSupDif", "WRAcc"]
)

# === Calcul des scores ===
st.markdown(f"Calcul des scores des motifs avec la mesure `{measure}`...")
scores = compute_scores(X_full, labels, measure)
print(scores)
# === Affichage des scores ===
st.subheader("📊 Distribution des scores")
fig, ax = plt.subplots()
ax.plot(np.sort(scores)[::-1])
ax.set_title("Scores décroissants des motifs")
ax.set_xlabel("Motifs (triés)")
ax.set_ylabel("Score")
st.pyplot(fig)

mode = st.selectbox("Sélection des motifs", ["Top-k motifs", "Tous les motifs"])

if mode == "Top-k motifs":
    top_k = st.slider("Nombre de motifs à sélectionner", 10, 300, 100)
else:
    top_k = X_full.shape[1]

# === Classification ===
if st.button("Lancer la classification"):
    st.markdown("🧠 Entraînement du modèle SVM en validation croisée...")
    f1, selected = pang_classify_with_selection(X_full, labels, scores, top_k=top_k, cv=5)
    
    st.success(f"🎯 F1-score moyen (CV = 5) : **{f1:.3f}**")
    st.markdown(f"Nombre de motifs utilisés : **{len(selected)}**")

    # Matrice de confusion
    from pang.classify import evaluate_with_predictions
    from sklearn.metrics import ConfusionMatrixDisplay

    from pang.vectorize import build_topk_representation
    X_k, _ = build_topk_representation(X_full, scores, top_k)
    _, y_true, y_pred = evaluate_with_predictions(X_k, labels)

    st.subheader("🔢 Matrice de confusion")
    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Tragédie", "Comédie"])
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)
