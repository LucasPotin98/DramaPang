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
X_full, Graphes, Patterns, labels, titles, noms, model = st.session_state.dracor_data

# === Choix de la mesure ===
measure = st.selectbox(
    "Choisir une mesure de qualité pour scorer les motifs :",
    ["Sup", "AbsSupDif", "WRAcc"]
)

# === Description dynamique de la mesure sélectionnée ===
if measure == "Sup":
    st.markdown("**🧮 Mesure : Support (Sup)**")
    st.latex(r"\text{Sup}(P) = \frac{\text{nb graphes contenant } P}{\text{nb total de graphes}}")
    st.markdown("Mesure simple, utile pour filtrer les motifs fréquents. Ne considère toutefois pas la classe négative.")

elif measure == "AbsSupDif":
    st.markdown("**🧮 Mesure : Absolute Support Difference (AbsSupDif)**")
    st.markdown("On note les supports du motif \\( P \\) dans chaque classe :")
    st.latex(r"\text{Sup}_+(P) = \frac{\text{nb graphes positifs contenant } P}{\text{nb graphes positifs}}")
    st.latex(r"\text{Sup}_-(P) = \frac{\text{nb graphes négatifs contenant } P}{\text{nb graphes négatifs}}")
    st.markdown("La mesure est alors définie par :")
    st.latex(r"\text{AbsSupDif}(P) = \left| \text{Sup}_+(P) - \text{Sup}_-(P) \right|")
    st.markdown("Elle favorise les motifs très asymétriques entre classes.")


elif measure == "WRAcc":
    st.markdown("**🧮 Mesure : Weighted Relative Accuracy (WRAcc)**")
    st.markdown("Soit un motif \\( P \\), on définit les éléments suivants :")
    st.latex(r"s = \text{Sup}(P) = \frac{\text{nb graphes contenant } P}{\text{nb total de graphes}}")
    st.latex(r"p = \Pr(\text{classe} = +1 \mid P)")
    st.latex(r"\pi = \Pr(\text{classe} = +1)")
    st.markdown("La mesure WRAcc est alors donnée par :")
    st.latex(r"\text{WRAcc}(P) = s \cdot (p - \pi)")
    st.markdown("Elle pondère le pouvoir discriminant du motif par sa fréquence dans l’ensemble.")





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
