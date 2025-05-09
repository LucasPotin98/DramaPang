import plotly.graph_objects as go
import networkx as nx
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np
import re

def plot_character_graph(G, title=None, node_names=None):
    pos = nx.spring_layout(G, seed=42)

    # === 1. Traces pour les arêtes, par type de poids ===
    edge_traces = {
    "Faible interaction (1 fois)": {"x": [], "y": [], "color": "gray"},
    "Interaction modérée (2 à 5 fois)": {"x": [], "y": [], "color": "black"},
    "Forte interaction (> 5 fois)": {"x": [], "y": [], "color": "red"},
}


    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get("color", 1)

        if weight == 1:
            key = "Faible interaction (1 fois)"
        elif weight == 2:
            key = "Interaction modérée (2 à 5 fois)"
        else:
            key = "Forte interaction (> 5 fois)"

        edge_traces[key]["x"].extend([x0, x1, None])
        edge_traces[key]["y"].extend([y0, y1, None])

    # === 2. Traces pour les nœuds, par genre ===
    node_traces = {
        "Homme": {"x": [], "y": [], "text": [], "color": "skyblue"},
        "Femme": {"x": [], "y": [], "text": [], "color": "lightpink"},
        "Inconnu": {"x": [], "y": [], "text": [], "color": "lightgray"},
    }

    for node in G.nodes():
        x, y = pos[node]
        label = G.nodes[node].get("color", 2)
        name = node_names[node] if node_names else str(node)

        if label == 0:
            key = "Homme"
        elif label == 1:
            key = "Femme"
        else:
            key = "Inconnu"

        node_traces[key]["x"].append(x)
        node_traces[key]["y"].append(y)
        node_traces[key]["text"].append(name)

    # === 3. Création de la figure ===
    fig = go.Figure()

    # Arêtes
    for label, data in edge_traces.items():
        fig.add_trace(go.Scatter(
            x=data["x"], y=data["y"],
            mode="lines",
            line=dict(color=data["color"], width=2),
            hoverinfo="none",
            name=label
        ))

    # Nœuds
    for genre, data in node_traces.items():
        fig.add_trace(go.Scatter(
            x=data["x"], y=data["y"],
            mode="markers+text",
            text=data["text"],
            textposition="top center",
            marker=dict(
                color=data["color"],
                size=22,
                line=dict(width=1.5, color="black")
            ),
            hoverinfo="text",
            name=genre
        ))

    # === 4. Layout ===
    fig.update_layout(
        title=dict(text=title or "", x=0.5, xanchor="center"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=14),
        legend=dict(
            x=1.05,
            y=1,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=16, color="black"),
            traceorder="normal",
            itemclick=False,
            itemdoubleclick=False
        ),
        margin=dict(l=20, r=200, t=60, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=True
    )

    return fig



def plot_decision_tree_highlighted(model, X_instance, feature_names=None, class_names=None, max_depth=4):
    """
    Affiche l'arbre de décision avec les nœuds du chemin de décision surlignés en rouge.

    Args:
        model: un modèle sklearn DecisionTreeClassifier déjà entraîné
        X_instance: vecteur numpy 1D représentant une pièce (exemple unique)
        feature_names: liste des noms des features (motifs)
        class_names: liste des classes (["Tragédie", "Comédie"])
        max_depth: profondeur maximale d'affichage
    """
    # Étape 1 : récupération du chemin
    path_nodes = model.decision_path(X_instance.reshape(1, -1)).indices

    # Étape 2 : affichage de l'arbre avec les IDs de nœud visibles
    fig, ax = plt.subplots(figsize=(18, 6))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        max_depth=max_depth,
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=10,
        ax=ax,
        node_ids=True  # nécessaire pour récupérer l'id du noeud
    )

    # Étape 3 : surlignage des nœuds du chemin
    for text in ax.texts:
        s = text.get_text()
        match = re.search(r'#(\d+)', s)  # récupère le numéro du noeud
        if match:
            node_id = int(match.group(1))
            if node_id in path_nodes:
                bbox = text.get_bbox_patch()
                if bbox:
                    bbox.set_edgecolor("red")
                    bbox.set_linewidth(3)

    return fig