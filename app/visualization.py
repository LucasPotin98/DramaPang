import plotly.graph_objects as go
import networkx as nx

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
