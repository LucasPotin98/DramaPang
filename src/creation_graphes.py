import pandas as pd
import networkx as nx
import os

NETWORK_DIR = "../data/dracor_networks_fre"
CHARACTER_DIR = "../data/dracor_characters_fre"
METADATA_FILE = "../data/metadata_dracor.csv"

GSPAN_FILE = "../data/graphs/dracor_graphs.txt"
LABEL_FILE = "../data/graphs/dracor_labels.txt"
TITLE_FILE = "../data/graphs/dracor_titles.txt"

def discretize_weight(w):
    if w == 1:
        return 1
    elif w <= 5:
        return 2
    else:
        return 3

def graph_to_gspan(G: nx.Graph, graph_id: int) -> str:
    node_ids = {node: i for i, node in enumerate(G.nodes())}
    lines = [f"t # {graph_id}"]

    for node, idx in node_ids.items():
        label = G.nodes[node].get("gender", "UNKNOWN")
        if label == "MALE":
            label = 0
        elif label == "FEMALE":
            label = 1
        else:
            label = 2
        name = node.replace(" ", "_")  # éviter les espaces dans les noms
        lines.append(f"v {idx} {label} # {name}")

    for u, v, data in G.edges(data=True):
        src = node_ids[u]
        tgt = node_ids[v]
        weight = data.get("weight", 1)
        label = discretize_weight(weight)
        lines.append(f"e {src} {tgt} {label}")

    return "\n".join(lines)

metadata_df = pd.read_csv(METADATA_FILE).set_index("name")

gspan_lines = []
label_lines = []
title_lines = []

counts = {0: 0, 1: 0}
LIMIT = 200

for file in sorted(os.listdir(NETWORK_DIR)):
    if not file.endswith(".csv"):
        continue

    play_id = file.replace(".csv", "")
    try:
        df_net = pd.read_csv(os.path.join(NETWORK_DIR, file))
        df_net.rename(columns={"Source": "source", "Target": "target", "Weight": "weight"}, inplace=True)
        G = nx.from_pandas_edgelist(df_net, source="source", target="target", edge_attr="weight")

        if not nx.is_connected(G):
            continue

        char_file = os.path.join(CHARACTER_DIR, f"{play_id}.csv")
        if not os.path.exists(char_file):
            continue

        df_char = pd.read_csv(char_file)
        for _, row in df_char.iterrows():
            name = row["id"]
            if name in G.nodes:
                G.nodes[name]["gender"] = row.get("gender", "UNKNOWN")

        genre = metadata_df.loc[play_id]["normalizedGenre"]
        if genre == "Comedy":
            label = 1
        elif genre == "Tragedy":
            label = 0
        else:
            continue

        if counts[label] >= LIMIT:
            continue

        gspan_lines.append(graph_to_gspan(G, len(gspan_lines)))
        label_lines.append(str(label))
        title_lines.append(metadata_df.loc[play_id]["title"])  # ou play_id si tu préfères
        counts[label] += 1

    except Exception as e:
        print(f"❌ Erreur pour {play_id} : {e}")

# === Sauvegardes
with open(GSPAN_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(gspan_lines))

with open(LABEL_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(label_lines))

with open(TITLE_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(title_lines))

print(f"\n✅ Export terminé : {len(gspan_lines)} graphes dans {GSPAN_FILE}")
print(f"✅ Répartition : Tragedy={counts[0]} | Comedy={counts[1]}")
print(f"✅ Titres enregistrés dans {TITLE_FILE}")
