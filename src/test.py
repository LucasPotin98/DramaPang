from grakel import GraphKernel
from grakel import Graph
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
import networkx as nx
import grakel
# --- Lecture GSPAN ---
def load_graphs(file_name, nb_graphs):
    """
    Load graphs from a SPMF-like file.
    
    Args:
        file_name (str): Path to the graph file.
        nb_graphs (int): Number of graphs expected in the file.
    
    Returns:
        graphs (list of networkx.Graph): The list of graphs.
        occurrences (list of list of int): Occurrence vectors for each graph.
        raw_texts (list of str): Raw text of each graph (used for traceability).
    """
    graphs = []
    occurrences = [[] for _ in range(nb_graphs)]
    raw_texts = []

    node_labels = []
    edge_lists = []

    current_text = ""
    current_graph_index = -1

    with open(file_name, "r") as f:
        for line in f:
            tokens = line.strip().split()

            if not tokens:
                continue

            if tokens[0] == "t":
                if current_graph_index >= 0:
                    raw_texts.append(current_text)
                current_graph_index += 1
                current_text = ""
                node_labels.append([])
                edge_lists.append([])

            elif tokens[0] == "v":
                label = int(tokens[2])
                node_labels[current_graph_index].append(label)
                current_text += line

            elif tokens[0] == "e":
                u, v, label = int(tokens[1]), int(tokens[2]), int(tokens[3])
                edge_lists[current_graph_index].append((u, v, label))
                current_text += line

            elif tokens[0] == "x":
                occurrence = [int(tok) for tok in tokens[1:] if tok != "#"]
                occurrences[current_graph_index] = occurrence

        # Append the last graph's text
        if current_text:
            raw_texts.append(current_text)

    # Build NetworkX graphs
    for i in range(nb_graphs):
        G = nx.Graph()
        for idx, label in enumerate(node_labels[i]):
            G.add_node(idx, color=label)
        for u, v, label in edge_lists[i]:
            G.add_edge(u, v, color=label)
        graphs.append(G)

    return graphs, occurrences, raw_texts

# --- Lecture des labels ---
def load_labels(path):
    y = []
    with open(path, 'r') as f:
        for line in f:
            label, _ = line.strip().split()
            y.append(int(label))
    return np.array(y)


# --- MAIN ---
graphs,xx,xx = load_graphs('../data/Cornell_graphs.txt', 617)
y = load_labels('../data/Cornell_labels.txt')

grakel_graphs = grakel.graph_from_networkx(graphs,node_labels_tag='color',edge_labels_tag='color')

# --- WL kernel ---
print("⚙️  Classification avec WL kernel + SVM...")
gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 3}, {"name": "vertex_histogram"}], normalize=True)
K = gk.fit_transform(grakel_graphs)

clf = SVC(kernel='precomputed')
scores = cross_val_score(clf, K, y, cv=5, scoring='f1')
print(f"🎯 F1-score (WL kernel) : {scores.mean():.3f} ± {scores.std():.3f}")
