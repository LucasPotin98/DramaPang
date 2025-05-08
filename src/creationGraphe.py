import os
import networkx as nx
from collections import defaultdict, Counter
import re
# --- Chemins de base ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
CHAR_FILE = os.path.join(DATA_DIR, 'movie_characters_metadata.txt')
LINES_FILE = os.path.join(DATA_DIR, 'movie_lines.txt')
CONV_FILE = os.path.join(DATA_DIR, 'movie_conversations.txt')
MOVIES_FILE = os.path.join(DATA_DIR, 'movie_titles_metadata.txt')
OUTPUT_FILE = os.path.join(DATA_DIR, 'Cornell_graphs.txt')
LABELS_OUT = os.path.join(DATA_DIR, 'Cornell_labels.txt')

DRAMATIC_GENRES = {
    "drama", "crime", "thriller", "mystery", "biography", "documentary", "film-noir", "history", "horror"
}

# --- Map genre (H/F/UNK) vers label numérique ---
def gender_to_label(g):
    return {'H': 0, 'F': 1}.get(g, 2)

# --- Discrétisation du poids d’arête ---
def discretize_weight(w):
    if w <= 10:
        return 0  # faible
    elif w <= 20:
        return 1  # modérée
    elif w <= 30:
        return 2  # forte
    else:
        return 3  # très forte

# --- Chargement des genres des personnages ---
def load_character_genders():
    char_to_gender = {}
    with open(CHAR_FILE, encoding='iso-8859-1') as f:
        for line in f:
            fields = line.strip().split("+++$+++")
            if len(fields) >= 5:
                char_id = fields[0].strip()
                gender = fields[4].strip().lower()
                if gender == 'm':
                    char_to_gender[char_id] = 'H'
                elif gender == 'f':
                    char_to_gender[char_id] = 'F'
                else:
                    char_to_gender[char_id] = 'UNK'
    return char_to_gender

# --- Chargement des lignes de dialogue ---
def load_line_characters():
    line_to_char = {}
    with open(LINES_FILE, encoding='iso-8859-1') as f:
        for line in f:
            fields = line.strip().split("+++$+++")
            if len(fields) >= 5:
                line_id = fields[0].strip()
                char_id = fields[1].strip()
                line_to_char[line_id] = char_id
    return line_to_char

# --- Construction des graphes (1 film = 1 graphe) ---
def build_graphs_per_film(char_to_gender, line_to_char):
    graphs = defaultdict(lambda: nx.Graph())

    with open(CONV_FILE, encoding='iso-8859-1') as f:
        for line in f:
            fields = line.strip().split("+++$+++")
            if len(fields) != 4:
                continue

            char1, char2, movie_id, utterance_ids_str = [s.strip() for s in fields]
            try:
                utterance_ids = eval(utterance_ids_str)
            except:
                continue

            lines = [lid for lid in utterance_ids if lid in line_to_char]
            if not lines:
                continue

            G = graphs[movie_id]

            for char in [char1, char2]:
                if not G.has_node(char):
                    gender = char_to_gender.get(char, 'UNK')
                    G.add_node(char, label=gender_to_label(gender))

            if G.has_edge(char1, char2):
                G[char1][char2]['weight'] += len(lines)
            else:
                G.add_edge(char1, char2, weight=len(lines))

    return graphs

# --- Sauvegarde en GSPAN (1 seul fichier pour tous les graphes) ---
def save_all_graphs_to_gspan(graphs, output_file):
    with open(output_file, 'w') as f:
        gid = 0
        for movie_id, G in graphs.items():
            if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
                continue

            f.write(f"t # {gid}\n")
            node_map = {}
            for i, node in enumerate(G.nodes()):
                label = G.nodes[node]['label']
                f.write(f"v {i} {label}\n")
                node_map[node] = i

            for u, v, data in G.edges(data=True):
                raw_weight = int(data.get('weight', 1))
                edge_label = discretize_weight(raw_weight)
                f.write(f"e {node_map[u]} {node_map[v]} {edge_label}\n")

            gid += 1

    print(f"✅ {gid} graphes sauvegardés dans : {output_file}")

def load_movie_genres(path):
    movie_genres = {}
    with open(path, encoding='iso-8859-1') as f:
        for line in f:
            fields = line.strip().split("+++$+++")
            if len(fields) >= 6:
                movie_id = fields[0].strip()
                genres = fields[-1].split()[0].strip()
                genres = re.sub(r"[,\[\]'\"]", '', genres).strip()
                print(genres)
                movie_genres[movie_id] = genres

    return movie_genres

# --- Générer les labels binaires groupés (label d'abord) ---
def build_grouped_label_file(film_graphs, movie_genres, output_path):
    with open(output_path, 'w') as f:
        gid = 0
        for movie_id, G in film_graphs.items():
            if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
                continue
            genre = movie_genres.get(movie_id, 'unknown')
            label = 0 if genre in DRAMATIC_GENRES else 1
            f.write(f"{label} {gid}\n")
            gid += 1
    print(f"✅ Fichier de labels binaires sauvegardé : {output_path}")

# --- MAIN ---
if __name__ == "__main__":

    print("🔧 Reconstruction des graphes...")
    char_to_gender = load_character_genders()
    line_to_char = load_line_characters()
    film_graphs = build_graphs_per_film(char_to_gender, line_to_char)
    print(film_graphs)

    print("🔧 Sauvegarde des graphes...")
    save_all_graphs_to_gspan(film_graphs, OUTPUT_FILE)

    print("📥 Lecture des genres...")
    movie_genres = load_movie_genres(MOVIES_FILE)
    print(movie_genres)

    print("📝 Écriture de labels_grouped.txt (label, graph_id)...")
    build_grouped_label_file(film_graphs, movie_genres, LABELS_OUT)