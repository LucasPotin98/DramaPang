import numpy as np

def ComputeRepresentationComplete(id_graphs, labels):
    """
    Construit la matrice binaire complète (présence des motifs dans chaque graphe).

    Parameters
    ----------
    id_graphs : dict[int, list[int]]
        Dictionnaire où chaque clé est un motif, et chaque valeur une liste d’ID de graphes où il apparaît.
    labels : list or array
        Liste des labels des graphes (utilisé uniquement pour déterminer leur nombre).

    Returns
    -------
    rep_binary : ndarray [n_graphs x n_patterns]
        Matrice binaire indiquant si un motif est présent dans un graphe.
    """
    nb_graphs = len(labels)
    nb_patterns = len(id_graphs)

    rep_binary = np.zeros((nb_graphs, nb_patterns), dtype=np.uint8)

    pattern_ids = range(nb_patterns)
    pattern_index = {p: idx for idx, p in enumerate(pattern_ids)}

    for p in pattern_ids:
        k = pattern_index[p]
        for j in id_graphs[p]:
            if 0 <= j < nb_graphs:
                rep_binary[j, k] = 1

    return rep_binary

def partialRepresentation(X, patterns):
    """
    Extrait une sous-matrice avec uniquement les motifs sélectionnés.

    Parameters
    ----------
    X : ndarray [n_graphs x n_patterns]
    patterns : list[int]
        Indices des colonnes à conserver.

    Returns
    -------
    X_partial : ndarray [n_graphs x len(patterns)]
    """
    return X[:, np.array(patterns)]



import numpy as np

def select_top_k_columns(X, scores, k):
    """
    Sélectionne les k colonnes (motifs) avec les meilleurs scores.

    Parameters
    ----------
    X : ndarray [n_graphs x n_motifs]
        Matrice complète
    scores : list[float]
        Score de chaque motif
    k : int
        Nombre de motifs à conserver

    Returns
    -------
    X_k : ndarray [n_graphs x k]
        Matrice restreinte
    selected_indices : list[int]
        Indices des colonnes sélectionnées
    """
    scores = np.array(scores)
    if k >= X.shape[1]:
        selected_indices = list(range(X.shape[1]))
    else:
        selected_indices = np.argsort(scores)[::-1][:k]  # tri décroissant

    X_k = X[:, selected_indices]
    return X_k, selected_indices.tolist()
