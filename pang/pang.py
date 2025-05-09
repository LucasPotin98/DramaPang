from pang.loading import readLabels, load_graphs, read_Sizegraph
from pang.vectorize import ComputeRepresentationComplete, select_top_k_columns
from pang.measures import get_measure_function
from pang.classify import evaluate_classifier
import pickle

def pang_load_and_represent(FILEGRAPHS, FILESUBGRAPHS, FILELABEL):
    """
    Charge les graphes, les motifs et les labels, puis construit la représentation binaire complète.

    Parameters
    ----------
    FILE_GRAPHS : str
        Fichier GSPAN contenant les graphes initiaux
    FILE_PATTERNS : str
        Fichier GSPAN contenant les motifs extraits
    FILE_LABELS : str
        Fichier contenant un label par ligne (0 ou 1)

    Returns
    -------
    X : ndarray
        Matrice binaire complète [n_graphs x n_motifs]
    Graphes : list[nx.Graph]
        Liste des graphes originaux
    labels : list[int]
        Labels binaires des graphes
    """

    TAILLEGRAPHE=read_Sizegraph(FILEGRAPHS)
    TAILLEPATTERN=read_Sizegraph(FILESUBGRAPHS)

    Graphes,xx,noms= load_graphs(FILEGRAPHS,TAILLEGRAPHE)
    xx,id_graphs,xx = load_graphs(FILESUBGRAPHS,TAILLEPATTERN)
    labels = readLabels(FILELABEL)
    

    X = ComputeRepresentationComplete(id_graphs, labels)
    return X, Graphes, labels, noms

def load_titles(FILETITLES):
    with open(FILETITLES, 'r', encoding='utf-8') as f:
        titles = f.readlines()
    return titles



def load_model(model_path):
    """
    Charge un modèle sérialisé depuis un fichier pickle.

    Args:
        model_path (str): Chemin vers le fichier .pkl contenant le modèle.

    Returns:
        sklearn.base.BaseEstimator: Modèle entraîné chargé.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def compute_scores(X, labels, measure="AbsSupDif"):
    """
    Calcule un score de qualité discriminante pour chaque motif (colonne de X).

    Parameters
    ----------
    X : ndarray [n_graphs x n_motifs]
        Matrice binaire complète
    labels : list or ndarray
        Labels binaires (0 ou 1)
    measure : str
        Nom de la mesure à appliquer (Sup, AbsSupDif, WRAcc)

    Returns
    -------
    scores : list[float]
        Score de chaque motif (colonne de X)
    """
    score_fn = get_measure_function(measure)
    scores = [score_fn(X[:, j], labels) for j in range(X.shape[1])]
    return scores


def pang_classify_with_selection(X_full, labels, scores, top_k=100, cv=5):
    """
    Sélectionne les top-k motifs, construit X_k, puis évalue le classifieur SVM.

    Parameters
    ----------
    X_full : ndarray
        Matrice complète [n_graphs x n_motifs]
    labels : list[int]
        Labels binaires
    scores : list[float]
        Score de chaque motif
    top_k : int
        Nombre de motifs à sélectionner
    cv : int
        Nombre de folds

    Returns
    -------
    mean_f1 : float
        Score F1 moyen sur les folds
    selected_indices : list[int]
        Indices des motifs sélectionnés
    """
    X_k, selected_indices = select_top_k_columns(X_full, scores, top_k)
    mean_f1, _ = evaluate_classifier(X_k, labels, cv=cv)
    return mean_f1, selected_indices