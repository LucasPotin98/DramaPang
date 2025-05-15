import numpy as np


# === Support (fréquence brute du motif dans la classe cible) ===
def sup(p, y):
    """
    Support du motif p dans la classe 1.
    p : vecteur binaire d’occurrence du motif (shape: [n_graphs])
    y : vecteur binaire de labels (shape: [n_graphs], 0 ou 1)
    """
    y = np.array(y)
    subset = p[y == 1]
    if len(subset) == 0:
        return 0.0
    else:
        return np.mean(subset)


# === Différence absolue de supports entre les deux classes ===
def absSupDiff(p, y):
    """
    |Support classe 1 - Support classe 0|
    """
    y = np.array(y)
    sup1 = np.sum(p[y == 1]) / len(y)
    sup0 = np.sum(p[y == 0]) / len(y)

    return abs(sup1 - sup0)


# === WRAcc : Weighted Relative Accuracy ===
def wracc(p, y):
    """
    WRAcc(p) = freq(p) * (P(c=1 | p) - P(c=1))
    """
    y = np.array(y)
    freq_p = np.mean(p)
    if freq_p == 0:
        return 0.0

    P_c1 = np.mean(y)
    P_c1_given_p = np.sum((p == 1) & (y == 1)) / np.sum(p)
    return freq_p * (P_c1_given_p - P_c1)


# === Sélectionneur de mesure ===
def get_measure_function(name):
    """
    Retourne une fonction de score à partir de son nom.
    """
    name = name.lower()
    if name == "sup":
        return sup
    elif name == "abssupdif":
        return absSupDiff
    elif name == "wracc":
        return wracc
    else:
        raise ValueError(
            f"Mesure {name} non disponible publiquement. Choisir parmi : Sup, AbsSupDif, WRAcc"
        )
