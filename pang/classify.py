import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import pandas as pd


def evaluate_classifier(X, y, cv=5, random_state=42, return_predictions=True):
    """
    Évalue un classifieur SVM linéaire via validation croisée.

    Parameters
    ----------
    X : ndarray [n_graphs x n_features]
        Matrice de représentation des graphes.
    y : array-like [n_graphs]
        Labels binaires (0 ou 1).
    cv : int
        Nombre de folds pour la cross-validation.
    random_state : int
        Pour reproductibilité.

    Returns
    -------
    mean_f1 : float
        F1-score moyen sur les folds.
    all_scores : list[float]
        F1-score pour chaque fold.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    f1_scores = []
    all_preds = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = np.array(y)[train_idx], np.array(y)[test_idx]

        clf = LinearSVC(max_iter=10000, dual=False)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        score = f1_score(y_test, y_pred)
        f1_scores.append(score)
        if return_predictions:
            for yt, yp in zip(y_test, y_pred):
                all_preds.append({"fold": i + 1, "y_true": yt, "y_pred": yp})

    if return_predictions:
        df_preds = pd.DataFrame(all_preds)
        return np.mean(f1_scores), f1_scores, df_preds
    else:
        return np.mean(f1_scores), f1_scores
