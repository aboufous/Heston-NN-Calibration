"""Modèles de machine learning classique pour l'étape 3 (comparaison avec le deep learning).

Tâche : 5 paramètres -> 88 volatilités. Les hyperparamètres viennent de configs/etape3/classical.yaml
et sont choisis sur la VALIDATION. Les surfaces avec des NaN sont à retirer avant l'entraînement
(voir le notebook) ; l'erreur est toujours calculée sur les points valides seulement.

Fonctions :
- `build_classical_model(name, hyperparams)` : Renvoie un modèle scikit-learn prêt à être entraîné.
- `tune_on_validation(name, grid, train, val)` : Essaie chaque combinaison de la grille et renvoie
  (meilleur modèle, meilleurs hyperparamètres).
- `fit_gaussian_process(X, Y, max_train_samples)` : Processus gaussien sur un sous-échantillon.
- `error_bp(y_true, y_pred)` : (erreur moyenne, erreur max) en pb, sans les NaN.
"""

import itertools

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from heston_nn.training.metrics import max_abs_error_bp, mean_abs_error_bp


def build_classical_model(name, hyperparams):
    """Renvoie un modèle scikit-learn. Les entrées X doivent déjà être standardisées."""
    if name == "ridge_poly":
        return make_pipeline(PolynomialFeatures(hyperparams["degree"]),
                             Ridge(alpha=hyperparams["alpha"]))
    if name == "knn":
        return KNeighborsRegressor(n_neighbors=hyperparams["n_neighbors"], weights="distance")
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=hyperparams["n_estimators"],
                                     max_depth=hyperparams["max_depth"],
                                     n_jobs=-1, random_state=0)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        one_output = LGBMRegressor(n_estimators=hyperparams["n_estimators"],
                                   learning_rate=hyperparams["learning_rate"],
                                   n_jobs=1, verbose=-1, random_state=0)
        return MultiOutputRegressor(one_output, n_jobs=-1)   # un modèle par volatilité (88)
    raise ValueError(f"Modèle inconnu : {name}")


def error_bp(y_true, y_pred):
    """(erreur moyenne, erreur max) en points de base, sur les points où y_true n'est pas NaN."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = np.isfinite(y_true)
    return mean_abs_error_bp(y_true[valid], y_pred[valid]), max_abs_error_bp(y_true[valid], y_pred[valid])


def tune_on_validation(name, grid, train, val):
    """Essaie toutes les combinaisons de la grille ; garde celle qui a la plus petite erreur sur la validation.

    grid  : dictionnaire {hyperparamètre: liste de valeurs}
    train : (X, Y) d'entraînement      val : (X, Y) de validation
    """
    X_train, Y_train = train
    X_val, Y_val = val

    best_model, best_params, best_error = None, None, np.inf
    names = list(grid.keys())
    for values in itertools.product(*grid.values()):
        params = dict(zip(names, values))
        model = build_classical_model(name, params)
        model.fit(X_train, Y_train)
        error, _ = error_bp(Y_val, model.predict(X_val))
        print(f"  {params} -> {error:.1f} pb")
        if error < best_error:
            best_model, best_params, best_error = model, params, error
    return best_model, best_params


def fit_gaussian_process(X, Y, max_train_samples):
    """Processus gaussien sur un sous-échantillon (trop lourd sur le jeu complet).

    On règle d'abord le noyau sur 1 000 points, puis on l'utilise tel quel sur max_train_samples points.
    """
    X, Y = X[:max_train_samples], Y[:max_train_samples]
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(1e-5)

    small = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    small.fit(X[:1000], Y[:1000])

    final = GaussianProcessRegressor(kernel=small.kernel_, optimizer=None, normalize_y=True)
    final.fit(X, Y)
    return final
