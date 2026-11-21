"""Découpage et standardisation des données, avec scikit-learn (comme dans le TP1).

Ce que fait ce fichier :
- découpe X, Y en entraînement / validation / test (80 / 10 / 10) avec train_test_split, avec
  random_state = split.seed de base.yaml : TOUJOURS le même découpage pour tout le monde ;
- standardise avec StandardScaler : fit_transform sur l'entraînement, transform sur la validation et le
  test. Un scaler pour les entrées (les 5 paramètres) et un pour les sorties (les 88 volatilités) ;
  on garde les deux scalers pour dénormaliser les prédictions (inverse_transform) ;
- prend un sous-échantillon de l'entraînement (courbe d'apprentissage, étape 3).
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_dataset(X, Y, cfg):
    """Découpe en entraînement / validation / test ; renvoie {"train": (X, Y), "val": (X, Y), "test": (X, Y)}."""
    seed = cfg["split"]["seed"]                            # 42 : même découpage pour tout le monde
    share_rest = cfg["split"]["val"] + cfg["split"]["test"]  # 0.1 + 0.1 = 0.2

    # 1er découpage : 80 % entraînement, 20 % mis de côté
    X_train, X_rest, Y_train, Y_rest = train_test_split(X, Y, test_size=share_rest, random_state=seed)

    # 2e découpage : les 20 % mis de côté, coupés en validation (10 %) et test (10 %)
    share_test = cfg["split"]["test"] / share_rest           # 0.1 / 0.2 = 0.5
    X_val, X_test, Y_val, Y_test = train_test_split(X_rest, Y_rest, test_size=share_test, random_state=seed)

    return {"train": (X_train, Y_train), "val": (X_val, Y_val), "test": (X_test, Y_test)}


def prepare_data(X, Y, cfg):
    """Découpe puis standardise ; renvoie (données standardisées, x_scaler, y_scaler)."""
    data = split_dataset(X, Y, cfg)

    x_scaler = StandardScaler()   # pour les 5 paramètres (échelles très différentes)
    y_scaler = StandardScaler()   # pour les 88 volatilités

    # Les moyennes et écarts-types sont calculés sur l'ENTRAÎNEMENT uniquement (fit_transform)
    X_train, Y_train = data["train"]
    scaled = {"train": (x_scaler.fit_transform(X_train), y_scaler.fit_transform(Y_train))}

    # ... puis appliqués tels quels à la validation et au test (transform), sans fuite d'information
    for name in ("val", "test"):
        X_part, Y_part = data[name]
        scaled[name] = (x_scaler.transform(X_part), y_scaler.transform(Y_part))

    return scaled, x_scaler, y_scaler


def subsample(X, Y, size, seed):
    """Sous-échantillon aléatoire de taille size (pour la courbe d'apprentissage de l'étape 3)."""
    if size >= len(X):
        return X, Y                                          # on demande tout : on renvoie tout
    rng = np.random.default_rng(seed)                        # tirage reproductible
    index = rng.choice(len(X), size=size, replace=False)     # size lignes différentes, au hasard
    return X[index], Y[index]
    