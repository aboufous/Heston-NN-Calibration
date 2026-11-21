"""Chargement et vérification du jeu de données produit par Julia (data/surfaces.npz).


Ce que fait ce fichier :
- charge X (N, 5) et Y (N, 88) depuis le fichier .npz écrit par julia/generate_data.jl ;
- VÉRIFIE le contrat : formes, paramètres dans les plages de base.yaml, volatilités plausibles
  (entre 0,01 et 2) ; lève une erreur claire sinon ;
- COMPTE les NaN de Y sans les refuser : ce sont les options dont le prix est trop petit pour en tirer une
  volatilité fiable (surtout T = 0,1 avec K = 0,5 ou 1,5). Ils sont ignorés dans la perte et les métriques.
"""

import numpy as np

PARAM_NAMES = ["v0", "theta", "kappa", "xi", "rho"]  # ordre des colonnes de X (contrat avec Julia)
N_PARAMS = 5
N_POINTS = 88                  # 8 maturités × 11 strikes
VOL_MIN, VOL_MAX = 0.01, 2.0   # volatilités plausibles : entre 1 % et 200 %


def load_dataset(path):
    """Charge X (N, 5) et Y (N, 88) depuis un fichier .npz."""
    with np.load(path) as data:
        return data["X"], data["Y"]


def check_dataset(X, Y, cfg):
    """Vérifie le contrat (erreur explicite sinon) ; renvoie un résumé, dont le nombre de NaN."""
    problems = []

    # 1. Les formes : si elles sont fausses, inutile d'aller plus loin
    if X.ndim != 2 or X.shape[1] != N_PARAMS:
        problems.append(f"X doit être de forme (N, 5), reçu {X.shape}")
    if Y.ndim != 2 or Y.shape[1] != N_POINTS:
        problems.append(f"Y doit être de forme (N, 88), reçu {Y.shape}")
    if len(X) != len(Y):
        problems.append(f"X et Y n'ont pas le même nombre de lignes : {len(X)} et {len(Y)}")
    if problems:
        raise ValueError("Jeu de données invalide :\n- " + "\n- ".join(problems))

    # 2. Les paramètres ne doivent jamais manquer, et doivent être dans leurs plages (configs/base.yaml)
    if np.isnan(X).any():
        problems.append("X (les paramètres) contient des NaN")
    for j, name in enumerate(PARAM_NAMES):
        low, high = cfg["param_ranges"][name]
        if np.nanmin(X[:, j]) < low - 1e-9 or np.nanmax(X[:, j]) > high + 1e-9:
            problems.append(f"{name} sort de sa plage [{low}, {high}] : "
                            f"min {np.nanmin(X[:, j]):.4g}, max {np.nanmax(X[:, j]):.4g}")

    # 3. Des volatilités plausibles (les NaN sont ignorés ici)
    if np.nanmin(Y) < VOL_MIN or np.nanmax(Y) > VOL_MAX:
        problems.append(f"volatilités hors de [{VOL_MIN}, {VOL_MAX}] : "
                        f"min {np.nanmin(Y):.4g}, max {np.nanmax(Y):.4g}")

    if problems:
        raise ValueError("Jeu de données invalide :\n- " + "\n- ".join(problems))

    # 4. Les NaN de Y sont acceptés : on les compte pour le rapport
    nan_points = np.isnan(Y)
    return {
        "n_surfaces": len(Y),
        "n_surfaces_with_nan": int(nan_points.any(axis=1).sum()),
        "n_nan_points": int(nan_points.sum()),
        "share_nan_points": float(nan_points.mean()),
    }