"""Métriques communes à tous les modèles (réseaux et modèles classiques).

Ce que fait ce fichier (toujours sur les VRAIES volatilités, après inverse_transform) :
- erreur moyenne absolue en volatilité implicite, en points de base (1 pb = 0,0001) : métrique principale ;
- erreur maximale absolue, en points de base : métrique complémentaire ;
- erreur moyenne par point de la grille (8, 11), pour une carte de chaleur des erreurs ;
- agrégation de plusieurs graines : moyenne et écart-type.
Les points NaN de y_true (pas de volatilité fiable) sont ignorés : nanmean et nanmax sautent les NaN.
"""

import numpy as np

BP = 1e-4  # 1 point de base = 0,0001 de volatilité


def mean_abs_error_bp(y_true, y_pred):
    """Erreur moyenne absolue, en points de base (note principale)."""
    errors = np.abs(np.asarray(y_pred) - np.asarray(y_true))  # écart de chaque point (NaN si pas de valeur)
    return float(np.nanmean(errors) / BP)                     # moyenne sur les points connus, en pb


def max_abs_error_bp(y_true, y_pred):
    """Erreur maximale absolue, en points de base (pire cas)."""
    errors = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    return float(np.nanmax(errors) / BP)                      # pire erreur parmi les points connus


def error_by_grid_point(y_true, y_pred, n_maturities=8, n_strikes=11):
    """Erreur moyenne en pb pour chacun des 88 points, renvoyée sous forme (8, 11)."""
    errors = np.abs(np.asarray(y_pred) - np.asarray(y_true))  # forme (N surfaces, 88)
    by_point = np.nanmean(errors, axis=0) / BP                # moyenne sur les surfaces connues : forme (88,)
    return by_point.reshape(n_maturities, n_strikes)          # remise en grille : maturités × strikes


def aggregate_seeds(values):
    """(moyenne, écart-type) d'une métrique sur les graines, ex. les 5 erreurs moyennes d'une config."""
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0  # écart-type « échantillon »
    return float(values.mean()), std