"""Calibration : trouver les 5 paramètres de Heston qui reproduisent au mieux une surface.

Ce que ce fichier fait :
- minimise l'écart entre la surface du modèle et la surface cible avec Levenberg–Marquardt
  (scipy.optimize.least_squares) ; le MÊME optimiseur pour Fourier et pour le réseau ;
- accepte n'importe quelle fonction de surface : pricing.fourier_bridge.price_surface (Julia)
  OU la fonction predict du réseau (models.mlp.make_predict_fn) — c'est le CONTRAT :
  surface_fn(params (5,)) -> surface (8, 11) (ou (88,), ligne par ligne) ;
- gère les bornes des paramètres (param_ranges) et ignore les points NaN de la surface cible ;
- part de plusieurs points de départ et garde le meilleur résultat ;
- renvoie les paramètres, l'erreur d'ajustement (pb), le temps de calibration et le nombre d'appels.

Conventions :
- ordre des paramètres : (v0, theta, kappa, xi, rho), le même que dans configs/base.yaml ;
- les volatilités implicites sont en décimal (0.20 = 20 %) ; les erreurs sont rapportées en
  points de base (1 pb = 1e-4 de volatilité) ;
- une surface (8, 11) est aplatie ligne par ligne (maturité par maturité), comme
  data.grid.surface_to_vector : np.ravel() donne exactement le même ordre.

Bornes et Levenberg–Marquardt :
  la méthode "lm" de scipy (MINPACK) ne gère pas les bornes. Pour garder LM ET respecter les
  bornes, on optimise sur une variable libre z avec  x = lo + (hi - lo) * sigmoid(z).
  Les paramètres renvoyés sont donc toujours dans [lo, hi]. La méthode "trf" (bornes natives)
  reste disponible avec method="trf" pour comparer.

Équité Fourier / réseau :
  à seed identique, les points de départ sont identiques pour les deux modèles (ils ne dépendent
  ni de surface_fn ni du résultat). Utiliser la même cible, les mêmes bornes, le même x0,
  le même n_starts et le même seed pour les deux appels.

Temps :
  le premier appel de Julia est lent (compilation) : faire un appel « à vide » avant de
  chronométrer. time_s couvre TOUS les points de départ (c'est le temps d'une calibration
  complète, celui à utiliser dans break_even).

Exemple :
    from functools import partial
    from heston_nn.pricing.fourier_bridge import price_surface
    fn_fourier = partial(price_surface, maturities=mats, moneyness=money)
    params, infos = calibrate(market_surface, fn_fourier, bounds=cfg["param_ranges"],
                              n_starts=5, seed=0)

Fonctions :
- `residuals(params, target_surface, surface_fn)` : écarts (n_valides,) entre surface_fn(params)
  et la surface cible, sans les points NaN de la cible.
- `calibrate(target_surface, surface_fn, bounds, x0, n_starts, seed)` : CONTRAT : renvoie
  (params (5,), infos). infos : {"fit_error_bp", "time_s", "n_evaluations", "success", ...}.
- `break_even(t_generation, t_training, t_calib_fourier, t_calib_network)` : seuil de rentabilité :
  n* = (T_génération + T_entraînement) / (t_calib_Fourier - t_calib_réseau).
"""

from __future__ import annotations

import math
import time
from typing import Callable, Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit
from scipy.stats import qmc

PARAM_NAMES = ("v0", "theta", "kappa", "xi", "rho")
N_PARAMS = len(PARAM_NAMES)

BP = 1e4  # 1 point de base = 1e-4 de volatilité : volatilité (décimal) -> pb
NAN_PENALTY = 1.0  # résidu imposé si le modèle renvoie NaN/inf là où la cible est valide (1.0 = 10 000 pb)
_EDGE = 1e-6  # marge (en fraction de la boîte) pour ne pas démarrer exactement sur une borne

SurfaceFn = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------------------
# Outils internes
# --------------------------------------------------------------------------------------
def _parse_bounds(bounds) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (lo, hi), deux tableaux (5,), à partir de formats usuels.

    Formats acceptés :
    - dict {"v0": [lo, hi], ..., "rho": [lo, hi]} (clés dans l'ordre PARAM_NAMES si elles y sont toutes,
      sinon dans l'ordre du dictionnaire) ;
    - tuple (lo, hi) avec lo et hi de taille 5 ;
    - tableau (5, 2) de paires [lo, hi].
    """
    if isinstance(bounds, Mapping):
        if all(name in bounds for name in PARAM_NAMES):
            pairs = [bounds[name] for name in PARAM_NAMES]
        else:
            pairs = list(bounds.values())
        arr = np.asarray(pairs, dtype=float)
    elif isinstance(bounds, tuple) and len(bounds) == 2:
        arr = np.stack([np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)], axis=1)
    else:
        arr = np.asarray(bounds, dtype=float)
        if arr.shape == (2, N_PARAMS):
            arr = arr.T

    if arr.shape != (N_PARAMS, 2):
        raise ValueError(f"bounds doit décrire {N_PARAMS} paramètres [lo, hi] ; forme obtenue : {arr.shape}.")
    lo, hi = arr[:, 0].copy(), arr[:, 1].copy()
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("Les bornes doivent être finies.")
    if not np.all(lo < hi):
        raise ValueError("Chaque borne inférieure doit être strictement inférieure à la borne supérieure.")
    return lo, hi


def _clip_inside(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Ramène x dans [lo, hi], avec une petite marge par rapport aux bornes."""
    margin = _EDGE * (hi - lo)
    return np.clip(x, lo + margin, hi - margin)


def _to_free(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Paramètres bornés -> variable libre z (inverse de _to_bounded)."""
    return logit((_clip_inside(x, lo, hi) - lo) / (hi - lo))


def _to_bounded(z: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Variable libre z -> paramètres dans [lo, hi]."""
    return lo + (hi - lo) * expit(z)


def _starting_points(lo, hi, x0, n_starts: int, seed) -> np.ndarray:
    """Points de départ : x0 (s'il est donné) puis un hypercube latin dans la boîte des bornes.

    Ne dépend ni du modèle ni de la cible : mêmes points pour Fourier et pour le réseau.
    """
    if n_starts < 1:
        raise ValueError("n_starts doit être >= 1.")
    starts = []
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).ravel()
        if x0.shape != (N_PARAMS,):
            raise ValueError(f"x0 doit avoir {N_PARAMS} éléments ; forme obtenue : {x0.shape}.")
        starts.append(_clip_inside(x0, lo, hi))
    n_random = n_starts - len(starts)
    if n_random > 0:
        sampler = qmc.LatinHypercube(d=N_PARAMS, seed=seed)
        unit = sampler.random(n_random)
        starts.extend(_clip_inside(lo + unit * (hi - lo), lo, hi))
    return np.array(starts)


# --------------------------------------------------------------------------------------
# Fonctions du contrat
# --------------------------------------------------------------------------------------
def residuals(params, target_surface, surface_fn: SurfaceFn) -> np.ndarray:
    """Écarts (n_valides,) entre surface_fn(params) et la surface cible, sans les points NaN.

    - target_surface : (8, 11) ou (88,) ; les points non finis (NaN) sont ignorés ;
    - surface_fn(params) doit renvoyer le même nombre de points que la cible ;
    - écart = modèle - cible, en volatilité (décimal) ;
    - si le modèle renvoie NaN/inf là où la cible est valide, l'écart vaut NAN_PENALTY
      (au lieu de NaN, qui ferait échouer l'optimiseur).
    """
    target = np.asarray(target_surface, dtype=float).ravel()
    mask = np.isfinite(target)
    model = np.asarray(surface_fn(np.asarray(params, dtype=float)), dtype=float).ravel()
    if model.shape != target.shape:
        raise ValueError(
            f"surface_fn a renvoyé {model.size} points, la surface cible en a {target.size}."
        )
    diff = model[mask] - target[mask]
    bad = ~np.isfinite(diff)
    if bad.any():
        diff = diff.copy()
        diff[bad] = NAN_PENALTY
    return diff


def calibrate(
    target_surface,
    surface_fn: SurfaceFn,
    bounds,
    x0=None,
    n_starts: int = 5,
    seed: int = 0,
    method: str = "lm",
    max_nfev: int | None = None,
    tol: float = 1e-8,
):
    """Calibre les 5 paramètres de Heston sur target_surface.

    Arguments :
    - target_surface : surface de marché (8, 11) ou (88,), avec des NaN possibles ;
    - surface_fn     : params (5,) -> surface (8, 11) ou (88,) ; Fourier OU réseau ;
    - bounds         : bornes des 5 paramètres (voir _parse_bounds), typiquement param_ranges ;
    - x0             : point de départ optionnel (compte pour l'un des n_starts) ;
    - n_starts       : nombre total de points de départ (x0 inclus) ;
    - seed           : graine des points de départ aléatoires (hypercube latin) ;
    - method         : "lm" (Levenberg–Marquardt, défaut, bornes par changement de variable)
                       ou "trf" (bornes natives) ;
    - max_nfev       : budget d'évaluations par point de départ (None = défaut de scipy) ;
    - tol            : tolérance (xtol, ftol, gtol) de l'optimiseur.

    Renvoie (params (5,), infos) avec infos :
    - fit_error_bp   : erreur absolue moyenne sur les points valides, en pb ;
    - time_s         : temps total (tous les points de départ), en secondes ;
    - n_evaluations  : nombre d'appels à surface_fn (différences finies du jacobien comprises) ;
    - success        : l'optimiseur a convergé pour le meilleur point de départ ;
    et, en plus : rmse_bp, max_error_bp, n_points, n_starts, best_start, method,
    start_errors_bp (erreur de chaque départ, pour voir la dispersion), start_params
    (paramètres trouvés par départ, pour juger la stabilité), errors (échecs éventuels).
    """
    if method not in ("lm", "trf"):
        raise ValueError("method doit valoir 'lm' ou 'trf'.")

    lo, hi = _parse_bounds(bounds)
    target = np.asarray(target_surface, dtype=float).ravel()
    n_valid = int(np.isfinite(target).sum())
    if n_valid < N_PARAMS:
        raise ValueError(
            f"Pas assez de points valides dans la surface cible : {n_valid} < {N_PARAMS} paramètres."
        )

    counter = {"n": 0}

    def counted_fn(p):
        counter["n"] += 1
        return surface_fn(p)

    t_start = time.perf_counter()
    starts = _starting_points(lo, hi, x0, n_starts, seed)

    runs = []  # un dictionnaire par départ réussi (sans exception)
    errors = []
    ls_options = dict(xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev)

    for i, start in enumerate(starts):
        try:
            if method == "lm":

                def fun(z):
                    return residuals(_to_bounded(z, lo, hi), target, counted_fn)

                res = least_squares(fun, _to_free(start, lo, hi), method="lm", **ls_options)
                x = _to_bounded(res.x, lo, hi)
            else:

                def fun(p):
                    return residuals(p, target, counted_fn)

                res = least_squares(fun, start, bounds=(lo, hi), method="trf", **ls_options)
                x = np.clip(res.x, lo, hi)
        except ValueError:
            raise  # erreur de forme, de bornes... : un bug, pas un échec numérique
        except Exception as exc:  # p. ex. erreur côté Julia pour des paramètres extrêmes
            errors.append(f"départ {i} : {type(exc).__name__}: {exc}")
            continue

        fit = np.asarray(res.fun, dtype=float)
        runs.append(
            {
                "start": i,
                "x": x,
                "fun": fit,
                "sse": float(np.sum(fit**2)),
                "success": bool(res.success),
            }
        )

    time_s = time.perf_counter() - t_start

    if not runs:
        raise RuntimeError("Toutes les calibrations ont échoué : " + " | ".join(errors))

    best = min(runs, key=lambda r: r["sse"])
    abs_bp = np.abs(best["fun"]) * BP

    infos = {
        "fit_error_bp": float(abs_bp.mean()),
        "time_s": float(time_s),
        "n_evaluations": int(counter["n"]),
        "success": bool(best["success"]),
        "rmse_bp": float(np.sqrt(np.mean(best["fun"] ** 2)) * BP),
        "max_error_bp": float(abs_bp.max()),
        "n_points": n_valid,
        "n_starts": int(len(starts)),
        "best_start": int(best["start"]),
        "method": method,
        "start_errors_bp": [float(np.mean(np.abs(r["fun"])) * BP) for r in runs],
        "start_params": np.array([r["x"] for r in runs]),
        "errors": errors,
    }
    return best["x"], infos


def break_even(t_generation: float, t_training: float, t_calib_fourier: float, t_calib_network: float) -> float:
    """Seuil de rentabilité : nombre de calibrations à partir duquel le réseau est rentable.

    n* = (T_génération + T_entraînement) / (t_calib_Fourier - t_calib_réseau)

    Tous les temps sont en secondes, et t_calib_* est le temps d'UNE calibration complète
    (infos["time_s"], même protocole des deux côtés). Renvoie math.inf si le réseau n'est pas
    plus rapide que Fourier (jamais rentable). Arrondir au supérieur avec math.ceil pour
    obtenir un nombre entier de calibrations.
    """
    for name, value in (
        ("t_generation", t_generation),
        ("t_training", t_training),
        ("t_calib_fourier", t_calib_fourier),
        ("t_calib_network", t_calib_network),
    ):
        if value < 0:
            raise ValueError(f"{name} doit être positif (reçu : {value}).")
    gain_per_calibration = t_calib_fourier - t_calib_network
    if gain_per_calibration <= 0:
        return math.inf
    return (t_generation + t_training) / gain_per_calibration
