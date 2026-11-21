"""Mesure des temps de réponse des modèles Python (réseaux, modèles classiques).

Le temps de Fourier est mesuré en Julia (julia/benchmark.jl) et lu ici depuis
results/etape0/fourier_time.json.

Ce que fait ce fichier :
- mesurer le temps pour UNE surface (un jeu de 5 paramètres), sur CPU, sur la même machine
  que la mesure Julia ; quelques appels « à vide », puis plusieurs centaines d'appels, MÉDIANE ;
- pour les réseaux Keras : appeler model(x, training=False) sur un tableau (1, 5), et PAS model.predict,
  qui ajoute un temps fixe important pour une seule surface et fausserait la comparaison ;
- calculer le gain : temps_Fourier / temps_modèle ;
- appliquer le seuil de vitesse (speed_threshold dans base.yaml).

Fonctions :
- `measure_time(fn, x, n_repeats, n_warmup)` : CONTRAT : temps médian (secondes) d'un appel fn(x).
- `load_fourier_time(path)` : Temps médian de Fourier pour une surface, mesuré en Julia
  (par défaut results/etape0/fourier_time.json).
- `speed_gain(t_fourier, t_model)` : Gain de vitesse du modèle par rapport à Fourier.
- `passes_threshold(gain, threshold)` : True si le gain respecte le seuil ; True aussi tant que le seuil n'est pas fixé (None).
"""


import json
import time

import numpy as np

from heston_nn.config import repo_path


def measure_time(fn, x, n_repeats=300, n_warmup=10):
    """Temps médian (secondes) d'un appel fn(x), après quelques appels à vide."""
    for _ in range(n_warmup):
        fn(x)
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def load_fourier_time(path=None):
    """Temps médian (secondes) de Fourier pour une surface, mesuré en Julia (julia/benchmark.jl).

    path : fichier JSON écrit par Julia (par défaut results/etape0/fourier_time.json).
    """
    path = path if path is not None else repo_path("results", "etape0", "fourier_time.json")
    with open(path, encoding="utf-8") as f:
        fourier = json.load(f)
    return float(fourier["median_s"])


def speed_gain(t_fourier, t_model):
    """Gain de vitesse du modèle par rapport à Fourier : combien de fois plus rapide (> 1 = plus rapide)."""
    if t_model <= 0:
        raise ValueError(f"Temps du modèle invalide : {t_model} s (doit être > 0)")
    return t_fourier / t_model


def passes_threshold(gain, threshold):
    """True si le gain respecte le seuil ; True aussi tant que le seuil n'est pas fixé (None)."""
    if threshold is None:
        return True
    return gain >= threshold
