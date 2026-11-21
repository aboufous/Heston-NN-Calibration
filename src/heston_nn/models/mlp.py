"""Réseau de neurones : perceptron multicouche (MLP) configurable, en TensorFlow / Keras (comme en cours).

Ce que fait ce fichier :
- get_model : construit un keras.Sequential, comme dans le TP1 :
  Input(shape=(5,)) -> `hidden_layers` couches Dense de `width` neurones -> Dense(88) sans activation
  (sortie linéaire : on prédit des nombres libres, c'est une régression) ;
- activation configurable (série C) : elu (référence, Horvath et al.), relu, leaky_relu, tanh, sigmoid,
  gelu, silu ;
- initialisation configurable (série B), via kernel_initializer :
  naive -> RandomNormal(stddev=0.01) ; xavier -> "glorot_normal" ; he -> "he_normal" ;
- compte les paramètres, pour expliquer les différences de vitesse à l'étape 1.

- make_predict_fn : CONTRAT : renvoie predict(params (5,)) -> surface (8, 11), en numpy pur.
  Pour UNE surface, Keras est plus lent que Fourier (≈ 3 ms de frais fixes par appel) : on recopie les poids
  et on refait le calcul couche par couche avec numpy (h = activation(h @ W + b)), environ 100 fois plus vite.
  Sert à mesurer les temps (étape 1) et à la calibration (étape 4).

- save_model / load_model : enregistre et recharge le réseau final dans un dossier (ex.
  results/reseau_final/) : reseau.keras (réseau + poids) et infos.json (les deux normalisations et la
  config). Usage : model, x_scaler, y_scaler, cfg = load_model(...) puis make_predict_fn(...).
"""

import json
from pathlib import Path

import keras
import numpy as np
from scipy.special import erf
from sklearn.preprocessing import StandardScaler

N_INPUTS = 5     # les 5 paramètres de Heston : v0, theta, kappa, xi, rho
N_OUTPUTS = 88   # la surface : 8 maturités × 11 strikes


def get_initializer(init):
    """Renvoie le kernel_initializer Keras correspondant à naive, xavier ou he (série B)."""
    if init == "naive":
        return keras.initializers.RandomNormal(mean=0.0, stddev=0.01)  # petits poids au hasard
    if init == "xavier":
        return "glorot_normal"   # Xavier / Glorot : pensée pour tanh et sigmoïde
    if init == "he":
        return "he_normal"       # He : pensée pour ReLU et ses variantes
    raise ValueError(f"Initialisation inconnue : {init!r} (attendu : naive, xavier ou he)")


def get_model(hidden_layers=4, width=30, activation="elu", init="he"):
    """Renvoie le keras.Sequential 5 -> ... -> 88, non compilé (la compilation se fait dans train.py)."""
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(N_INPUTS,)))   # entrée : 5 paramètres

    for _ in range(hidden_layers):                      # couches cachées, toutes identiques
        model.add(keras.layers.Dense(width,
                                     activation=activation,
                                     kernel_initializer=get_initializer(init)))

    # sortie : 88 volatilités, SANS activation (régression)
    model.add(keras.layers.Dense(N_OUTPUTS, kernel_initializer=get_initializer(init)))
    return model


def build_model(model_cfg):
    """Construit le réseau à partir de la section model de la config (ex. cfg["model"])."""
    return get_model(hidden_layers=model_cfg["hidden_layers"],
                     width=model_cfg["width"],
                     activation=model_cfg["activation"],
                     init=model_cfg["init"])


def count_parameters(model):
    """Nombre de paramètres (poids + biais) du réseau."""
    return model.count_params()
    

# Les activations de Keras, réécrites en numpy (mêmes formules, mêmes réglages par défaut)
NUMPY_ACTIVATIONS = {
    "elu": lambda z: np.where(z > 0, z, np.expm1(np.minimum(z, 0))),   # exp(z) - 1 si z < 0
    "relu": lambda z: np.maximum(z, 0),
    "leaky_relu": lambda z: np.where(z > 0, z, 0.2 * z),                # pente 0,2 : défaut de Keras
    "tanh": np.tanh,
    "sigmoid": lambda z: 1 / (1 + np.exp(-z)),
    "gelu": lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2))),
    "silu": lambda z: z / (1 + np.exp(-z)),
    "linear": lambda z: z,                                              # couche de sortie, témoin linéaire
}


def make_predict_fn(model, x_scaler, y_scaler):
    """CONTRAT : renvoie predict(params (5,)) -> surface (8, 11), en numpy pur (rapide pour UNE surface)."""
    # on recopie une fois pour toutes les poids, biais et activations de chaque couche Dense
    layers = []
    for layer in model.layers:
        W, b = layer.get_weights()
        layers.append((W, b, NUMPY_ACTIVATIONS[layer.activation.__name__]))

    def predict(params):
        h = (np.asarray(params, dtype=float) - x_scaler.mean_) / x_scaler.scale_   # même normalisation
        for W, b, activation in layers:
            h = activation(h @ W + b)                                             # une couche
        surface = h * y_scaler.scale_ + y_scaler.mean_                            # vraies volatilités
        return surface.reshape(8, 11)

    return predict


def save_model(path, model, x_scaler, y_scaler, cfg):
    """Enregistre le réseau final dans un dossier : reseau.keras (réseau + poids) et infos.json
    (les deux normalisations et la config)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save(path / "reseau.keras")
    infos = {
        "x_mean": x_scaler.mean_.tolist(), "x_scale": x_scaler.scale_.tolist(),   # normalisation des entrées
        "y_mean": y_scaler.mean_.tolist(), "y_scale": y_scaler.scale_.tolist(),   # normalisation des sorties
        "cfg": cfg,                                                               # architecture, activation...
    }
    with open(path / "infos.json", "w", encoding="utf-8") as f:
        json.dump(infos, f, indent=2, default=str)


def _rebuild_scaler(mean, scale):
    """Reconstruit un StandardScaler à partir de sa moyenne et de son écart-type."""
    scaler = StandardScaler()
    scaler.mean_ = np.array(mean)
    scaler.scale_ = np.array(scale)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(mean)
    return scaler


def load_model(path):
    """Recharge le réseau final : renvoie (model, x_scaler, y_scaler, cfg)."""
    path = Path(path)
    model = keras.models.load_model(path / "reseau.keras", compile=False)   # pas besoin de la perte pour prédire
    with open(path / "infos.json", encoding="utf-8") as f:
        infos = json.load(f)
    x_scaler = _rebuild_scaler(infos["x_mean"], infos["x_scale"])
    y_scaler = _rebuild_scaler(infos["y_mean"], infos["y_scale"])
    return model, x_scaler, y_scaler, infos["cfg"]
