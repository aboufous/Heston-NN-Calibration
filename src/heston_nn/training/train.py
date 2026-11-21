"""Entraînement commun à toutes les expériences (étapes 1, 2 et 3), en Keras (comme dans les TP).


Ce que fait ce fichier :
- entraînement accéléré (steps_per_execution, jit_compile pour les gros réseaux) : mêmes calculs, moins
  de frais fixes ; une barre de progression par entraînement (une case par époque) ;
- fixe toutes les graines (Python, numpy, TensorFlow) pour que chaque essai soit reproductible ;
- utilise une perte MSE MASQUÉE : les points NaN de Y (environ 0,4 % des points, options dont le prix est
  trop petit pour en tirer une volatilité) sont ignorés dans la perte et dans les métriques ;
- compile_train, sur le modèle du TP3 : construit le réseau, le compile (perte MSE) et l'entraîne avec
  model.fit pendant exactement `epochs` époques (budget fixe), en gardant les poids de la meilleure
  époque sur la validation ; renvoie le réseau et l'historique (DataFrame, comme dans le TP1) ;
- run_experiment : entraîne, évalue sur le test en points de base (après dénormalisation), et
  sauvegarde métriques (JSON), historique (CSV) et réseau (.keras) dans results/<name>/seed_<k>/.

CONTRAT : compile_train(model_cfg, train_cfg, seed, scaled) -> (modèle, historique).
"""

import json
import time

import keras
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from heston_nn.config import repo_path
from heston_nn.models.mlp import build_model, count_parameters
from heston_nn.training.metrics import max_abs_error_bp, mean_abs_error_bp
from heston_nn.training.optimizers import build_optimizer

# --- Perte et métrique MASQUÉES : on ignore les points NaN de Y (volatilité impossible à calculer) ---------
# Principe : là où y_true est NaN, on remplace la vraie valeur par la prédiction -> erreur nulle sur ce point,
# puis on divise par le nombre de points connus. Le décorateur permet de recharger le modèle .keras plus tard.

@keras.saving.register_keras_serializable(package="heston_nn")
def masked_mse(y_true, y_pred):
    """MSE sur les points connus uniquement (perte d'entraînement)."""
    known = ~keras.ops.isnan(y_true)                                   # True là où la valeur existe
    y_true = keras.ops.where(known, y_true, y_pred)                    # case vide -> erreur nulle
    n_known = keras.ops.sum(keras.ops.cast(known, "float32"), axis=-1)
    return keras.ops.sum(keras.ops.square(y_true - y_pred), axis=-1) / n_known


@keras.saving.register_keras_serializable(package="heston_nn")
def masked_mae(y_true, y_pred):
    """MAE sur les points connus uniquement (courbe suivie pendant l'entraînement)."""
    known = ~keras.ops.isnan(y_true)
    y_true = keras.ops.where(known, y_true, y_pred)
    n_known = keras.ops.sum(keras.ops.cast(known, "float32"), axis=-1)
    return keras.ops.sum(keras.ops.abs(y_true - y_pred), axis=-1) / n_known


class EpochBar(keras.callbacks.Callback):
    """Barre de progression : une case par époque, avec la perte de validation."""

    def __init__(self, epochs, desc):
        super().__init__()
        self.bar = tqdm(total=epochs, desc=desc, unit="époque")

    def on_epoch_end(self, epoch, logs=None):
        self.bar.update(1)
        self.bar.set_postfix(val_loss=f"{logs['val_loss']:.2e}")

    def on_train_end(self, logs=None):
        self.bar.close()


def set_seed(seed):
    """Fixe les graines de Python, numpy et TensorFlow en une fois (poids de départ, ordre des batchs)."""
    keras.utils.set_random_seed(seed)


def compile_train(model_cfg, train_cfg, seed, scaled, verbose=0, desc="entraînement"):
    """CONTRAT : construit, compile et entraîne le réseau ; renvoie (modèle, historique en DataFrame).

    scaled : données standardisées, sortie de data.split.prepare_data
             {"train": (X, Y), "val": (X, Y), "test": (X, Y)}.
    """
    set_seed(seed)                                   # même graine -> même réseau de départ, même ordre

    model = build_model(model_cfg)                   # le réseau, encore vide (mlp.py)
    optimizer = build_optimizer(train_cfg["optimizer"], train_cfg["lr"])   # un NOUVEL optimiseur
    # pertes qui ignorent les NaN ; deux réglages de vitesse qui ne changent pas les calculs :
    model.compile(loss=masked_mse, optimizer=optimizer, metrics=[masked_mae],
                  steps_per_execution=50,                  # enchaîne 50 pas d'un coup : moins de frais fixes
                  jit_compile=model_cfg["width"] >= 512)   # compile les calculs : utile pour les gros réseaux

    # Garder les poids de la meilleure époque sur la validation.
    # patience = epochs : on n'arrête jamais avant la fin (budget fixe), on restaure juste le meilleur.
    keep_best = keras.callbacks.EarlyStopping(monitor="val_loss",
                                              patience=train_cfg["epochs"],
                                              restore_best_weights=True)

    X_train, Y_train = scaled["train"]
    history = model.fit(X_train, Y_train,
                        validation_data=scaled["val"],
                        epochs=train_cfg["epochs"],
                        batch_size=train_cfg["batch_size"],
                        callbacks=[keep_best, EpochBar(train_cfg["epochs"], desc)],
                        verbose=verbose)

    return model, pd.DataFrame(history.history)      # colonnes : loss, val_loss, masked_mae, val_masked_mae


def evaluate_bp(model, X, Y, y_scaler):
    """Erreurs moyenne et max en points de base, sur les VRAIES volatilités (après dénormalisation)."""
    prediction = np.asarray(model(X, training=False))        # volatilités normalisées prédites
    y_pred = y_scaler.inverse_transform(prediction)          # -> vraies volatilités
    y_true = y_scaler.inverse_transform(Y)
    return mean_abs_error_bp(y_true, y_pred), max_abs_error_bp(y_true, y_pred)


def run_experiment(cfg, seed, prepared, results_root=None, verbose=0):
    """Entraîne une config avec une graine, l'évalue sur le test et sauvegarde tout ; renvoie les métriques.

    prepared : sortie complète de data.split.prepare_data, c'est-à-dire (scaled, x_scaler, y_scaler).
    results_root : dossier des résultats (par défaut results/ à la racine du dépôt).
    """
    scaled, x_scaler, y_scaler = prepared

    start = time.perf_counter()
    model, history = compile_train(cfg["model"], cfg["train"], seed, scaled, verbose=verbose,
                                   desc=f"{cfg['name']} – graine {seed}")
    training_time = time.perf_counter() - start

    val_mean, val_max = evaluate_bp(model, *scaled["val"], y_scaler)     # pour faire les choix
    test_mean, test_max = evaluate_bp(model, *scaled["test"], y_scaler)  # note finale

    metrics = {
        "name": cfg["name"],
        "seed": seed,
        "val_mean_bp": val_mean,
        "val_max_bp": val_max,
        "test_mean_bp": test_mean,
        "test_max_bp": test_max,
        "best_epoch": int(history["val_loss"].idxmin()) + 1,
        "n_params": count_parameters(model),
        "training_time_s": training_time,
    }

    # Sauvegarde : results/<name>/seed_<k>/
    out_dir = (results_root if results_root is not None else repo_path("results")) / cfg["name"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    history.to_csv(out_dir / "history.csv", index_label="epoch")
    model.save(out_dir / "model.keras")

    return metrics