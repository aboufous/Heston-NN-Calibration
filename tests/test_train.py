"""Tests de training/train.py, sur de fausses données qui ont quelque chose à apprendre."""

import json

import numpy as np

from heston_nn.config import deep_merge, load_config
from heston_nn.data.split import prepare_data
from heston_nn.training.train import compile_train, run_experiment

# Fausses données : Y dépend vraiment de X (sinon le réseau n'aurait rien à apprendre)
RNG = np.random.default_rng(0)
X = RNG.uniform(size=(1000, 5))
W = RNG.normal(size=(5, 88))
Y = np.sin(X @ W) + 0.01 * RNG.normal(size=(1000, 88))

# Config de référence, en petit pour que le test soit rapide
CFG = deep_merge(load_config("configs/etape1/arch_4x30.yaml"),
                 {"name": "test_train", "train": {"epochs": 20, "batch_size": 64}})
PREPARED = prepare_data(X, Y, CFG)
SCALED = PREPARED[0]


def test_l_erreur_de_validation_baisse():
    model, history = compile_train(CFG["model"], CFG["train"], seed=0, scaled=SCALED)
    assert len(history) == 20                                        # budget fixe : 20 époques
    assert history["val_loss"].min() < history["val_loss"].iloc[0]  # il apprend


def test_meme_graine_meme_resultat():
    _, h1 = compile_train(CFG["model"], CFG["train"], seed=1, scaled=SCALED)
    _, h2 = compile_train(CFG["model"], CFG["train"], seed=1, scaled=SCALED)
    assert np.allclose(h1["val_loss"], h2["val_loss"])


def test_run_experiment_sauvegarde_tout(tmp_path):
    metrics = run_experiment(CFG, seed=0, prepared=PREPARED, results_root=tmp_path)
    out = tmp_path / "test_train" / "seed_0"
    assert (out / "metrics.json").exists()
    assert (out / "history.csv").exists()
    assert (out / "model.keras").exists()
    assert metrics["test_mean_bp"] > 0
    assert json.loads((out / "metrics.json").read_text())["seed"] == 0
    