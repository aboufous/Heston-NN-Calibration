"""Tests de data/split.py, sur de fausses données au bon format."""

import numpy as np

from heston_nn.config import load_config
from heston_nn.data.split import prepare_data, split_dataset, subsample

CFG = load_config("configs/etape1/arch_4x30.yaml")
RNG = np.random.default_rng(0)
X = RNG.uniform(size=(1000, 5))    # 1 000 fausses surfaces : 5 paramètres...
Y = RNG.uniform(size=(1000, 88))   # ... et 88 volatilités


def test_tailles_80_10_10():
    data = split_dataset(X, Y, CFG)
    assert data["train"][0].shape == (800, 5)
    assert data["val"][0].shape == (100, 5)
    assert data["test"][1].shape == (100, 88)


def test_meme_decoupage_a_chaque_fois():
    a = split_dataset(X, Y, CFG)
    b = split_dataset(X, Y, CFG)
    assert np.array_equal(a["test"][0], b["test"][0])


def test_standardisation_sur_entrainement():
    scaled, x_scaler, y_scaler = prepare_data(X, Y, CFG)
    X_train = scaled["train"][0]
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6)   # moyenne 0
    assert np.allclose(X_train.std(axis=0), 1, atol=1e-6)    # écart-type 1


def test_retour_a_l_echelle_d_origine():
    scaled, x_scaler, y_scaler = prepare_data(X, Y, CFG)
    raw = split_dataset(X, Y, CFG)
    assert np.allclose(y_scaler.inverse_transform(scaled["test"][1]), raw["test"][1])


def test_sous_echantillon():
    X_small, Y_small = subsample(X, Y, size=100, seed=0)
    assert X_small.shape == (100, 5) and Y_small.shape == (100, 88)