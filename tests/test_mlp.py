"""Tests de models/mlp.py."""

import numpy as np
import pytest

from heston_nn.config import load_config
from heston_nn.models.mlp import build_model, count_parameters, get_model

ACTIVATIONS = ["elu", "relu", "leaky_relu", "tanh", "sigmoid", "gelu", "silu"]
INITS = ["naive", "xavier", "he"]


def test_formes_entree_sortie():
    model = get_model(hidden_layers=4, width=30)
    x = np.random.rand(32, 5).astype("float32")           # 32 jeux de paramètres
    assert model(x, training=False).shape == (32, 88)     # 32 surfaces de 88 points


def test_nombre_de_parametres_reference():
    # calcul à la main : 30*(5+1) + 3*30*(30+1) + 88*(30+1) = 5 698
    assert count_parameters(get_model(hidden_layers=4, width=30)) == 5698


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("init", INITS)
def test_toutes_les_combinaisons_se_construisent(activation, init):
    model = get_model(hidden_layers=2, width=8, activation=activation, init=init)
    assert model(np.zeros((1, 5), dtype="float32")).shape == (1, 88)


def test_init_inconnue_refusee():
    with pytest.raises(ValueError):
        get_model(init="mauvaise")


def test_construction_depuis_une_config():
    cfg = load_config("configs/etape1/arch_2x32.yaml")
    model = build_model(cfg["model"])
    assert count_parameters(model) == 32 * 6 + 32 * 33 + 88 * 33

@pytest.mark.parametrize("activation", ACTIVATIONS + ["linear"])
def test_prediction_numpy_identique_a_keras(activation):
    from sklearn.preprocessing import StandardScaler
    from heston_nn.models.mlp import make_predict_fn
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    Y = rng.normal(size=(100, 88))
    x_scaler, y_scaler = StandardScaler().fit(X), StandardScaler().fit(Y)
    model = get_model(hidden_layers=3, width=16, activation=activation)
    predict = make_predict_fn(model, x_scaler, y_scaler)

    expected = y_scaler.inverse_transform(np.asarray(model(x_scaler.transform(X[:1]), training=False)))
    assert predict(X[0]).shape == (8, 11)
    assert np.allclose(predict(X[0]).ravel(), expected[0], atol=1e-5)   # même résultat que Keras


def test_sauvegarde_et_rechargement(tmp_path):
    from sklearn.preprocessing import StandardScaler
    from heston_nn.models.mlp import load_model, save_model
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(50, 5)), rng.normal(size=(50, 88))
    x_scaler, y_scaler = StandardScaler().fit(X), StandardScaler().fit(Y)
    model = get_model(hidden_layers=2, width=8)

    save_model(tmp_path / "reseau_final", model, x_scaler, y_scaler, {"model": {"width": 8}})
    model2, x_scaler2, y_scaler2, cfg = load_model(tmp_path / "reseau_final")

    x = x_scaler.transform(X[:3])
    assert np.allclose(model(x), model2(x))                                  # mêmes prédictions
    assert np.allclose(x_scaler2.transform(X[:3]), x)                        # même normalisation
    assert np.allclose(y_scaler2.inverse_transform(Y[:3]), y_scaler.inverse_transform(Y[:3]))
    assert cfg["model"]["width"] == 8
