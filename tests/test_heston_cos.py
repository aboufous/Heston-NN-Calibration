"""Tests du pricer COS en Python (src/heston_nn/pricing/heston_cos.py), copie du pricer Julia."""

import numpy as np
import pytest

from heston_nn.config import load_config, repo_path
from heston_nn.data.grid import get_grid
from heston_nn.pricing import heston_cos as hc

CFG = load_config("configs/base.yaml")
TS, KS = get_grid(CFG)


def test_reference_fang_oosterlee():
    """Prix publié par Fang & Oosterlee (2008), même test que côté Julia."""
    price = hc.otm_prices([1.0], 1.0, [0.0175, 0.0398, 1.5768, 0.5751, -0.5711])[0]
    assert price * 100 == pytest.approx(5.785155435, abs=1e-7)


def test_inversion_black():
    for K in KS:
        for sigma in (0.05, 0.2, 0.6):
            price = hc.black_otm(K, 1.0, sigma)
            if price > 1e-10:
                assert hc.implied_vol([price], [K], 1.0)[0] == pytest.approx(sigma, abs=1e-8)


def test_limite_black_scholes():
    surface = hc.price_surface([0.04, 0.04, 2.0, 0.01, 0.0], TS, KS)
    assert np.sum(~np.isnan(surface)) >= 80
    assert np.nanmax(np.abs(surface - 0.2)) < 1e-3


@pytest.mark.parametrize("params, T, K, reference", [
    # références indépendantes (formule de Lewis, intégration numérique), mêmes cas que les tests Julia
    ([0.0332, 0.0197, 1.9677, 1.151, -0.9054], 1.2, 1.5, 0.082825),
    ([0.0191, 0.0726, 0.95, 1.326, -0.467], 1.2, 0.5, 0.351051),
    ([0.135, 0.1415, 1.0408, 1.3088, -0.9138], 2.0, 1.5, 0.133554),
])
def test_cas_difficiles(params, T, K, reference):
    assert hc.price_surface(params, [T], [K])[0, 0] == pytest.approx(reference, abs=1e-5)


def test_tirage_des_parametres():
    X = hc.draw_params(CFG, np.random.default_rng(0), 2000)
    assert X.shape == (2000, 5)
    for j, name in enumerate(hc.PARAM_NAMES):
        low, high = CFG["param_ranges"][name]
        assert np.all((X[:, j] >= low - 1e-12) & (X[:, j] <= high + 1e-12))
    assert np.all(2 * X[:, 2] * X[:, 1] / X[:, 3] ** 2 >= CFG["min_feller_ratio"])


def test_meme_resultat_que_julia():
    """Les 30 premières surfaces de data/surfaces.npz (écrites par Julia) : mêmes NaN, écart < 0,01 pb."""
    data = np.load(repo_path("data", "surfaces.npz"))
    X, Y = data["X"][:30], data["Y"][:30]
    Y_python = hc.generate_surfaces(X, TS, KS)
    assert np.array_equal(np.isnan(Y_python), np.isnan(Y))
    both = ~np.isnan(Y)
    assert np.max(np.abs(Y_python[both] - Y[both])) < 1e-6
