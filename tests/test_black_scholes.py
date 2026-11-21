"""Tests de Black-76 côté Python (src/heston_nn/pricing/black_scholes.py)."""

import numpy as np
import pytest
from scipy.stats import norm

from heston_nn.pricing.black_scholes import black76, implied_vol


def test_prix_connu():
    """À la monnaie (F = K = 1), le call vaut 2 N(σ√T / 2) - 1."""
    assert black76(1.0, 1.0, 1.0, 0.2, True) == pytest.approx(2 * norm.cdf(0.1) - 1, abs=1e-14)


def test_parite_call_put():
    """C - P = F - K, quel que soit σ."""
    for F, K, T, sigma in [(1.0, 0.8, 0.5, 0.3), (5000.0, 5200.0, 0.1, 0.15), (1.0, 1.4, 2.0, 0.6)]:
        call, put = black76(F, K, T, sigma, True), black76(F, K, T, sigma, False)
        assert call - put == pytest.approx(F - K, rel=1e-12, abs=1e-12)


def test_vectorise():
    K = np.array([0.8, 1.0, 1.2])
    prices = black76(1.0, K, 0.5, 0.2, K >= 1)
    assert prices.shape == (3,)
    for k, price in zip(K, prices):
        assert price == pytest.approx(float(black76(1.0, k, 0.5, 0.2, k >= 1)))


@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_aller_retour(is_call):
    """implied_vol(black76(σ)) redonne σ, pour des forwards de l'ordre du SPX comme pour F = 1."""
    for F in (1.0, 5000.0):
        for moneyness in (0.7, 0.9, 1.0, 1.1, 1.3):
            for T in (0.1, 1.0, 2.0):
                for sigma in (0.08, 0.2, 0.5):
                    K = moneyness * F
                    price = float(black76(F, K, T, sigma, is_call))
                    intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
                    if price - intrinsic > 1e-10 * F:      # sinon le prix ne contient plus d'information sur σ
                        assert implied_vol(price, F, K, T, is_call) == pytest.approx(sigma, abs=1e-6)


def test_nan_hors_des_bornes_d_arbitrage():
    F, K, T = 1.0, 0.9, 1.0
    assert np.isnan(implied_vol(0.05, F, K, T, True))      # call sous sa valeur intrinsèque (0,1)
    assert np.isnan(implied_vol(1.5, F, K, T, True))       # call plus cher que le forward
    assert np.isnan(implied_vol(-0.01, F, K, T, False))    # prix négatif
    assert np.isnan(implied_vol(0.95, F, K, T, False))     # put plus cher que le strike


def test_une_seule_version_dans_le_projet():
    """market.py et heston_cos.py utilisent ce module, pas une copie."""
    from heston_nn.data import market
    from heston_nn.pricing import heston_cos

    assert market.implied_vol is implied_vol
    assert heston_cos.black76 is black76
