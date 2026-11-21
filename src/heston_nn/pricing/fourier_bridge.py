"""Pont vers le pricer Fourier (utilise la copie fidèle Python heston_cos.py)."""

from heston_nn.pricing.heston_cos import price_surface as _cos_price_surface


def price_surface(params, maturities, moneyness):
    """CONTRAT : params (5,) -> surface (8, 11) de volatilités implicites."""
    return _cos_price_surface(params, maturities, moneyness)
