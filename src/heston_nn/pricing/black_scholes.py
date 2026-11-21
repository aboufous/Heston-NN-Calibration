"""Formules de Black-76 (prix en forward, taux nul) et volatilité implicite : LA version Python du projet.

Utilisée par :
- data/market.py : convertir les prix SPX en volatilités implicites ;
- pricing/heston_cos.py : prix Black-76 des options hors de la monnaie (F = 1), pour inverser les prix de Heston.

Conventions : prix exprimés en forward (prix actualisé / facteur d'actualisation), F = forward, K = strike,
T en années, sigma en décimal (0,20 = 20 %).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr          # fonction de répartition de la loi normale (rapide, vectorisée)

VOL_LO, VOL_HI = 1e-4, 5.0              # intervalle de recherche de la volatilité implicite


def black76(F, K, T, sigma, is_call):
    """Prix Black-76 d'un call (is_call vrai) ou d'un put. Vectorisé : les arguments peuvent être des tableaux."""
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    return np.where(is_call, call, put)


def implied_vol(price, F, K, T, is_call):
    """Volatilité implicite d'une option (scalaire), par la méthode de Brent.

    NaN si le prix est hors des bornes d'arbitrage (valeur intrinsèque < prix < F pour un call, < K pour un put)
    ou si aucune volatilité de [VOL_LO, VOL_HI] ne le reproduit.
    """
    if is_call:
        intrinsic, upper = max(F - K, 0.0), F
    else:
        intrinsic, upper = max(K - F, 0.0), K
    if not intrinsic < price < upper:
        return np.nan

    def error(sigma):
        return float(black76(F, K, T, sigma, is_call)) - price

    try:
        return brentq(error, VOL_LO, VOL_HI)
    except ValueError:                  # pas de changement de signe : prix non atteignable
        return np.nan
