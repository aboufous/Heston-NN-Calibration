"""Tests de data/market.py, hors ligne : une fausse chaîne d'options SPX générée par Heston (paramètres connus),
avec un taux, un dividende et une fourchette bid / ask, remplace Yahoo Finance."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from heston_nn.data import market
from heston_nn.pricing import heston_cos

PARAMS = np.array([0.02, 0.035, 2.0, 0.9, -0.7])     # v0, theta, kappa, xi, rho
SPOT, RATE, CARRY = 6000.0, 0.04, 0.03               # actualisation e^{-rT}, forward F = S e^{cT}
TRADE_DATE = dt.date(2026, 11, 20)
DAYS = [10, 38, 87, 150, 276, 395, 577, 759, 850]


def synthetic_chain(trade_date=TRADE_DATE, spot=SPOT):
    """Chaîne d'options au format de download_chain : calls et puts sur une grille de strikes, mid ± 1 %."""
    rows = []
    for days in DAYS:
        T = days / 365
        D, F = np.exp(-RATE * T), spot * np.exp(CARRY * T)
        K = np.arange(0.4, 1.8, 0.01) * spot
        otm = heston_cos.otm_prices(K / F, T, PARAMS) * F * D          # prix réels des options hors de la monnaie
        put = np.where(K < F, otm, otm - (F - K) * D)                   # parité call-put pour les autres
        call = np.where(K >= F, otm, otm + (F - K) * D)
        expiry = (trade_date + dt.timedelta(days)).isoformat()
        for kind, prices in (("call", call), ("put", put)):
            rows.append(pd.DataFrame({"strike": K, "bid": 0.99 * prices, "ask": 1.01 * prices,
                                      "volume": 10, "openInterest": 10, "type": kind, "expiry": expiry}))
    raw = pd.concat(rows, ignore_index=True)
    raw["spot"], raw["trade_date"] = spot, trade_date.isoformat()
    return raw


@pytest.fixture(scope="module")
def built():
    return market.build_surface(synthetic_chain())


def test_forward_et_actualisation(built):
    smiles = built[2]
    assert np.allclose(smiles["F"], SPOT * np.exp(CARRY * smiles["T"]), rtol=2e-3)
    assert np.allclose(smiles["D"], np.exp(-RATE * smiles["T"]), atol=5e-3)


def test_taux_du_jour(built):
    """Le taux utilisé pour toutes les échéances est celui de la chaîne synthétique (4 %)."""
    assert built[4]["rate"] == pytest.approx(RATE, abs=2e-3)


def test_taux_robuste_aux_echeances_aberrantes():
    """Des échéances au facteur d'actualisation absurde (taux de -36 % ou +150 %) ne changent pas le taux du jour."""
    estimates = [(0.05, np.exp(0.36 * 0.05)), (0.1, np.exp(-1.5 * 0.1)),            # courtes : ignorées
                 (0.5, np.exp(-0.04 * 0.5)), (1.0, np.exp(-0.041 * 1.0)), (2.0, np.exp(-0.039 * 2.0)),
                 (1.2, 0.835)]                                                       # une longue aberrante
    assert market.day_rate(estimates) == pytest.approx(0.0405, abs=1e-3)


def test_surface_proche_de_heston(built):
    """Sur les points cotés, la surface reconstruite suit la vraie surface de Heston (écart dû au lissage SVI)."""
    surface, mask, _, _, info = built
    true = heston_cos.price_surface(PARAMS, market.GRID_T, market.GRID_K)
    gap_bp = 1e4 * np.abs(surface - true)[mask == 1]
    assert surface.shape == mask.shape == (len(market.GRID_T), len(market.GRID_K))
    assert mask.mean() > 0.7                                   # la plupart de la grille est couverte
    assert np.mean(gap_bp) < 15 and np.max(gap_bp) < 60
    assert info["n_calendar_violations"] == 0


def test_fichier_de_calibration_plusieurs_jours(tmp_path):
    """Deux jours collectés : le fichier contient les deux, NaN là où le masque vaut 0, le dernier jour dans surface."""
    path = tmp_path / "spx_surface.npz"
    for day, spot in [(TRADE_DATE, SPOT), (TRADE_DATE + dt.timedelta(3), 1.01 * SPOT)]:
        raw_file = tmp_path / f"{day}_raw.csv"
        synthetic_chain(day, spot).to_csv(raw_file, index=False)
        market.collect(tmp_path / "spx", from_raw=raw_file, calibration_file=path, figures_dir=tmp_path / "figures")

    data = np.load(path)
    assert list(data["dates"]) == ["2026-11-20", "2026-11-23"]
    assert data["surfaces"].shape == (2, 8, 11)
    assert np.array_equal(np.isnan(data["surfaces"]), data["masks"] == 0)
    assert np.array_equal(data["surface"], data["surfaces"][-1], equal_nan=True)
    assert np.allclose(data["surfaces"][data["masks"] == 1], data["surfaces_full"][data["masks"] == 1])
    assert (tmp_path / "figures" / "2026-11-23_surface.png").exists()
    assert not (tmp_path / "spx" / "plots").exists()                # pas de figure dans les données


def test_zone_de_calibration():
    """La zone de base.yaml : strikes 0,7 à 1,3 (7) et maturités >= 0,3 (7), soit 49 points de la grille."""
    from heston_nn.config import load_config
    from heston_nn.data.grid import calibration_zone

    zone = calibration_zone(load_config("configs/base.yaml"))
    assert zone.shape == (8, 11) and zone.sum() == 49
    assert not zone[0].any()                                   # pas de T = 0,1
    assert not zone[:, [0, 1, 9, 10]].any()                    # pas de K = 0,5 ; 0,6 ; 1,4 ; 1,5
