"""Tests de calibration.calibrate sur un modèle jouet (sans Julia ni réseau).

Le modèle jouet est une fonction lisse de 5 paramètres vers une surface (8, 11) : il sert à
vérifier la mécanique (bornes, NaN, points de départ, compteurs, seuil de rentabilité), pas Heston.
"""

import math

import numpy as np
import pytest

from heston_nn.calibration.calibrate import break_even, calibrate, residuals

MATS = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
LOGK = np.log(np.linspace(0.5, 1.5, 11))

LO = np.array([0.01, 0.01, 0.5, 0.1, -0.95])
HI = np.array([0.16, 0.16, 5.0, 1.0, -0.10])
TRUE = np.array([0.04, 0.06, 2.0, 0.5, -0.7])


def toy_surface(p):
    v0, theta, kappa, xi, rho = p
    T = MATS[:, None]
    k = LOGK[None, :]
    w = (1.0 - np.exp(-kappa * T)) / (kappa * T)
    atm = np.sqrt(theta + (v0 - theta) * w)
    skew = 0.3 * rho * xi * k / np.sqrt(T + 0.2)
    smile = 0.2 * xi**2 * k**2 / (T + 0.2)
    return atm + skew + smile


def test_residuals_ignorent_les_nan():
    target = toy_surface(TRUE)
    target[0, :3] = np.nan
    res = residuals(TRUE, target, toy_surface)
    assert res.shape == (88 - 3,)
    assert np.allclose(res, 0.0)


def test_residuals_penalise_un_modele_nan():
    target = toy_surface(TRUE)

    def bad_fn(p):
        out = toy_surface(p)
        out[0, 0] = np.nan
        return out

    res = residuals(TRUE, target, bad_fn)
    assert np.all(np.isfinite(res))
    assert res.max() > 0.5


def test_residuals_mauvaise_forme():
    with pytest.raises(ValueError):
        residuals(TRUE, np.zeros((8, 11)), lambda p: np.zeros(10))


@pytest.mark.parametrize("method", ["lm", "trf"])
def test_calibrate_retrouve_la_surface(method):
    target = toy_surface(TRUE)
    params, infos = calibrate(target, toy_surface, (LO, HI), n_starts=6, seed=1, method=method)
    assert params.shape == (5,)
    assert np.all(params >= LO) and np.all(params <= HI)
    assert infos["fit_error_bp"] < 1.0
    assert infos["n_evaluations"] > 0
    assert infos["time_s"] > 0
    assert infos["n_points"] == 88
    for key in ("fit_error_bp", "time_s", "n_evaluations", "success"):
        assert key in infos


def test_calibrate_avec_nan_et_bruit():
    rng = np.random.default_rng(0)
    target = toy_surface(TRUE) + rng.normal(0, 5e-4, size=(8, 11))  # ~5 pb de bruit
    target[0, [0, 1, 9, 10]] = np.nan  # strikes extrêmes à maturité courte
    params, infos = calibrate(target, toy_surface, (LO, HI), n_starts=6, seed=2)
    assert infos["n_points"] == 84
    assert infos["fit_error_bp"] < 15.0
    assert np.all(params >= LO) and np.all(params <= HI)


def test_bornes_respectees_quand_l_optimum_est_hors_boite():
    # surface générée avec kappa = 8, hors de la boîte [0.5, 5] : kappa doit rester <= 5
    outside = np.array([0.04, 0.06, 8.0, 0.5, -0.7])
    params, _ = calibrate(toy_surface(outside), toy_surface, (LO, HI), n_starts=4, seed=3)
    assert np.all(params >= LO) and np.all(params <= HI)


def test_bornes_en_dictionnaire():
    bounds = {
        "v0": [LO[0], HI[0]],
        "theta": [LO[1], HI[1]],
        "kappa": [LO[2], HI[2]],
        "xi": [LO[3], HI[3]],
        "rho": [LO[4], HI[4]],
    }
    params, infos = calibrate(toy_surface(TRUE), toy_surface, bounds, n_starts=3, seed=0)
    assert infos["fit_error_bp"] < 1.0


def test_memes_points_de_depart_pour_deux_modeles():
    # deux appels avec la même graine explorent les mêmes points de départ (équité Fourier / réseau)
    seen_a, seen_b = [], []

    def fn_a(p):
        seen_a.append(np.array(p))
        return toy_surface(p)

    def fn_b(p):
        seen_b.append(np.array(p))
        return toy_surface(p) * 1.0001

    target = toy_surface(TRUE)
    _, ia = calibrate(target, fn_a, (LO, HI), n_starts=3, seed=7, max_nfev=1)
    _, ib = calibrate(target, fn_b, (LO, HI), n_starts=3, seed=7, max_nfev=1)
    assert np.allclose(seen_a[0], seen_b[0])
    assert ia["n_starts"] == ib["n_starts"] == 3


def test_x0_compte_comme_un_depart():
    _, infos = calibrate(toy_surface(TRUE), toy_surface, (LO, HI), x0=TRUE, n_starts=1)
    assert infos["n_starts"] == 1
    assert infos["fit_error_bp"] < 1e-3


def test_pas_assez_de_points():
    target = np.full((8, 11), np.nan)
    target[0, :3] = 0.2
    with pytest.raises(ValueError):
        calibrate(target, toy_surface, (LO, HI))


def test_echec_total_leve_une_erreur():
    def boom(p):
        raise RuntimeError("erreur Julia")

    with pytest.raises(RuntimeError):
        calibrate(toy_surface(TRUE), boom, (LO, HI), n_starts=2)


def test_break_even():
    assert break_even(100.0, 400.0, 3.0, 0.5) == pytest.approx(200.0)
    assert break_even(100.0, 400.0, 0.5, 0.5) == math.inf
    assert break_even(100.0, 400.0, 0.2, 0.5) == math.inf
    with pytest.raises(ValueError):
        break_even(-1.0, 0.0, 1.0, 0.1)
