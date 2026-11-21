"""Tests de data/dataset.py."""

import numpy as np
import pytest

from heston_nn.config import load_config
from heston_nn.data.dataset import check_dataset, load_dataset

CFG = load_config("configs/etape1/arch_4x30.yaml")


def fake_dataset(n=100):
    """Petit jeu valide : paramètres dans leurs plages, volatilités à 20 %."""
    rng = np.random.default_rng(0)
    ranges = [CFG["param_ranges"][k] for k in ("v0", "theta", "kappa", "xi", "rho")]
    low, high = np.array(ranges).T                  # plages lues dans base.yaml
    X = low + rng.uniform(size=(n, 5)) * (high - low)
    Y = np.full((n, 88), 0.20)
    return X, Y


def test_jeu_valide_et_nan_comptes():
    X, Y = fake_dataset()
    Y[0, 0] = Y[0, 10] = Y[5, 0] = np.nan          # 3 points NaN sur 2 surfaces
    summary = check_dataset(X, Y, CFG)             # les NaN de Y sont acceptés
    assert summary["n_nan_points"] == 3
    assert summary["n_surfaces_with_nan"] == 2


def test_sauvegarde_puis_chargement(tmp_path):
    X, Y = fake_dataset()
    np.savez(tmp_path / "surfaces.npz", X=X, Y=Y)
    X2, Y2 = load_dataset(tmp_path / "surfaces.npz")
    assert np.array_equal(X, X2) and np.array_equal(Y, Y2)


def test_mauvaise_forme_refusee():
    X, Y = fake_dataset()
    with pytest.raises(ValueError):
        check_dataset(X[:, :4], Y, CFG)            # une colonne en moins


def test_parametre_hors_plage_refuse():
    X, Y = fake_dataset()
    X[0, 4] = 0.5                                  # rho positif, hors de sa plage
    with pytest.raises(ValueError):
        check_dataset(X, Y, CFG)


def test_vrai_jeu_de_donnees():
    X, Y = load_dataset("data/surfaces.npz")
    summary = check_dataset(X, Y, CFG)
    assert X.shape == (50000, 5) and Y.shape == (50000, 88)
    # ≈ 1,9 % des points manquent (seuil de prix 1e-10, vols basses), presque tous à T = 0,1 sur les strikes
    # extrêmes : on tolère 3 % au total, mais moins de 1 % hors de T = 0,1
    assert summary["share_nan_points"] < 0.03
    assert np.isnan(Y[:, 11:]).mean() < 0.01