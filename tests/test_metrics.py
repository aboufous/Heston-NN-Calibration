"""Tests de training/metrics.py."""

import numpy as np

from heston_nn.training.metrics import (aggregate_seeds, error_by_grid_point,
                                        max_abs_error_bp, mean_abs_error_bp)

# 2 surfaces de 88 points, toutes à 20 % de volatilité
Y_TRUE = np.full((2, 88), 0.20)
Y_PRED = Y_TRUE.copy()
Y_PRED[0, 0] = 0.2010    # 1re surface, 1er point : erreur de 10 pb
Y_PRED[1, 87] = 0.1995   # 2e surface, dernier point : erreur de 5 pb


def test_erreur_max():
    assert np.isclose(max_abs_error_bp(Y_TRUE, Y_PRED), 10)


def test_erreur_moyenne():
    # (10 + 5) pb répartis sur 2 × 88 = 176 points
    assert np.isclose(mean_abs_error_bp(Y_TRUE, Y_PRED), 15 / 176)


def test_erreur_par_point_de_grille():
    grid = error_by_grid_point(Y_TRUE, Y_PRED)
    assert grid.shape == (8, 11)
    assert np.isclose(grid[0, 0], 5)      # 10 pb sur 1 des 2 surfaces : moyenne 5
    assert np.isclose(grid[7, 10], 2.5)   # 5 pb sur 1 des 2 surfaces : moyenne 2,5


def test_aggregation_des_graines():
    mean, std = aggregate_seeds([12, 14, 13, 11, 15])
    assert np.isclose(mean, 13)
    assert np.isclose(std, np.sqrt(2.5))



def test_les_nan_sont_ignores():
    y_true = Y_TRUE.copy()
    y_true[0, 5] = np.nan                                       # un point sans valeur
    assert np.isclose(max_abs_error_bp(y_true, Y_PRED), 10)
    assert np.isclose(mean_abs_error_bp(y_true, Y_PRED), 15 / 175)   # 176 - 1 points connus
    assert not np.isnan(error_by_grid_point(y_true, Y_PRED)).any()