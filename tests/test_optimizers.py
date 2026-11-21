"""Tests de training/optimizers.py."""

import numpy as np
import pytest

from heston_nn.training.optimizers import OPTIMIZERS, build_optimizer


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_chaque_optimiseur_se_construit(name):
    optimizer = build_optimizer(name, learning_rate=1e-3)
    assert np.isclose(float(optimizer.learning_rate.numpy()), 1e-3)


def test_momentum_vaut_0_9():
    assert np.isclose(build_optimizer("momentum", 1e-3).momentum, 0.9)


def test_un_nouvel_optimiseur_a_chaque_appel():
    assert build_optimizer("adam", 1e-3) is not build_optimizer("adam", 1e-3)


def test_nom_inconnu_refuse():
    with pytest.raises(ValueError):
        build_optimizer("mauvais", 1e-3)