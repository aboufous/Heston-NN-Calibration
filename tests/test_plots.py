"""Tests de plots.py : les figures se construisent sans erreur."""

import matplotlib

matplotlib.use("Agg")   # pas de fenêtre pendant les tests

import numpy as np
import pandas as pd

from heston_nn.plots import plot_error_vs_time, show_results


def test_show_results():
    history = pd.DataFrame({"loss": [1.0, 0.5], "val_loss": [1.2, 0.6]})
    results = [{"type": "a", "history": history}, {"type": "a", "history": history * 1.1}]
    show_results(results)


def test_plot_error_vs_time(tmp_path):
    table = pd.DataFrame({"val_mean_bp": [5.0, 2.0], "val_mean_bp_std": [0.5, 0.2],
                          "time_ms": [0.03, 0.3]}, index=["etape1_arch_4x30", "etape1_arch_6x512"])
    plot_error_vs_time(table, t_fourier=1.2e-3, threshold=10, save_path=tmp_path / "fig.png")
    assert (tmp_path / "fig.png").exists()
