"""Figures du rapport, partagées entre les notebooks pour garder le même style partout.

- show_results : comme dans le TP3, courbes d'apprentissage de plusieurs configs sur le même graphique,
  moyenne sur les graines + bande ± écart-type, entraînement en trait plein et validation en pointillés,
  une couleur par config ; échelle logarithmique pour la perte (régression) ;
- plot_error_vs_time : étape 1, erreur moyenne (± écart-type sur les graines) en fonction du temps de
  réponse pour une surface, avec le temps de Fourier et le seuil de vitesse en lignes verticales ;
- plot_surface : une surface 8 × 11, un smile par maturité ;
- plot_error_heatmap : erreur par point de la grille (8 × 11), utilisée aussi pour le réseau.

Chaque figure peut être sauvegardée dans results/figures/ pour le rapport.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from heston_nn.experiments import agregate_result


def show_results(results, metric_name="loss", n_std=1, log_scale=True, title=None, save_path=None):
    """Courbes d'apprentissage moyenne ± écart-type, une couleur par config (comme dans le TP3).

    results     : liste results (experiments.run_configs ou experiments.collect_results).
    metric_name : "loss" (MSE) ou "mae".
    n_std       : largeur de la bande, en écarts-types (le TP3 utilise 3).
    save_path   : chemin du fichier image, ex. repo_path("results", "figures", "etape1.png").
    """
    network_types = sorted({r["type"] for r in results})   # les configs présentes, dans un ordre fixe
    colors = plt.get_cmap("tab10")                          # 10 couleurs bien distinctes
    figure, axis = plt.subplots(figsize=(11, 6))

    for index, network_type in enumerate(network_types):
        color = colors(index % 10)
        training, validation = agregate_result(results, network_type, metric_name)  # graines × époques
        epochs = np.arange(1, training.shape[1] + 1)

        # entraînement en trait plein, validation en pointillés
        for curves, style, label in ((training, "-", network_type), (validation, "--", None)):
            mean, std = curves.mean(axis=0), curves.std(axis=0)   # moyenne et dispersion sur les graines
            lower, upper = mean - n_std * std, mean + n_std * std
            if log_scale:
                lower = np.maximum(lower, mean / 10)   # en échelle log, la bande ne peut pas descendre sous 0
            axis.plot(epochs, mean, ls=style, lw=2, color=color, label=label)
            axis.fill_between(epochs, lower, upper, color=color, alpha=0.1)

    if log_scale:
        axis.set_yscale("log")

    # légende des styles de trait
    axis.plot([], [], "k-", label="entraînement")
    axis.plot([], [], "k--", label="validation")

    axis.set_xlabel("Époque")
    axis.set_ylabel(metric_name)
    axis.set_title(title or f"{metric_name} au fil de l'entraînement (moyenne ± {n_std} écart-type)")
    axis.legend()
    axis.grid(True, alpha=0.3)

    if save_path is not None:                                  # sauvegarde pour le rapport
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return figure

def plot_error_vs_time(table, t_fourier, threshold=None, save_path=None):
    """Étape 1 : erreur de validation en fonction du temps pour une surface, une étiquette par architecture.

    table     : une ligne par config, colonnes val_mean_bp, val_mean_bp_std et time_ms.
    t_fourier : temps de Fourier pour une surface, en secondes.
    threshold : gain minimal demandé par rapport à Fourier (None tant qu'il n'est pas fixé).
    """
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.errorbar(table["time_ms"], table["val_mean_bp"], yerr=table["val_mean_bp_std"],
                  fmt="o", capsize=4, color="tab:blue")
    for name, row in table.iterrows():                         # nom court à côté de chaque point
        axis.annotate(name.split("_")[-1], (row["time_ms"], row["val_mean_bp"]),
                      textcoords="offset points", xytext=(6, 4))

    axis.axvline(t_fourier * 1000, color="gray", ls=":", label=f"Fourier ({t_fourier * 1000:.2f} ms)")
    if threshold is not None:                                  # temps maximal autorisé par le seuil
        axis.axvline(t_fourier * 1000 / threshold, color="tab:red", ls="--",
                     label=f"seuil : {threshold:g} fois plus rapide que Fourier")

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Temps pour une surface (ms)")
    axis.set_ylabel("Erreur moyenne de validation (pb)")
    axis.set_title("Étape 1 : précision et vitesse des architectures")
    axis.legend()
    axis.grid(True, which="both", alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return figure


def plot_surface(surface, maturities, moneyness, title=None, axis=None, save_path=None):
    """Une surface 8 × 11 : un smile (vol en % selon le strike) par maturité, du clair (court) au foncé (long).

    Les NaN (options trop loin de la monnaie) ne sont simplement pas tracés.
    axis : axe matplotlib existant (pour mettre plusieurs surfaces côte à côte) ; sinon une nouvelle figure.
    """
    if axis is None:
        figure, axis = plt.subplots(figsize=(7, 4.5))
    else:
        figure = axis.figure
    colors = plt.cm.viridis(np.linspace(0.9, 0.1, len(maturities)))
    for row, T, color in zip(np.asarray(surface), maturities, colors):
        axis.plot(moneyness, 100 * row, "o-", color=color, markersize=3, label=f"T = {T:g}")
    axis.set_xlabel("Strike K (moneyness)")
    axis.set_ylabel("Volatilité implicite (%)")
    if title:
        axis.set_title(title)
    axis.legend(fontsize=7, ncol=2)
    axis.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return figure


def plot_error_heatmap(errors, maturities, moneyness, title=None, unit="pb", save_path=None):
    """Erreur par point de la grille (tableau 8 × 11, déjà agrégée : moyenne, max…), valeur écrite dans chaque case."""
    errors = np.asarray(errors, dtype=float)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    image = axis.imshow(errors, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(moneyness)), [f"{k:g}" for k in moneyness])
    axis.set_yticks(range(len(maturities)), [f"{t:g}" for t in maturities])
    axis.set_xlabel("Strike K (moneyness)")
    axis.set_ylabel("Maturité T (années)")
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            if np.isfinite(errors[i, j]):
                axis.text(j, i, f"{errors[i, j]:.2g}", ha="center", va="center", fontsize=7, color="white")
    figure.colorbar(image, ax=axis, label=unit)
    if title:
        axis.set_title(title)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return figure
