"""Lancer des séries d'expériences et rassembler leurs résultats, depuis les notebooks.

Organisation reprise du TP3 : chaque entraînement donne un dictionnaire
{"type": nom de la config, "seed": graine, "history": DataFrame, "metrics": dict}, rangé dans une
liste `results` ; on agrège ensuite les graines pour tracer moyenne ± écart-type.

Ce que fait ce fichier :
- my_share : répartit une liste d'expériences entre les 4 machines, pour partager les calculs ;
- run_configs : lance une liste de (config, graine), en sautant celles déjà faites (résultats présents
  dans results/), et renvoie la liste results ;
- collect_results : relit tous les résultats d'une étape dans results/ (calculés sur toutes les
  machines, une fois copiés au même endroit) et reconstruit la liste results ;
- agregate_result : comme dans le TP3, deux matrices (graines × époques) pour une config et une
  métrique : une pour l'entraînement, une pour la validation ;
- summarize : tableau des notes finales, moyenne ± écart-type par config, pour le rapport.
"""

import json

import pandas as pd
from tqdm.auto import tqdm

from heston_nn.config import load_config, repo_path
from heston_nn.training.train import run_experiment


def my_share(jobs, machine_id, n_machines=4):
    """Part des expériences pour cette machine : une sur quatre, en commençant à machine_id."""
    return jobs[machine_id::n_machines]


def _results_root(results_root):
    """Dossier des résultats : results/ à la racine du dépôt, sauf si on en précise un autre."""
    return results_root if results_root is not None else repo_path("results")


def load_result(run_dir):
    """Relit une expérience terminée (dossier results/<name>/seed_<k>/) au format de la liste results."""
    with open(run_dir / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    history = pd.read_csv(run_dir / "history.csv", index_col="epoch")
    return {"type": metrics["name"], "seed": metrics["seed"], "history": history, "metrics": metrics}


def run_configs(jobs, prepared, results_root=None, verbose=0):
    """Lance chaque (chemin de config, graine) de jobs ; saute ceux déjà faits ; renvoie la liste results."""
    root = _results_root(results_root)
    results = []
    for config_path, seed in tqdm(jobs, desc="Entraînements", unit="run"):
        cfg = load_config(config_path)
        run_dir = root / cfg["name"] / f"seed_{seed}"
        if (run_dir / "metrics.json").exists():
            print(f"{cfg['name']} – graine {seed} : déjà fait, on relit le résultat")
        else:
            print(f"{cfg['name']} – graine {seed} : entraînement…")
            run_experiment(cfg, seed, prepared, results_root=root, verbose=verbose)
        results.append(load_result(run_dir))
    return results


def collect_results(etape, results_root=None):
    """Liste results de tous les entraînements d'une étape (dossiers results/etape<k>_*/seed_*/)."""
    root = _results_root(results_root)
    results = []
    for run_dir in sorted(root.glob(f"etape{etape}_*/seed_*")):
        if (run_dir / "metrics.json").exists():          # on ignore une expérience inachevée
            results.append(load_result(run_dir))
    return results


def agregate_result(results, network_type, metric_name="loss"):
    """Comme dans le TP3 : matrices (graines × époques) d'entraînement et de validation pour une config."""
    histories = [r["history"] for r in results if r["type"] == network_type]
    training = pd.concat([h[metric_name] for h in histories], axis=1).to_numpy().T
    validation = pd.concat([h[f"val_{metric_name}"] for h in histories], axis=1).to_numpy().T
    return training, validation


def summarize(results):
    """Tableau des notes finales par config : moyenne et écart-type sur les graines, trié par validation."""
    table = pd.DataFrame([r["metrics"] for r in results])
    summary = table.groupby("name").agg(
        n_seeds=("seed", "count"),
        val_mean_bp=("val_mean_bp", "mean"),
        val_mean_bp_std=("val_mean_bp", "std"),
        test_mean_bp=("test_mean_bp", "mean"),
        test_mean_bp_std=("test_mean_bp", "std"),
        test_max_bp=("test_max_bp", "mean"),
        n_params=("n_params", "first"),
        training_time_s=("training_time_s", "mean"),
    )
    return summary.sort_values("val_mean_bp")           # le meilleur sur la VALIDATION en premier