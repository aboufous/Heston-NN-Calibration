"""Lecture des fichiers de configuration.

Ce que fait ce fichier :
- lit configs/base.yaml (règles communes) ;
- lit une config d'expérience (ex. configs/etape1/arch_2x32.yaml) ;
- fusionne les deux : la config d'expérience remplace seulement les clés qu'elle définit ;
- renvoie un dictionnaire complet, utilisé par les notebooks et experiments.py ;
- fournit repo_path(...) : un chemin depuis la racine du dépôt, pour que les notebooks (lancés depuis
  notebooks/) et Julia trouvent les mêmes fichiers (configs/, data/, results/).

Exemple :
    cfg = load_config("configs/etape1/arch_2x32.yaml")
    cfg["model"]["width"]       # 32  (vient de la config d'expérience)
    cfg["train"]["epochs"]      # 200 (vient de base.yaml)
"""

import copy
from pathlib import Path

import yaml

# Racine du dépôt : config.py est dans src/heston_nn/, donc on remonte de deux dossiers.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Fichier des règles communes.
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def repo_path(*parts):
    """Chemin absolu depuis la racine du dépôt, ex. repo_path("data", "surfaces.npz")."""
    return REPO_ROOT.joinpath(*parts)


def load_yaml(path):
    """Lit un fichier YAML et renvoie son contenu sous forme de dictionnaire."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}  # fichier vide -> dictionnaire vide


def deep_merge(base, override):
    """Fusionne override dans base, groupe par groupe, sans modifier base.

    Une clé présente dans override remplace celle de base.
    Si les deux valeurs sont des dictionnaires (ex. le groupe "model"), on les fusionne à leur tour,
    pour ne pas perdre les clés de base que override ne précise pas.
    """
    merged = copy.deepcopy(base)  # copie : base.yaml reste intact
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)  # même groupe des deux côtés : on descend
        else:
            merged[key] = copy.deepcopy(value)  # sinon : la valeur de l'expérience remplace
    return merged


def load_config(path, base_path=BASE_CONFIG):
    """Renvoie la config complète : base.yaml fusionnée avec la config d'expérience."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path  # chemin relatif -> depuis la racine du dépôt
    base = load_yaml(base_path)
    experiment = load_yaml(path)
    return deep_merge(base, experiment)