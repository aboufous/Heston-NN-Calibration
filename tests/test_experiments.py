"""Tests de experiments.py, sur de fausses données et des entraînements très courts."""

import numpy as np
import yaml

from heston_nn.config import load_config
from heston_nn.data.split import prepare_data
from heston_nn.experiments import agregate_result, collect_results, my_share, run_configs, summarize

RNG = np.random.default_rng(0)
X = RNG.uniform(size=(300, 5))
Y = np.sin(X @ RNG.normal(size=(5, 88)))
PREPARED = prepare_data(X, Y, load_config("configs/etape1/arch_4x30.yaml"))


def make_config(folder, name, width):
    """Écrit une petite config d'essai (3 époques) dans un dossier temporaire."""
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"name": name,
                                    "model": {"hidden_layers": 2, "width": width},
                                    "train": {"epochs": 3, "batch_size": 64}}))
    return path


def test_my_share_repartit_tout_sans_doublon():
    jobs = list(range(10))
    parts = [my_share(jobs, k) for k in range(4)]
    assert sorted(sum(parts, [])) == jobs                     # tout est fait, une seule fois
    assert max(map(len, parts)) - min(map(len, parts)) <= 1   # parts équilibrées


def test_run_configs_puis_resume(tmp_path):
    configs = [make_config(tmp_path, "etape1_essai_a", 8), make_config(tmp_path, "etape1_essai_b", 16)]
    jobs = [(c, s) for c in configs for s in (0, 1)]
    results = run_configs(jobs, PREPARED, results_root=tmp_path / "results")
    assert len(results) == 4

    # relancer ne refait rien et redonne les mêmes résultats
    again = run_configs(jobs, PREPARED, results_root=tmp_path / "results")
    assert [r["metrics"] for r in again] == [r["metrics"] for r in results]

    # relecture de toute l'étape, agrégation et tableau
    collected = collect_results(1, results_root=tmp_path / "results")
    assert len(collected) == 4
    training, validation = agregate_result(collected, "etape1_essai_a", "loss")
    assert training.shape == (2, 3) and validation.shape == (2, 3)   # 2 graines × 3 époques
    table = summarize(collected)
    assert list(table["n_seeds"]) == [2, 2]