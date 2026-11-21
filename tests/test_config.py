"""Tests de config.py.

Vérifie : la lecture d'un YAML, la fusion base + expérience, et que base.yaml n'est jamais modifié.
Lancer depuis la racine du dépôt :  pytest tests/test_config.py -v
"""

from heston_nn.config import BASE_CONFIG, deep_merge, load_config, load_yaml, repo_path


def test_load_yaml_lit_base():
    # base.yaml se lit et contient les règles communes
    cfg = load_yaml(BASE_CONFIG)
    assert cfg["train"]["epochs"] == 200
    assert cfg["model"]["activation"] == "elu"


def test_deep_merge_garde_les_cles_non_precisees():
    # l'expérience change la taille, mais l'activation de la base est conservée
    base = {"model": {"hidden_layers": 4, "width": 30, "activation": "elu"}}
    override = {"model": {"hidden_layers": 2, "width": 32}}
    merged = deep_merge(base, override)
    assert merged["model"] == {"hidden_layers": 2, "width": 32, "activation": "elu"}


def test_deep_merge_ne_modifie_pas_base():
    # la base reste intacte après une fusion
    base = {"model": {"width": 30}}
    deep_merge(base, {"model": {"width": 32}})
    assert base["model"]["width"] == 30


def test_deep_merge_ajoute_une_nouvelle_cle():
    # une clé absente de la base est ajoutée
    merged = deep_merge({"a": 1}, {"name": "essai"})
    assert merged == {"a": 1, "name": "essai"}


def test_load_config_architecture():
    # config complète d'une expérience de l'étape 1
    cfg = load_config("configs/etape1/arch_2x32.yaml")
    assert cfg["model"]["hidden_layers"] == 2   # vient de l'expérience
    assert cfg["model"]["width"] == 32          # vient de l'expérience
    assert cfg["model"]["activation"] == "elu"  # vient de base.yaml
    assert cfg["train"]["epochs"] == 200        # vient de base.yaml
    assert cfg["name"] == "etape1_arch_2x32"


def test_toutes_les_configs_etape1_se_chargent():
    # les 10 configs de l'étape 1 se lisent sans erreur
    for path in sorted(repo_path("configs", "etape1").glob("*.yaml")):
        cfg = load_config(path)
        assert cfg["model"]["hidden_layers"] in (2, 4, 6)