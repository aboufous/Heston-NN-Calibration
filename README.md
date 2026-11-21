# Calibration du modèle de Heston par réseau de neurones

Projet de Deep Learning, Université Paris Dauphine-PSL, novembre 2026.
Adam Boufous, Elise Ricourt, Matthieu Galtier, Omar Guenoun.

## Le projet

Calibrer le modèle de Heston, c'est trouver les 5 paramètres (v0, θ, κ, ξ, ρ) qui reproduisent au mieux une
surface de volatilité implicite du marché. Avec la méthode de Fourier, chaque surface coûte quelques millisecondes
et l'optimiseur doit la recalculer des centaines de fois.

Comme Horvath et al. (2021), nous entraînons un réseau de neurones qui prédit la surface (8 maturités × 11 strikes)
à partir des 5 paramètres, puis nous l'utilisons à la place de Fourier pendant la calibration. Nous mesurons ce
que nous gagnons en temps et ce que nous perdons en précision, sur des surfaces simulées et sur le SPX.

Le rapport complet est dans `rapport/` (PDF compilé automatiquement, voir plus bas).

## Les étapes

| Étape | Notebook | Contenu | Réalisé par |
|---|---|---|---|
| 0. Données | `00_donnees.ipynb` (+ `julia/`) | Pricer Fourier, 50 000 surfaces simulées, surfaces SPX | Elise Ricourt |
| 1. Architecture | `01_architectures.ipynb` | 10 architectures, précision et temps de réponse | Adam Boufous |
| 2. Entraînement | `02_choix_entrainement.ipynb` | Optimiseurs, initialisations, activations, réseau final | Adam Boufous |
| 3. DL contre ML | `03_dl_vs_ml.ipynb` | Réseau contre méthodes classiques, courbe d'apprentissage | Omar Guenoun |
| 4. Calibration | `04_calibration.ipynb` | Calibration réseau contre Fourier, SPX, seuil de rentabilité | Matthieu Galtier |

Les notebooks sont enregistrés avec leurs sorties : ils se lisent directement sur GitHub.

## Organisation du dépôt

```
DL_Heston_/
├── julia/                 Pricer Fourier (méthode COS) et génération des données d'entraînement
├── src/heston_nn/         Paquet Python appelé par les notebooks
│   ├── data/              Grille, chargement et découpage des données, surfaces SPX
│   ├── pricing/           Black-76, copie Python du pricer COS
│   ├── models/            Réseau (MLP Keras) et modèles classiques
│   ├── training/          Entraînement (perte qui ignore les NaN), métriques
│   ├── timing/            Temps de réponse des modèles
│   ├── calibration/       Calibration (Levenberg–Marquardt)
│   ├── experiments.py     Lancer des séries d'expériences et rassembler les résultats
│   └── plots.py           Figures communes
├── notebooks/             Un notebook par étape (00 à 04)
├── configs/               Règles communes (base.yaml) et une config par expérience
├── data/                  Jeu d'entraînement (surfaces.npz) et surfaces SPX
├── results/               Métriques, historiques, temps, réseau final et figures
├── tests/                 Tests Python (pytest)
└── rapport/               Rapport LaTeX, une section par fichier
```

Les notebooks restent courts : tout le code réutilisable est dans `src/heston_nn/`. Les règles communes (grille,
plages des paramètres, découpage 80 / 10 / 10, 200 époques, batch 32, graines, seuil de vitesse) sont écrites une
seule fois dans `configs/base.yaml`, lu par Julia comme par Python.

Chaque entraînement est enregistré dans `results/<expérience>/seed_<k>/` (erreurs, historique). Un entraînement
déjà présent n'est pas relancé : relancer un notebook relit simplement les résultats. Pour partager les calculs,
`MACHINE_ID` (0 à 3) dans les notebooks 01 et 02 fait calculer à chaque machine sa part des entraînements.

## Installation

Python 3.12 (TensorFlow ne suit pas toujours la dernière version de Python) :
```
pip install -r requirements.txt
pip install -e .
python -m pytest tests
```

Julia, seulement pour regénérer les données : installer Julia (julialang.org), puis `julia julia/setup.jl`.

## Reproduire les résultats

**Données (Julia), depuis la racine du dépôt :**
```
julia --project=julia julia/test/runtests.jl                                          # tests du pricer
julia --project=julia --threads=auto julia/generate_data.jl 50000 data/surfaces.npz   # jeu d'entraînement
julia --project=julia julia/benchmark.jl                                              # temps de Fourier
```
Le tirage est fixé par une graine : la même commande redonne exactement le même `data/surfaces.npz`.

**Expériences :** `jupyter notebook notebooks`, puis exécuter les notebooks dans l'ordre (00 à 04).

## Rapport

À chaque push qui modifie `rapport/` ou `results/figures/`, GitHub compile le rapport : onglet **Actions** →
« Rapport (PDF) » → **Artifacts** → `rapport-pdf`. En local : `latexmk -pdf main.tex` depuis `rapport/`.
## Utilisation de l'IA

Un assistant d'IA (Claude, Anthropic) a été utilisé pour aider à écrire le code et à rédiger le rapport, dans un but
de recherche et pour gagner du temps. Les choix de méthode ont été faits par l'équipe, et tout le contenu (code,
résultats, rapport) a été relu et vérifié par ses membres.
## Référence

Horvath, B., Muguruza, A. et Tomas, M. (2021), « Deep learning volatility: a deep neural network perspective on
pricing and calibration in (rough) volatility models », *Quantitative Finance*, 21(1), 11–27.
