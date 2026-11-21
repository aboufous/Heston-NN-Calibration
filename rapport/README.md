# Rapport (LaTeX)

Une section par fichier dans `sections/`, assemblées par `main.tex`. Les figures viennent directement de `results/figures/` :
relancer un notebook met le rapport à jour. Bibliographie : `references.bib`.

| Fichier | Contenu |
|---|---|
| `sections/01_introduction.tex` | Introduction |
| `sections/02_donnees.tex` | Étape 0 : pricer, données simulées, surfaces SPX |
| `sections/02b_mise_en_oeuvre.tex` | Organisation du code et des calculs |
| `sections/03_architectures.tex` | Étape 1 : choix de l'architecture |
| `sections/04_entrainement.tex` | Étape 2 : optimiseurs, initialisation, activation |
| `sections/05_dl_vs_ml.tex` | Étape 3 : réseau contre ML classique |
| `sections/06_calibration.tex` | Étape 4 : calibration (simulé et SPX) |
| `sections/07_conclusion.tex` | Conclusion |

## Obtenir le PDF

**Sans rien installer :** à chaque push qui modifie `rapport/` ou `results/figures/`, GitHub compile le rapport.
Sur GitHub : onglet **Actions** → dernière exécution de « Rapport (PDF) » → en bas, **Artifacts** → `rapport-pdf`.

**Sur son PC (pour voir le PDF en écrivant) :**
1. Installer **MiKTeX** (https://miktex.org/download), en acceptant l'installation automatique des paquets manquants.
2. Dans VS Code, installer l'extension **LaTeX Workshop**.
3. Ouvrir `rapport/main.tex` et enregistrer (Ctrl + S) : le PDF se compile ; l'icône d'aperçu en haut à droite l'affiche.

En ligne de commande, depuis `rapport/` : `latexmk -pdf main.tex` (MiKTeX demande d'installer Perl pour latexmk ;
sinon : `pdflatex main`, `bibtex main`, `pdflatex main`, `pdflatex main`).
