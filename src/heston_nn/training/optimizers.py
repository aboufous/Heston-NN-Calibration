"""Optimiseurs comparés dans la série A de l'étape 2, en Keras (comme dans le TP3).

Ce que fait ce fichier :
- construit l'optimiseur Keras demandé à partir de son nom et du taux d'apprentissage ;
- garde les autres hyperparamètres aux valeurs par défaut de Keras, identiques pour tous les essais ;
- crée un NOUVEL optimiseur à chaque appel : un optimiseur Keras ne se réutilise pas d'un réseau à l'autre.
"""

import keras

OPTIMIZERS = ("sgd", "momentum", "adagrad", "rmsprop", "adam")  # les 5 optimiseurs de la série A


def build_optimizer(name, learning_rate):
    """Renvoie un nouvel optimiseur Keras correspondant à name (ex. "adam"), avec ce taux d'apprentissage."""
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate)                 # descente de gradient simple
    if name == "momentum":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)   # SGD avec inertie
    if name == "adagrad":
        return keras.optimizers.Adagrad(learning_rate=learning_rate)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    raise ValueError(f"Optimiseur inconnu : {name!r} (attendu : {', '.join(OPTIMIZERS)})")
    