"""Modèle vide à copier : `cp -r reseaux/_modele reseaux/mon_reseau`.

Les dossiers dont le nom commence par « _ » sont ignorés par l'entraîneur :
renommez votre copie sans « _ » pour qu'elle apparaisse dans la liste.

Seule `act` est obligatoire. Tout le reste est facultatif mais rend
l'entraînement (learn), la sauvegarde (save/load) et l'affichage du
raisonnement (explain) possibles.
"""

import numpy as np

from hextrainer.api import SWAP, Brain, Decision, Explanation, Layer


class Reseau(Brain):
    name = "Mon réseau"
    description = "Une phrase qui décrit l'architecture."

    def __init__(self, size):
        super().__init__(size)            # self.size = n (plateau n × n)
        self.rng = np.random.default_rng()

    def act(self, obs):
        """Choisir un coup.

        obs.planes()        -> float32 (3, n, n) vue canonique : le joueur au
                               trait relie toujours le HAUT au BAS
        obs.legal_canonical()  cases jouables dans cette même vue
        obs.cell_to_real(c) -> convertit une case canonique en case réelle
        obs.can_swap        -> vous pouvez renvoyer SWAP (règle d'échange)
        obs.training        -> vrai pendant l'entraînement : pensez à explorer

        Renvoyez Decision(case_reelle, politique_reelle_nxn, valeur) ou juste
        un tuple (ligne, colonne) / SWAP.
        """
        legal = np.argwhere(obs.legal_canonical())
        cell = tuple(legal[self.rng.integers(len(legal))])
        return Decision(obs.cell_to_real(cell))

    def learn(self, samples):
        """`samples` : liste de Sample(obs, move, outcome, actor, policy, color_outcome).

        outcome        ±1 : le joueur qui a choisi ce coup a-t-il gagné ?
        color_outcome  ±1 : la couleur au trait a-t-elle gagné ? (cible « valeur »)
        `move` est en orientation réelle : obs.cell_to_canonical(move) le ramène
        dans la vue canonique. Renvoyez un dict de mesures à tracer."""
        return {}

    def explain(self, obs):
        """Facultatif : les couches à dessiner (voir hextrainer/api.py, Layer)."""
        return Explanation(layers=[Layer("Entrée", obs.planes(),
                                         channels=("mes pierres", "adversaire", "vides"))])

    def save(self, folder):
        pass  # ex. np.savez(os.path.join(folder, "poids.npz"), ...)

    def load(self, folder):
        return False
