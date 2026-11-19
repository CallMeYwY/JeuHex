"""Contrat entre l'entraîneur et vos réseaux de neurones.

Un réseau est un objet `Brain` (voir `reseaux/README.md`) :

    act(obs)       -> Decision           choisir un coup          (obligatoire)
    learn(samples) -> dict de mesures    apprendre d'un lot       (facultatif)
    explain(obs)   -> Explanation        montrer son raisonnement (facultatif)
    save(dossier) / load(dossier)        sauvegarde des poids     (facultatif)

Tout ce qui est échangé est en numpy, sans dépendance à une bibliothèque de
réseaux particulière : numpy pur, PyTorch, etc. conviennent.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from hexgame.engine import BLUE, EMPTY, RED

from .board import SWAP

Move = Union[Tuple[int, int], str]


class Observation:
    """Ce que voit le réseau au moment de jouer.

    Attributs (orientation réelle du plateau, comme à l'écran) :
      size       taille n du plateau
      board      tableau n×n int8 : 0 vide, 1 rouge, 2 bleu
      to_play    couleur au trait (1 rouge : haut↔bas, 2 bleu : gauche↔droite)
      legal      tableau n×n bool des cases jouables
      can_swap   vrai si « swap » (règle d'échange) est un coup permis
      moves      coups déjà joués : (ligne, colonne) ou "swap"
      training   vrai pendant l'entraînement (explorer), faux en évaluation

    Vue canonique : `planes()` présente toujours le plateau du point de vue du
    joueur au trait, qui relie le HAUT au BAS. Pour bleu, le plateau est
    transposé (ligne ↔ colonne), ce qui conserve les voisinages du Hex. Un même
    réseau peut ainsi jouer les deux couleurs.
    """

    __slots__ = ("size", "board", "to_play", "legal", "can_swap", "moves", "training")

    def __init__(self, size, board, to_play, can_swap=False, moves=(), training=False):
        self.size = size
        self.board = np.asarray(board, dtype=np.int8).reshape(size, size)
        self.to_play = to_play
        self.legal = self.board == EMPTY
        self.can_swap = can_swap
        self.moves = list(moves)
        self.training = training

    @classmethod
    def from_game(cls, game, training=False):
        """Depuis un `FastHex` de l'entraîneur ou un `HexGame` du jeu."""
        if hasattr(game, "cells"):
            board = game.cells
            moves = game.moves
        else:
            board = game.board
            moves = [m[:2] for m in game.moves]
            if game.swapped:
                moves.insert(1, SWAP)
        obs = cls(game.size, board, game.turn, game.can_swap, moves, training)
        if game.winner is not None:
            obs.legal[:] = False
        return obs

    @property
    def transposed(self):
        """Vrai si la vue canonique est transposée (bleu au trait)."""
        return self.to_play == BLUE

    def to_canonical(self, grid):
        """Passe une grille n×n (ou c×n×n) de l'orientation réelle à la canonique."""
        return np.swapaxes(grid, -1, -2) if self.transposed else grid

    to_real = to_canonical  # La transposition est sa propre inverse.

    def cell_to_canonical(self, cell):
        return (cell[1], cell[0]) if self.transposed else tuple(cell)

    cell_to_real = cell_to_canonical

    def planes(self):
        """float32 (3, n, n) canonique : [mes pierres, pierres adverses, cases vides]."""
        board = self.to_canonical(self.board)
        return np.stack([board == self.to_play, board == 3 - self.to_play,
                         board == EMPTY]).astype(np.float32)

    def legal_canonical(self):
        return self.to_canonical(self.legal)

    def legal_moves(self):
        return [tuple(int(v) for v in cell) for cell in np.argwhere(self.legal)]


@dataclass
class Decision:
    """Réponse de `act`. `policy` (n×n, orientation réelle, somme 1) et `value`
    (dans [-1, 1], point de vue du joueur au trait) sont facultatifs mais
    permettent à l'interface d'afficher le raisonnement."""
    move: Move
    policy: Optional[np.ndarray] = None
    value: Optional[float] = None


@dataclass
class Sample:
    """Une décision prise pendant une partie, avec son issue.

    outcome        +1 si le JOUEUR qui a pris la décision a fini par gagner, -1 sinon
    color_outcome  +1 si la COULEUR au trait (obs.to_play) a gagné, -1 sinon :
                   la bonne cible pour une tête « valeur » (qui va gagner d'ici ?)
    actor          qui l'a prise : "reseau", "regles", "aleatoire" ou "partie"
    policy         la politique renvoyée par le réseau à ce moment, si elle existe
    game, ply      numéro de la partie et rang du coup dans celle-ci (0, 1, 2…) :
                   les décisions d'une même partie se suivent dans l'ordre, ce qui
                   permet aussi un apprentissage coup par coup (TD : comparer la
                   valeur estimée à un coup et au suivant)

    Les deux issues ne diffèrent qu'avec la règle d'échange : le premier coup
    (sa pierre change de camp) et la décision d'échanger elle-même.
    """
    obs: Observation
    move: Move
    outcome: float
    actor: str
    policy: Optional[np.ndarray] = None
    color_outcome: float = 0.0
    game: int = 0
    ply: int = 0


@dataclass
class Layer:
    """Une couche telle que l'interface doit la dessiner.

    values   1D : neurones ; 2D (h, w) : une carte ; 3D (c, h, w) : c cartes
    weights  facultatif, (nb_sorties, nb_entrées) : poids depuis la couche
             source (aplatie). Permet de tracer les connexions et de voir
             « ce que regarde » un neurone en le survolant.
    source   nom de la couche d'où viennent ces poids (par défaut : la précédente),
             utile pour deux têtes branchées sur la même couche
    channels noms des cartes d'une couche 3D (ex. ["mes pierres", ...])
    """
    name: str
    values: np.ndarray
    weights: Optional[np.ndarray] = None
    channels: Sequence[str] = ()
    note: str = ""
    source: Optional[str] = None


@dataclass
class Explanation:
    layers: List[Layer] = field(default_factory=list)
    policy: Optional[np.ndarray] = None   # n×n, orientation réelle
    value: Optional[float] = None         # [-1, 1], joueur au trait
    notes: List[str] = field(default_factory=list)


class Brain:
    """Classe de base des réseaux. Seule `act` est obligatoire."""

    #: Nom affiché dans l'entraîneur (par défaut : le nom du dossier).
    name = None
    #: Une phrase de présentation.
    description = ""

    def __init__(self, size):
        self.size = size

    def act(self, obs: Observation) -> Union[Decision, Move]:
        raise NotImplementedError("Un réseau doit implémenter act(obs).")

    def learn(self, samples: List[Sample]) -> dict:
        """Apprend d'un lot de décisions ; renvoie des mesures à tracer
        (ex. {"perte": 0.42}). Par défaut, n'apprend rien."""
        return {}

    def explain(self, obs: Observation) -> Optional[Explanation]:
        """Détail des couches pour l'affichage. Par défaut : rien."""
        return None

    def save(self, folder: str) -> None:
        """Écrit les poids dans `folder` (dossier existant)."""

    def load(self, folder: str) -> bool:
        """Relit les poids depuis `folder` ; renvoie False s'il n'y a rien."""
        return False


def normalize_decision(result, obs):
    """Accepte une Decision, un tuple (ligne, colonne) ou "swap"."""
    decision = result if isinstance(result, Decision) else Decision(result)
    move = decision.move
    if isinstance(move, str):
        if move.lower() not in ("swap", "echange", "échange"):
            raise ValueError("Coup inconnu : {!r}".format(move))
        decision.move = SWAP
    else:
        decision.move = (int(move[0]), int(move[1]))
    return decision


def is_legal(move, obs):
    if move == SWAP:
        return obs.can_swap
    r, c = move
    return 0 <= r < obs.size and 0 <= c < obs.size and bool(obs.legal[r, c])


__all__ = ["Brain", "Decision", "Explanation", "Layer", "Observation", "Sample",
           "SWAP", "RED", "BLUE", "EMPTY", "normalize_decision", "is_legal"]
