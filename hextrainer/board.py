"""Plateau de Hex rapide pour l'entraînement.

Mêmes règles que `hexgame.engine.HexGame` (rouge haut↔bas, bleu gauche↔droite,
règle d'échange au deuxième coup) mais sans historique ni copie : un tableau à
plat et un union-find détectent la victoire en temps quasi constant.
"""

from collections import deque

from hexgame.engine import BLUE, EMPTY, NEIGHBORS, RED

SWAP = "swap"
_NEIGHBORS_CACHE = {}


def neighbor_table(size):
    """Voisins de chaque case, en indices à plat (mis en cache par taille)."""
    if size not in _NEIGHBORS_CACHE:
        table = []
        for r in range(size):
            for c in range(size):
                table.append(tuple((r + dr) * size + c + dc for dr, dc in NEIGHBORS
                                   if 0 <= r + dr < size and 0 <= c + dc < size))
        _NEIGHBORS_CACHE[size] = table
    return _NEIGHBORS_CACHE[size]


def cell_label(row, col):
    """Même notation que l'interface du jeu : colonne en lettre, ligne en chiffre."""
    return "{}{}".format(chr(65 + col), row + 1)


def move_label(move):
    return "echange" if move == SWAP else cell_label(*move)


def parse_move(text):
    text = text.strip()
    if text.lower() in ("echange", "échange", "swap"):
        return SWAP
    return int(text[1:]) - 1, ord(text[0].upper()) - 65


class FastHex:
    """Partie en cours. `owners[couleur]` donne le joueur (1 ou 2) qui la contrôle."""

    __slots__ = ("size", "cells", "parent", "turn", "owners", "moves", "swapped",
                 "winner", "swap_rule", "neighbors", "_top", "_bottom", "_left", "_right")

    def __init__(self, size, swap_rule=True):
        if not isinstance(size, int) or size < 2:
            raise ValueError("Le plateau doit mesurer au moins 2 × 2.")
        n = size * size
        self.size = size
        self.cells = [EMPTY] * n
        self.parent = list(range(n + 4))
        self._top, self._bottom, self._left, self._right = n, n + 1, n + 2, n + 3
        self.turn = RED
        self.owners = [0, 1, 2]
        self.moves = []
        self.swapped = False
        self.winner = None
        self.swap_rule = swap_rule
        self.neighbors = neighbor_table(size)

    @classmethod
    def from_moves(cls, size, moves, swap_rule=True):
        game = cls(size, swap_rule)
        for move in moves:
            if not game.apply(move):
                raise ValueError("Coup illégal : {}".format(move_label(move)))
        return game

    # --- Union-find ---------------------------------------------------------
    def _find(self, i):
        parent = self.parent
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self.parent[ra] = rb

    # --- État -----------------------------------------------------------------
    @property
    def current_player(self):
        return self.owners[self.turn]

    @property
    def can_swap(self):
        return (self.swap_rule and len(self.moves) == 1 and not self.swapped
                and self.winner is None)

    @property
    def last_move(self):
        for move in reversed(self.moves):
            if move != SWAP:
                return move
        return None

    def at(self, row, col):
        return self.cells[row * self.size + col]

    def grid(self):
        n = self.size
        return [self.cells[r * n:(r + 1) * n] for r in range(n)]

    def legal_indices(self):
        if self.winner is not None:
            return []
        return [i for i, v in enumerate(self.cells) if v == EMPTY]

    def legal_moves(self):
        n = self.size
        return [(i // n, i % n) for i in self.legal_indices()]

    # --- Coups ----------------------------------------------------------------
    def apply(self, move):
        return self.swap() if move == SWAP else self.play(*move)

    def play(self, row, col):
        n = self.size
        if self.winner is not None or not (0 <= row < n and 0 <= col < n):
            return False
        return self.play_index(row * n + col)

    def play_index(self, i):
        if self.winner is not None or self.cells[i] != EMPTY:
            return False
        n, color = self.size, self.turn
        row, col = divmod(i, n)
        self.cells[i] = color
        self.moves.append((row, col))
        for j in self.neighbors[i]:
            if self.cells[j] == color:
                self._union(i, j)
        if color == RED:
            if row == 0:
                self._union(i, self._top)
            if row == n - 1:
                self._union(i, self._bottom)
            won = self._find(self._top) == self._find(self._bottom)
        else:
            if col == 0:
                self._union(i, self._left)
            if col == n - 1:
                self._union(i, self._right)
            won = self._find(self._left) == self._find(self._right)
        if won:
            self.winner = color
        else:
            self.turn = 3 - color
        return True

    def swap(self):
        if not self.can_swap:
            return False
        self.owners[RED], self.owners[BLUE] = self.owners[BLUE], self.owners[RED]
        self.swapped = True
        self.moves.append(SWAP)
        # Comme dans le jeu : la pierre reste, bleu joue toujours le coup suivant.
        return True

    def would_win(self, i, color):
        """Vrai si poser `color` en `i` relie ses deux bords (sans modifier le plateau)."""
        n = self.size
        row, col = divmod(i, n)
        if color == RED:
            start, end = self._find(self._top), self._find(self._bottom)
            touch_start, touch_end = row == 0, row == n - 1
        else:
            start, end = self._find(self._left), self._find(self._right)
            touch_start, touch_end = col == 0, col == n - 1
        cells = self.cells
        for j in self.neighbors[i]:
            if cells[j] == color:
                root = self._find(j)
                touch_start = touch_start or root == start
                touch_end = touch_end or root == end
        return touch_start and touch_end

    def clone(self):
        other = FastHex.__new__(FastHex)
        for name in FastHex.__slots__:
            value = getattr(self, name)
            if isinstance(value, list) and name != "neighbors":
                value = value[:]
            setattr(other, name, value)
        return other

    def winning_path(self):
        """Chaîne gagnante (pour l'affichage), retrouvée par BFS comme dans le jeu."""
        color, n = self.winner, self.size
        if color is None:
            return []
        starts = [c for c in range(n)] if color == RED else [r * n for r in range(n)]
        queue = deque(i for i in starts if self.cells[i] == color)
        previous = {i: None for i in queue}
        while queue:
            i = queue.popleft()
            r, c = divmod(i, n)
            if (r if color == RED else c) == n - 1:
                path = []
                while i is not None:
                    path.append(divmod(i, n))
                    i = previous[i]
                return path[::-1]
            for j in self.neighbors[i]:
                if j not in previous and self.cells[j] == color:
                    previous[j] = i
                    queue.append(j)
        return []
