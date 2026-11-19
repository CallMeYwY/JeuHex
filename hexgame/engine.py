"""Règles du Hex, sans dépendance graphique.

Rouge relie le haut au bas ; bleu relie la gauche à la droite.
L'échange conserve la pierre en place et échange les couleurs des joueurs.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

EMPTY, RED, BLUE = 0, 1, 2
Cell = Tuple[int, int]
NEIGHBORS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))


@dataclass
class Snapshot:
    board: List[List[int]]
    turn: int
    owners: List[int]
    moves: List[Tuple[int, int, int]]
    swapped: bool
    winner: Optional[int]
    path: List[Cell]


class HexGame:
    def __init__(self, size=9):
        if not isinstance(size, int) or size < 2:
            raise ValueError("Le plateau doit mesurer au moins 2 × 2.")
        self.size = size
        self.board = [[EMPTY] * size for _ in range(size)]
        self.turn = RED
        # Index = couleur ; valeur = numéro du joueur.
        self.owners = [0, 1, 2]
        self.moves = []
        self.swapped = False
        self.winner = None
        self.path = []
        self.history = []

    @property
    def current_player(self):
        return self.owners[self.turn]

    @property
    def can_swap(self):
        return len(self.moves) == 1 and not self.swapped and self.winner is None

    @property
    def last_move(self):
        return self.moves[-1][:2] if self.moves else None

    def neighbors(self, row, col):
        for dr, dc in NEIGHBORS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                yield nr, nc

    def legal_moves(self):
        if self.winner is not None:
            return []
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.board[r][c] == EMPTY]

    def _snapshot(self):
        return Snapshot([row[:] for row in self.board], self.turn, self.owners[:],
                        self.moves[:], self.swapped, self.winner, self.path[:])

    def play(self, row, col):
        if (self.winner is not None or not 0 <= row < self.size
                or not 0 <= col < self.size or self.board[row][col] != EMPTY):
            return False
        self.history.append(self._snapshot())
        self.board[row][col] = self.turn
        self.moves.append((row, col, self.turn))
        self.path = self.winning_path(self.turn)
        if self.path:
            self.winner = self.turn
        else:
            self.turn = 3 - self.turn
        return True

    def swap(self):
        if not self.can_swap:
            return False
        self.history.append(self._snapshot())
        self.owners[RED], self.owners[BLUE] = self.owners[BLUE], self.owners[RED]
        self.swapped = True
        # Bleu joue toujours le deuxième coup, désormais avec le joueur 1.
        return True

    def undo(self):
        if not self.history:
            return False
        state = self.history.pop()
        self.board = state.board
        self.turn = state.turn
        self.owners = state.owners
        self.moves = state.moves
        self.swapped = state.swapped
        self.winner = state.winner
        self.path = state.path
        return True

    def winning_path(self, color):
        """Retrouve par BFS une chaîne entre les deux bords de la couleur."""
        starts = ([(0, c) for c in range(self.size)] if color == RED
                  else [(r, 0) for r in range(self.size)])
        queue = deque(cell for cell in starts if self.board[cell[0]][cell[1]] == color)
        previous = {cell: None for cell in queue}
        while queue:
            cell = queue.popleft()
            r, c = cell
            if (r if color == RED else c) == self.size - 1:
                path = []
                while cell is not None:
                    path.append(cell)
                    cell = previous[cell]
                return path[::-1]
            for neighbor in self.neighbors(r, c):
                nr, nc = neighbor
                if neighbor not in previous and self.board[nr][nc] == color:
                    previous[neighbor] = (r, c)
                    queue.append(neighbor)
        return []
