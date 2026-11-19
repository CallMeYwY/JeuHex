"""IA heuristique : gains immédiats, blocages et plus courts chemins.

Ce joueur est volontairement simple, sans apprentissage ni service externe.
"""

import heapq
import math

from .engine import EMPTY, RED


def connection_cost(game, color):
    """Nombre minimum de cases vides pour joindre les deux bords."""
    size = game.size
    distances = {}
    queue = []
    for i in range(size):
        r, c = (0, i) if color == RED else (i, 0)
        value = game.board[r][c]
        if value in (EMPTY, color):
            cost = int(value == EMPTY)
            distances[(r, c)] = cost
            heapq.heappush(queue, (cost, r, c))
    while queue:
        cost, r, c = heapq.heappop(queue)
        if cost != distances[(r, c)]:
            continue
        if (r if color == RED else c) == size - 1:
            return cost
        for nr, nc in game.neighbors(r, c):
            value = game.board[nr][nc]
            if value not in (EMPTY, color):
                continue
            next_cost = cost + int(value == EMPTY)
            if next_cost < distances.get((nr, nc), math.inf):
                distances[(nr, nc)] = next_cost
                heapq.heappush(queue, (next_cost, nr, nc))
    return size * size + 1


def immediate_action(game):
    """Victoire immédiate, puis blocage d'une victoire adverse, sans mutation."""
    color, opponent = game.turn, 3 - game.turn
    threats = []
    for r, c in game.legal_moves():
        try:
            game.board[r][c] = color
            if game.winning_path(color):
                return r, c
            game.board[r][c] = opponent
            if game.winning_path(opponent):
                threats.append((r, c))
        finally:
            game.board[r][c] = EMPTY
    return threats[0] if threats else None


def choose_action(game, rng=None):
    """Renvoie une case, 'swap', ou None lorsque la partie est finie.

    L'appelant fournit une copie du jeu si le calcul tourne dans un thread.
    Chaque case temporaire est restaurée même en cas d'erreur.
    """
    legal = game.legal_moves()
    if not legal:
        return None
    if game.can_swap:
        row, col, _ = game.moves[0]
        if 0 < row < game.size - 1 and 0 < col < game.size - 1:
            return "swap"
    immediate = immediate_action(game)
    if immediate is not None:
        return immediate
    color, opponent = game.turn, 3 - game.turn
    best_cell, best_score = legal[0], -math.inf
    scores = []
    middle = (game.size - 1) / 2
    for r, c in legal:
        try:
            game.board[r][c] = color
            own_cost = connection_cost(game, color)
            other_cost = connection_cost(game, opponent)
            adjacent = sum(game.board[nr][nc] == color
                           for nr, nc in game.neighbors(r, c))
            centrality = abs(r - middle) + abs(c - middle)
            score = other_cost * 1.15 - own_cost + adjacent * 0.10 - centrality * 0.025
            scores.append((score, (r, c)))
            if score > best_score:
                best_cell, best_score = (r, c), score
        finally:
            game.board[r][c] = EMPTY
    if rng is not None:
        candidates = [cell for score, cell in scores if score >= best_score - 0.06]
        return candidates[int(rng.integers(len(candidates)))]
    return best_cell
