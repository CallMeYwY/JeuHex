"""L'« Ordi à règles » du jeu (`hexgame.ai`), réécrit pour être rapide.

Même décision que `hexgame.ai.choose_action` (les tests le vérifient) :
1. échange si le premier coup n'est pas sur un bord ;
2. victoire immédiate, sinon blocage d'une victoire adverse immédiate ;
3. sinon la case qui maximise
   coût_adverse × 1.15 − coût_propre + voisins × 0.10 − éloignement_centre × 0.025.

Accélérations : l'union-find teste les victoires immédiates en O(1), et les
coûts après un coup se déduisent de deux parcours 0-1 BFS (depuis chaque bord)
au lieu d'un Dijkstra par case candidate.
"""

from collections import deque

from hexgame.engine import EMPTY, RED

from .board import SWAP


def _distances(game, color, from_start, blocked=-1):
    """Nombre minimal de cases vides pour atteindre chaque case depuis un bord."""
    n, cells, neighbors = game.size, game.cells, game.neighbors
    inf = n * n + 1
    dist = [inf] * (n * n)
    queue = deque()
    if color == RED:
        edge = range(0, n) if from_start else range((n - 1) * n, n * n)
    else:
        edge = range(0, n * n, n) if from_start else range(n - 1, n * n, n)
    for i in edge:
        value = cells[i]
        if i == blocked or (value != EMPTY and value != color):
            continue
        cost = 1 if value == EMPTY else 0
        dist[i] = cost
        if cost:
            queue.append(i)
        else:
            queue.appendleft(i)
    while queue:
        i = queue.popleft()
        base = dist[i]
        for j in neighbors[i]:
            value = cells[j]
            if j == blocked or (value != EMPTY and value != color):
                continue
            step = 1 if value == EMPTY else 0
            if base + step < dist[j]:
                dist[j] = base + step
                if step:
                    queue.append(j)
                else:
                    queue.appendleft(j)
    return dist


def _end_cost(game, color, dist):
    n = game.size
    edge = range((n - 1) * n, n * n) if color == RED else range(n - 1, n * n, n)
    return min(dist[i] for i in edge)


def connection_cost(game, color, blocked=-1):
    """Équivalent de `hexgame.ai.connection_cost` (n² + 1 si aucun chemin)."""
    return _end_cost(game, color, _distances(game, color, True, blocked))


def immediate_action(game, legal):
    color, opponent = game.turn, 3 - game.turn
    threat = None
    for i in legal:
        if game.would_win(i, color):
            return i
        if threat is None and game.would_win(i, opponent):
            threat = i
    return threat


def score_moves(game, legal):
    """Score heuristique de chaque case légale, dans l'ordre de `legal`."""
    n = game.size
    inf = n * n + 1
    color, opponent = game.turn, 3 - game.turn
    own_a = _distances(game, color, True)
    own_b = _distances(game, color, False)
    own_now = _end_cost(game, color, own_a)
    other_a = _distances(game, opponent, True)
    other_b = _distances(game, opponent, False)
    other_now = _end_cost(game, opponent, other_a)
    middle = (n - 1) / 2
    cells, neighbors = game.cells, game.neighbors
    scores = []
    for i in legal:
        # Poser ici rend la case gratuite pour nous : les chemins qui la
        # traversent coûtent un de moins, les autres ne changent pas.
        own_cost = own_now
        if own_a[i] < inf and own_b[i] < inf:
            own_cost = min(own_now, own_a[i] + own_b[i] - 2)
        # La case devient un mur pour l'adversaire : seul un chemin minimal
        # passant par elle oblige à recalculer.
        other_cost = other_now
        if other_now < inf and other_a[i] + other_b[i] - 1 <= other_now:
            other_cost = connection_cost(game, opponent, blocked=i)
        r, c = divmod(i, n)
        adjacent = sum(cells[j] == color for j in neighbors[i])
        centrality = abs(r - middle) + abs(c - middle)
        scores.append(other_cost * 1.15 - own_cost + adjacent * 0.10 - centrality * 0.025)
    return scores


def choose_action(game, rng=None):
    """Renvoie (ligne, colonne), `SWAP`, ou None si la partie est finie.

    Avec `rng` (un `random.Random`), tire au hasard parmi les coups à moins de
    0.06 du meilleur, comme le jeu, pour varier les parties.
    """
    legal = game.legal_indices()
    if not legal:
        return None
    n = game.size
    if game.can_swap:
        row, col = game.moves[0]
        if 0 < row < n - 1 and 0 < col < n - 1:
            return SWAP
    immediate = immediate_action(game, legal)
    if immediate is not None:
        return divmod(immediate, n)
    scores = score_moves(game, legal)
    best_score = max(scores)
    if rng is not None:
        candidates = [i for s, i in zip(scores, legal) if s >= best_score - 0.06]
        return divmod(candidates[rng.randrange(len(candidates))], n)
    return divmod(legal[scores.index(best_score)], n)
