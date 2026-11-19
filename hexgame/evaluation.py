"""Estime l'avantage des rouges à partir des chemins de connexion."""

from .ai import connection_cost
from .engine import BLUE, RED


def position_score(game):
    """Indice signé de −100 (victoire bleue) à +100 (victoire rouge)."""
    if game.winner is not None:
        return 100 if game.winner == RED else -100
    red_cost = connection_cost(game, RED)
    blue_cost = connection_cost(game, BLUE)
    return round(100 * (blue_cost - red_cost) / (blue_cost + red_cost))
