"""Privilégie le blocage du chemin adverse."""

from hexgame.ai import choose_action as choose_by_path


def choose_action(game):
    return choose_by_path(game, opponent_weight=1.8)
