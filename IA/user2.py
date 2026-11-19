"""Privilégie son propre chemin."""

from hexgame.ai import choose_action as choose_by_path


def choose_action(game):
    return choose_by_path(game, opponent_weight=0.6)
