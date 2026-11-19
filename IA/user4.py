"""Privilégie les groupes de pierres connectées."""

from hexgame.ai import choose_action as choose_by_path


def choose_action(game):
    return choose_by_path(game, adjacency_weight=0.65)
