"""Joueurs (réseau, règles, hasard) et déroulement d'une partie rapide."""

import random
from dataclasses import dataclass, field
from typing import List, Optional

from .api import Observation, Sample, is_legal, normalize_decision
from .board import SWAP, FastHex
from . import rules


class Agent:
    actor = "?"
    label = "?"

    def choose(self, game, obs=None):
        """Renvoie (coup, Decision ou None)."""
        raise NotImplementedError


class RandomAgent(Agent):
    actor, label = "aleatoire", "Aléatoire"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose(self, game, obs=None):
        if game.can_swap and self.rng.random() < 0.5:
            return SWAP, None
        return divmod(self.rng.choice(game.legal_indices()), game.size), None


class RulesAgent(Agent):
    """L'ordi à règles du jeu ; `varied` ajoute un peu de hasard entre coups équivalents."""
    actor, label = "regles", "Ordi à règles"

    def __init__(self, varied=True, seed=None, opening_noise=0.0):
        self.rng = random.Random(seed) if varied else None
        self.opening_noise = opening_noise
        self._noise_rng = random.Random(seed)

    def choose(self, game, obs=None):
        # Un premier coup parfois aléatoire diversifie les parties générées.
        if self.opening_noise and len(game.moves) < 2 and not game.can_swap \
                and self._noise_rng.random() < self.opening_noise:
            return divmod(self._noise_rng.choice(game.legal_indices()), game.size), None
        return rules.choose_action(game, self.rng), None


class BrainAgent(Agent):
    """Enveloppe un réseau. Un coup illégal est compté puis remplacé au hasard."""
    actor = "reseau"

    def __init__(self, brain, label="Réseau", training=False, seed=None):
        self.brain = brain
        self.label = label
        self.training = training
        self.illegal = 0
        self.rng = random.Random(seed)

    def choose(self, game, obs=None):
        if obs is None:
            obs = Observation.from_game(game, self.training)
        decision = normalize_decision(self.brain.act(obs), obs)
        if not is_legal(decision.move, obs):
            self.illegal += 1
            decision.move = divmod(self.rng.choice(game.legal_indices()), game.size)
        return decision.move, decision


@dataclass
class GameRecord:
    size: int
    moves: list
    winner_color: int
    winner_player: int          # 1 ou 2 : l'agent gagnant (après un éventuel échange)
    swapped: bool
    labels: tuple
    samples: List[Sample] = field(default_factory=list)


def play_game(size, agent1, agent2, swap_rule=True, collect=False, on_move=None,
              should_stop=None):
    """Joue une partie complète ; `agent1` commence avec les rouges.

    collect   construit un `Sample` par décision (pour l'apprentissage)
    on_move   rappel `on_move(game, decision)` après chaque coup (mode visible)
    """
    game = FastHex(size, swap_rule)
    agents = {1: agent1, 2: agent2}
    pending = []  # (obs, coup, joueur, acteur, politique)
    while game.winner is None:
        if should_stop is not None and should_stop():
            return None
        player = game.current_player
        agent = agents[player]
        needs_obs = collect or isinstance(agent, BrainAgent)
        obs = Observation.from_game(game, getattr(agent, "training", False)) if needs_obs else None
        move, decision = agent.choose(game, obs)
        if collect:
            pending.append((obs, move, player, agent.actor,
                            decision.policy if decision is not None else None))
        if not game.apply(move):
            raise RuntimeError("{} a joué un coup illégal : {}".format(agent.label, move))
        if on_move is not None:
            on_move(game, decision)
    winner_player = game.owners[game.winner]
    record = GameRecord(size, list(game.moves), game.winner, winner_player, game.swapped,
                        (agent1.label, agent2.label))
    record.samples = [Sample(obs, move, 1.0 if player == winner_player else -1.0, actor, policy,
                             1.0 if obs.to_play == game.winner else -1.0, ply=ply)
                      for ply, (obs, move, player, actor, policy) in enumerate(pending)]
    return record
