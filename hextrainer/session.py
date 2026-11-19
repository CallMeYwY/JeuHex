"""Boucle d'entraînement, sans interface : utilisée par l'appli et par la ligne de commande.

Chaque tour : jouer (ou relire) quelques parties, donner leurs décisions au
réseau (`brain.learn`), puis de temps en temps évaluer contre l'ordi à règles et
le hasard, et sauvegarder.
"""

import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

from . import registry
from .agents import BrainAgent, RandomAgent, RulesAgent, play_game
from .dataset import load_samples

OPPONENTS = {
    "autojeu": "Autojeu",
    "regles": "Ordi à règles",
    "aleatoire": "Aléatoire",
    "parties": "Parties enregistrées",
}


@dataclass
class TrainConfig:
    network: str
    size: int = 7
    opponent: str = "autojeu"
    dataset: Optional[str] = None      # chemin d'un .jsonl (mode « parties »)
    swap_rule: bool = True
    games_per_update: int = 16         # parties par appel à learn()
    eval_every: int = 500              # parties d'entraînement entre deux évaluations
    eval_games: int = 20               # parties par adversaire d'évaluation
    save_every: float = 30.0           # secondes entre deux sauvegardes
    move_delay: float = 0.0            # > 0 : parties jouées au ralenti (mode visible)
    max_games: Optional[int] = None
    max_seconds: Optional[float] = None
    seed: Optional[int] = None

    def to_dict(self):
        return asdict(self)


class Journal:
    """Historique des courbes, sauvegardé avec les poids (journal.json)."""

    LIMIT = 600

    def __init__(self, points=None, evals=None, games=0):
        self.points = points or []   # {"parties", "mesures", "victoires"}
        self.evals = evals or []     # {"parties", "vs_regles", "vs_aleatoire"}
        self.games = games

    @classmethod
    def load(cls, folder):
        try:
            data = json.loads((folder / "journal.json").read_text(encoding="utf-8"))
            return cls(data.get("points"), data.get("evaluations"), data.get("parties", 0))
        except (OSError, ValueError):
            return cls()

    def to_json(self):
        return json.dumps({"parties": self.games, "points": self.points,
                           "evaluations": self.evals}, ensure_ascii=False)

    def add_point(self, point):
        self.points.append(point)
        if len(self.points) > self.LIMIT:
            self.points = self.points[::2]

    def add_eval(self, entry):
        self.evals.append(entry)
        if len(self.evals) > self.LIMIT:
            self.evals = self.evals[::2]


class Session:
    def __init__(self, config, emit=None):
        self.cfg = config
        self.emit = emit or (lambda kind, data: None)
        self.rng = random.Random(config.seed)
        self.brain = registry.create(config.network, config.size)
        self.loaded = registry.load_checkpoint(self.brain, config.network, config.size)
        self.folder = registry.checkpoint_dir(config.network, config.size)
        self.journal = Journal.load(self.folder)
        self.base_games = self.journal.games
        self.stop_requested = False
        self.agent = BrainAgent(self.brain, "Réseau", training=True, seed=self.rng.random())
        self.session_games = 0
        self.session_moves = 0
        self.started = time.perf_counter()
        self._last_stats = self._last_game = self._last_save = time.perf_counter()
        self._window = {"wins": 0, "games": 0, "red": 0, "metrics": {}, "updates": 0}
        self._last_point = time.perf_counter()
        self._next_eval = config.eval_every if config.eval_every else None
        self._dataset = None
        self.last_metrics = {}
        self.last_eval = self.journal.evals[-1] if self.journal.evals else None

    # --- Commandes ------------------------------------------------------------
    def stop(self):
        self.stop_requested = True

    def _poll(self, poll):
        if poll is not None:
            poll(self)
        return self.stop_requested

    # --- Adversaires ------------------------------------------------------------
    def _opponent(self):
        kind = self.cfg.opponent
        if kind == "regles":
            return RulesAgent(varied=True, seed=self.rng.random())
        if kind == "aleatoire":
            return RandomAgent(seed=self.rng.random())
        return self.agent  # autojeu : le réseau contre lui-même

    def _dataset_games(self, count):
        if self._dataset is None:
            if not self.cfg.dataset:
                raise ValueError("Aucun fichier de parties choisi pour le mode « parties ».")
            self._dataset = [g for g in load_samples(self.cfg.dataset, self.cfg.size) if g]
            if not self._dataset:
                raise ValueError("Aucune partie de taille {} dans {}.".format(
                    self.cfg.size, self.cfg.dataset))
            self._order = []
        batch = []
        for _ in range(count):
            if not self._order:
                self._order = list(range(len(self._dataset)))
                self.rng.shuffle(self._order)
            batch.append(self._dataset[self._order.pop()])
        return batch

    # --- Boucle ---------------------------------------------------------------
    def run(self, poll=None):
        cfg = self.cfg
        self.emit("hello", {"charge": self.loaded, "dossier": str(self.folder),
                            "journal": json.loads(self.journal.to_json())})
        if cfg.eval_every and not self.journal.evals:
            self.evaluate(poll)
        while not self._poll(poll):
            if cfg.max_games is not None and self.session_games >= cfg.max_games:
                break
            if cfg.max_seconds is not None and time.perf_counter() - self.started > cfg.max_seconds:
                break
            samples = self._collect(poll)
            if samples is None:
                break
            metrics = (self.brain.learn(samples) or {}) if samples else {}
            self.last_metrics = {k: float(v) for k, v in metrics.items()}
            self._accumulate_metrics()
            self._maybe_point()
            if self._next_eval is not None and self.session_games >= self._next_eval:
                self._next_eval += cfg.eval_every
                self.evaluate(poll)
            now = time.perf_counter()
            if now - self._last_stats > 0.25:
                self._send_stats()
            if now - self._last_save > cfg.save_every:
                self.save()
        self._maybe_point(force=True)
        self.save()
        self._send_stats()
        self.emit("stopped", {})

    def _collect(self, poll):
        cfg = self.cfg
        samples = []
        if cfg.opponent == "parties":
            for game_samples in self._dataset_games(cfg.games_per_update):
                for sample in game_samples:
                    sample.game = self.total_games
                samples.extend(game_samples)
                self._count_game([s.move for s in game_samples], None, len(game_samples),
                                 ("Partie enregistrée", "Partie enregistrée"))
            return samples
        opponent = self._opponent()
        selfplay = opponent is self.agent
        visible = cfg.move_delay > 0
        # Au ralenti, on apprend après chaque partie pour voir le réseau évoluer.
        count = 1 if visible else cfg.games_per_update
        for _ in range(count):
            brain_first = selfplay or self.total_games % 2 == 0
            agents = (self.agent, opponent) if brain_first else (opponent, self.agent)
            brain_player = None if selfplay else (1 if brain_first else 2)
            labels = tuple(agent.label for agent in agents)
            on_move = self._visible_move(poll, labels, brain_player) if visible else None
            record = play_game(cfg.size, *agents, swap_rule=cfg.swap_rule, collect=True,
                               on_move=on_move, should_stop=lambda: self.stop_requested)
            if record is None:
                return samples or None
            for sample in record.samples:
                sample.game = self.total_games
            samples.extend(record.samples)
            self._count_game(record.moves, record, len(record.moves), record.labels,
                             brain_player)
            if self._poll(poll):
                break
        return samples

    def _visible_move(self, poll, labels, brain_player):
        def on_move(game, decision):
            self.emit("position", {"coups": list(game.moves), "taille": game.size,
                                   "echange": self.cfg.swap_rule, "joueurs": list(labels),
                                   "reseau": brain_player, "numero": self.total_games + 1})
            deadline = time.perf_counter() + self.cfg.move_delay
            while time.perf_counter() < deadline and not self._poll(poll):
                time.sleep(min(0.02, max(0.0, deadline - time.perf_counter())))
        return on_move

    def _count_game(self, moves, record, length, labels, brain_player=None):
        self.session_games += 1
        self.session_moves += length
        window = self._window
        window["games"] += 1
        if record is not None:
            window["red"] += record.winner_color == 1
            if brain_player is not None:
                window["wins"] += record.winner_player == brain_player
        now = time.perf_counter()
        if record is not None and (self.cfg.move_delay > 0 or now - self._last_game > 0.5):
            self._last_game = now
            self.emit("game", {"coups": list(moves), "taille": self.cfg.size,
                               "echange": self.cfg.swap_rule, "joueurs": list(labels),
                               "reseau": brain_player, "numero": self.total_games})

    def _accumulate_metrics(self):
        window = self._window
        window["updates"] += 1
        for key, value in self.last_metrics.items():
            window["metrics"][key] = window["metrics"].get(key, 0.0) + value

    def _maybe_point(self, force=False):
        window = self._window
        now = time.perf_counter()
        if not window["games"] or not (force or now - self._last_point > 1.5):
            return
        self._last_point = now
        updates = max(1, window["updates"])
        point = {"parties": self.total_games,
                 "mesures": {k: v / updates for k, v in window["metrics"].items()},
                 "rouge": window["red"] / window["games"]}
        if self.cfg.opponent in ("regles", "aleatoire"):
            point["victoires"] = window["wins"] / window["games"]
        self.journal.add_point(point)
        self.emit("point", point)
        self._window = {"wins": 0, "games": 0, "red": 0, "metrics": {}, "updates": 0}

    @property
    def total_games(self):
        return self.base_games + self.session_games

    def _send_stats(self):
        self._last_stats = time.perf_counter()
        elapsed = max(1e-6, self._last_stats - self.started)
        self.emit("stats", {"parties": self.total_games, "parties_session": self.session_games,
                            "parties_par_s": self.session_games / elapsed,
                            "coups_par_s": self.session_moves / elapsed,
                            "illegaux": self.agent.illegal, "mesures": self.last_metrics,
                            "duree": elapsed})

    # --- Évaluation & sauvegarde ------------------------------------------------
    def evaluate(self, poll=None):
        """Le réseau (sans exploration) contre l'ordi à règles puis le hasard."""
        greedy = BrainAgent(self.brain, "Réseau", training=False, seed=self.rng.random())
        result = {"parties": self.total_games}
        for key, opponent in (("vs_regles", RulesAgent(varied=True, seed=self.rng.random())),
                              ("vs_aleatoire", RandomAgent(seed=self.rng.random()))):
            wins = played = 0
            for i in range(self.cfg.eval_games):
                if self._poll(poll):
                    break
                agents = (greedy, opponent) if i % 2 == 0 else (opponent, greedy)
                record = play_game(self.cfg.size, *agents, swap_rule=self.cfg.swap_rule)
                played += 1
                wins += record.winner_player == (1 if i % 2 == 0 else 2)
            if played:
                result[key] = wins / played
        if len(result) > 1:
            self.journal.add_eval(result)
            self.last_eval = result
            self.emit("eval", result)
        return result

    def save(self):
        self.journal.games = self.total_games
        folder = registry.save_checkpoint(self.brain, self.cfg.network, self.cfg.size,
                                          {"journal.json": self.journal.to_json()})
        self._last_save = time.perf_counter()
        self.emit("checkpoint", {"dossier": str(folder), "parties": self.total_games})
