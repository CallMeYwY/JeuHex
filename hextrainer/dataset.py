"""Parties pré-jouées : format, lecture, rejeu et génération.

Une partie par ligne dans un fichier `.jsonl` du dossier `parties/` :

    {"taille": 7, "coups": ["D4", "echange", "C3", ...], "gagnant": "rouge",
     "joueurs": ["regles", "regles"]}

Les coups utilisent la notation de l'interface (colonne en lettre, ligne en
chiffre) ; « echange » est la règle d'échange.
"""

import json
import random
from pathlib import Path

from .agents import RulesAgent, play_game
from .api import Observation, Sample
from .board import FastHex, move_label, parse_move
from .registry import DATASETS_DIR

COLOR_NAMES = {1: "rouge", 2: "bleu"}


def list_datasets(size=None):
    """Fichiers `.jsonl` disponibles, avec le nombre de parties à la bonne taille."""
    if not DATASETS_DIR.is_dir():
        return []
    found = []
    for path in sorted(DATASETS_DIR.glob("*.jsonl")):
        count = sum(1 for game in read_games(path) if size is None or game["taille"] == size)
        if count:
            found.append((path, count))
    return found


def read_games(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            game = json.loads(line)
            game["coups"] = [parse_move(m) if isinstance(m, str) else tuple(m)
                             for m in game["coups"]]
            yield game


def record_to_json(record, players=None):
    return json.dumps({"taille": record.size,
                       "coups": [move_label(m) for m in record.moves],
                       "gagnant": COLOR_NAMES[record.winner_color],
                       "joueurs": list(players or record.labels)}, ensure_ascii=False)


def replay_samples(size, moves, swap_rule=True, actor="partie"):
    """Rejoue une partie et renvoie un `Sample` par décision, issue comprise."""
    game = FastHex(size, swap_rule)
    pending = []
    for move in moves:
        obs = Observation.from_game(game)
        pending.append((obs, move, game.current_player))
        if not game.apply(move):
            raise ValueError("Partie invalide au coup {}".format(move_label(move)))
    if game.winner is None:
        return []
    winner = game.owners[game.winner]
    return [Sample(obs, move, 1.0 if player == winner else -1.0, actor,
                   color_outcome=1.0 if obs.to_play == game.winner else -1.0, ply=ply)
            for ply, (obs, move, player) in enumerate(pending)]


def load_samples(path, size):
    """Liste de parties, chacune sous forme de liste de `Sample`."""
    return [replay_samples(size, g["coups"]) for g in read_games(path) if g["taille"] == size]


def generate(size, count, path=None, seed=None, progress=None, should_stop=None):
    """Fait jouer l'ordi à règles contre lui-même (avec un peu de hasard)."""
    DATASETS_DIR.mkdir(exist_ok=True)
    path = Path(path) if path else DATASETS_DIR / "regles_{}.jsonl".format(size)
    rng = random.Random(seed)
    written = 0
    with open(path, "a", encoding="utf-8") as handle:
        for i in range(count):
            if should_stop is not None and should_stop():
                break
            a = RulesAgent(varied=True, seed=rng.random(), opening_noise=0.7)
            b = RulesAgent(varied=True, seed=rng.random(), opening_noise=0.7)
            record = play_game(size, a, b, swap_rule=True)
            handle.write(record_to_json(record, ["regles", "regles"]) + "\n")
            written += 1
            if progress is not None:
                progress(written, count)
    return path, written
