"""Tests de l'entraîneur : fidélité aux règles du jeu, contrat des réseaux, boucle, interface."""

import os
import random
import tempfile
import time
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from hexgame import ai
from hexgame.engine import BLUE, RED, HexGame
from hextrainer import registry, rules
from hextrainer.agents import BrainAgent, RandomAgent, RulesAgent, play_game
from hextrainer.api import Observation
from hextrainer.board import SWAP, FastHex, move_label, parse_move
from hextrainer.dataset import generate, read_games, replay_samples
from hextrainer.session import Session, TrainConfig


class TempCheckpoints:
    """Redirige les sauvegardes vers un dossier temporaire (processus fils compris)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("HEXTRAINER_SAUVEGARDES")
        os.environ["HEXTRAINER_SAUVEGARDES"] = self._tmp.name

    def tearDown(self):
        if self._old is None:
            os.environ.pop("HEXTRAINER_SAUVEGARDES", None)
        else:
            os.environ["HEXTRAINER_SAUVEGARDES"] = self._old
        self._tmp.cleanup()


def random_positions(count, seed=7, sizes=(3, 5, 7, 9)):
    rng = random.Random(seed)
    for _ in range(count):
        size = rng.choice(sizes)
        game, fast = HexGame(size), FastHex(size)
        for _ in range(rng.randrange(size * size)):
            if game.winner:
                break
            if game.can_swap and rng.random() < 0.3:
                game.swap()
                fast.swap()
                continue
            move = rng.choice(game.legal_moves())
            game.play(*move)
            fast.play(*move)
        yield game, fast


class FastBoardTests(unittest.TestCase):
    def test_same_state_and_winner_as_the_game_engine(self):
        for game, fast in random_positions(150):
            self.assertEqual(game.board, fast.grid())
            self.assertEqual((game.turn, game.owners, game.winner),
                             (fast.turn, fast.owners, fast.winner))
            self.assertEqual(game.can_swap, fast.can_swap)
            if game.winner:
                self.assertEqual(len(game.path), len(fast.winning_path()))

    def test_would_win_matches_a_real_move(self):
        for _, fast in random_positions(80):
            for i in fast.legal_indices():
                trial = fast.clone()
                trial.play_index(i)
                self.assertEqual(fast.would_win(i, fast.turn), trial.winner is not None)

    def test_move_labels_round_trip(self):
        for move in [(0, 0), (6, 3), (10, 10), SWAP]:
            self.assertEqual(parse_move(move_label(move)), move)
        self.assertEqual(move_label((3, 1)), "B4")


class RulesTests(unittest.TestCase):
    def test_fast_rules_choose_exactly_like_the_game_ai(self):
        checked = 0
        for game, fast in random_positions(120, seed=3):
            if game.winner:
                continue
            expected = ai.choose_action(game)
            actual = rules.choose_action(fast)
            self.assertEqual(expected, "swap" if actual == SWAP else actual)
            checked += 1
        self.assertGreater(checked, 80)

    def test_connection_cost_matches(self):
        for game, fast in random_positions(60, seed=11):
            for color in (RED, BLUE):
                self.assertEqual(ai.connection_cost(game, color),
                                 rules.connection_cost(fast, color))


class ObservationTests(unittest.TestCase):
    def test_blue_view_is_transposed(self):
        game = FastHex(5)
        game.play(1, 3)  # rouge
        obs = Observation.from_game(game)
        self.assertEqual(obs.to_play, BLUE)
        planes = obs.planes()
        self.assertEqual(planes.shape, (3, 5, 5))
        self.assertEqual(planes[1, 3, 1], 1.0)  # pierre adverse, transposée
        self.assertEqual(obs.cell_to_real(obs.cell_to_canonical((2, 4))), (2, 4))
        self.assertTrue(obs.can_swap)

    def test_from_hexgame_after_swap(self):
        game = HexGame(5)
        game.play(2, 2)
        game.swap()
        obs = Observation.from_game(game)
        self.assertEqual(obs.moves, [(2, 2), SWAP])
        self.assertFalse(obs.can_swap)


class NetworkTests(TempCheckpoints, unittest.TestCase):
    def test_networks_are_discovered_and_the_template_is_hidden(self):
        names = registry.discover()
        self.assertIn("exemple_mlp", names)
        self.assertNotIn("_modele", names)

    def test_template_plays_legal_moves(self):
        brain = registry.create("_modele", 5)
        record = play_game(5, BrainAgent(brain), RandomAgent(1))
        self.assertIn(record.winner_player, (1, 2))

    def test_example_learns_explains_saves_and_loads(self):
        brain = registry.create("exemple_mlp", 5)
        record = play_game(5, BrainAgent(brain, training=True), RulesAgent(seed=1),
                           collect=True)
        metrics = brain.learn(record.samples)
        self.assertIn("perte_politique", metrics)
        obs = Observation.from_game(FastHex(5))
        explanation = brain.explain(obs)
        self.assertAlmostEqual(float(explanation.policy.sum()), 1.0, places=4)
        by_name = {layer.name: layer for layer in explanation.layers}
        for previous, layer in zip(explanation.layers, explanation.layers[1:]):
            source = by_name[layer.source] if layer.source else previous
            self.assertEqual(layer.weights.shape,
                             (np.size(layer.values), np.size(source.values)))
        registry.save_checkpoint(brain, "exemple_mlp", 5)
        other = registry.create("exemple_mlp", 5)
        self.assertTrue(registry.load_checkpoint(other, "exemple_mlp", 5))
        np.testing.assert_allclose(other.explain(obs).policy, explanation.policy)

    def test_samples_are_numbered_in_order(self):
        record = play_game(5, RandomAgent(1), RandomAgent(2), collect=True)
        self.assertEqual([s.ply for s in record.samples], list(range(len(record.moves))))

    def test_samples_outcome_follows_the_swap(self):
        for seed in range(10):
            record = play_game(5, RulesAgent(seed=seed, opening_noise=1.0),
                               RulesAgent(seed=seed + 1), collect=True)
            for sample in record.samples:
                self.assertEqual(sample.color_outcome > 0,
                                 sample.obs.to_play == record.winner_color)
                if sample.move == SWAP:  # Le décideur prend les rouges.
                    self.assertEqual(sample.outcome > 0, record.winner_color == RED)
                elif not record.swapped:
                    self.assertEqual(sample.outcome, sample.color_outcome)


class DatasetTests(unittest.TestCase):
    def test_generate_read_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, written = generate(5, 4, path=os.path.join(tmp, "p.jsonl"), seed=2)
            games = list(read_games(path))
            self.assertEqual(written, 4)
            self.assertEqual(len(games), 4)
            for game in games:
                samples = replay_samples(5, game["coups"])
                self.assertEqual(len(samples), len(game["coups"]))
                final = FastHex.from_moves(5, game["coups"])
                self.assertEqual({1: "rouge", 2: "bleu"}[final.winner], game["gagnant"])


class SessionTests(TempCheckpoints, unittest.TestCase):
    def run_session(self, opponent, **extra):
        events = []
        config = TrainConfig("exemple_mlp", 5, opponent, max_games=16, games_per_update=8,
                             eval_every=8, eval_games=2, **extra)
        session = Session(config, lambda kind, data: events.append(kind))
        session.run()
        return session, events

    def test_every_opponent_trains_evaluates_and_saves(self):
        for opponent in ("autojeu", "regles", "aleatoire"):
            session, events = self.run_session(opponent)
            self.assertGreaterEqual(session.session_games, 16)
            for kind in ("hello", "eval", "checkpoint", "stopped"):
                self.assertIn(kind, events)
        self.assertTrue((registry.checkpoint_dir("exemple_mlp", 5) / "poids.npz").exists())
        # La session suivante reprend le compteur et les courbes.
        session, _ = self.run_session("autojeu")
        self.assertTrue(session.loaded)
        self.assertGreaterEqual(session.total_games, 64)

    def test_dataset_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = generate(5, 5, path=os.path.join(tmp, "p.jsonl"), seed=4)
            session, events = self.run_session("parties", dataset=str(path))
            self.assertGreaterEqual(session.session_games, 16)


class TrainerInterfaceTests(TempCheckpoints, unittest.TestCase):
    def setUp(self):
        super().setUp()
        from hextrainer.ui import TrainerApp
        self.app = TrainerApp(size=5)
        self.app.draw()
        self.app.present()

    def tearDown(self):
        self.app.close()
        super().tearDown()

    def test_sandbox_shows_the_network_reasoning(self):
        app = self.app
        app.activate("sandbox_mode")
        app.sandbox_moves = [(2, 2)]
        app.activate("net_move")
        self.assertEqual(len(app.sandbox_moves), 2)
        app.draw()
        self.assertGreater(len(app.layer_views), 2)
        app.mouse = tuple(app.layer_views[1].centers[0])
        app.draw()
        self.assertEqual(app.hover_node, (1, 0))

    def wait_for_network(self, count):
        deadline = time.time() + 5
        while time.time() < deadline and len(self.app.sandbox_moves) < count:
            self.app.update()
            time.sleep(0.02)

    def test_duel_the_network_replies_by_itself(self):
        app = self.app
        app.activate("versus")
        app.activate("versus")  # Vous jouez bleu : le réseau commence.
        self.assertEqual(app.versus, 2)
        self.wait_for_network(1)
        self.assertEqual(len(app.sandbox_moves), 1)
        game = app.view_game()
        human = next(m for m in game.legal_moves())
        app.sandbox_moves.append(human)
        self.wait_for_network(3)
        self.assertEqual(len(app.sandbox_moves), 3)
        app.draw()
        app.activate("sb_undo")  # Revient à votre tour.
        self.assertEqual(app.view_game().current_player, 2)
        self.assertFalse(app.network_turn)

    def test_size_choices_including_custom_odd_sizes(self):
        app = self.app
        app.activate("size9")
        self.assertEqual(app.size, 9)
        app.activate("custom")
        app.activate("custom_plus")
        self.assertEqual(app.size % 2, 1)
        self.assertNotIn(app.size, (5, 7, 9, 11))
        app.draw()

    def test_training_runs_in_another_process(self):
        app = self.app
        app.activate("opp_aleatoire")
        app.activate("start")
        self.assertTrue(app.training)
        deadline = time.time() + 30
        while time.time() < deadline and not app.stats.get("parties"):
            app.update()
            time.sleep(0.05)
        app.draw()
        self.assertGreater(app.stats.get("parties", 0), 0)
        app.activate("start")  # pause : arrêt et sauvegarde
        self.assertIsNone(app.worker)
        self.assertIsNone(app.error)


if __name__ == "__main__":
    unittest.main()
