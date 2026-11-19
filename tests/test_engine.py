import random
import unittest

from hexgame.ai import choose_action, connection_cost
from hexgame.engine import BLUE, EMPTY, RED, HexGame


class HexRulesTests(unittest.TestCase):
    def test_supported_boards_and_legal_moves(self):
        for size in (9, 11):
            with self.subTest(size=size):
                game = HexGame(size)
                self.assertEqual(len(game.legal_moves()), size ** 2)
                self.assertFalse(game.play(-1, 0))
                self.assertFalse(game.play(size, 0))
                self.assertTrue(game.play(0, 0))
                self.assertFalse(game.play(0, 0))
                self.assertEqual(game.turn, BLUE)
                self.assertEqual(len(game.moves), 1)

    def test_six_neighbors_follow_board_geometry(self):
        game = HexGame(9)
        self.assertEqual(set(game.neighbors(4, 4)),
                         {(3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4)})
        self.assertEqual(set(game.neighbors(0, 0)), {(0, 1), (1, 0)})
        self.assertEqual(len(list(game.neighbors(0, 8))), 3)

    def test_red_wins_and_undo_reopens_game(self):
        game = HexGame(9)
        for row in range(9):
            self.assertTrue(game.play(row, 0))
            if row < 8:
                self.assertTrue(game.play(row, 8))
        self.assertEqual(game.winner, RED)
        self.assertEqual(game.path, [(r, 0) for r in range(9)])
        self.assertFalse(game.play(8, 8))
        self.assertTrue(game.undo())
        self.assertIsNone(game.winner)
        self.assertEqual(game.turn, RED)
        self.assertEqual(game.board[8][0], EMPTY)
        self.assertEqual(game.path, [])

    def test_blue_wins_with_edge_connected_diagonal(self):
        game = HexGame(11)
        for i in range(11):
            self.assertTrue(game.play(0 if i < 10 else 1, i))
            self.assertTrue(game.play(10 - i, i))
        self.assertEqual(game.winner, BLUE)
        self.assertEqual(len(game.path), 11)

    def test_non_neighboring_diagonal_is_not_a_chain(self):
        game = HexGame(9)
        for i in range(9):
            game.board[i][i] = RED
        self.assertEqual(game.winning_path(RED), [])

    def test_swap_changes_owners_and_is_undoable(self):
        game = HexGame(9)
        self.assertFalse(game.swap())
        game.play(4, 4)
        self.assertEqual(game.current_player, 2)
        self.assertTrue(game.swap())
        self.assertEqual(game.owners, [0, 2, 1])
        self.assertEqual(game.current_player, 1)
        self.assertEqual(game.turn, BLUE)
        self.assertEqual(game.board[4][4], RED)
        self.assertFalse(game.swap())
        self.assertTrue(game.undo())
        self.assertTrue(game.can_swap)
        self.assertEqual(game.current_player, 2)
        game.play(1, 2)
        self.assertFalse(game.can_swap)

    def test_every_random_full_board_has_exactly_one_winner(self):
        rng = random.Random(17)
        for size in (9, 11):
            for _ in range(50):
                game = HexGame(size)
                game.board = [[rng.choice((RED, BLUE)) for _ in range(size)]
                              for _ in range(size)]
                winners = sum(bool(game.winning_path(color)) for color in (RED, BLUE))
                self.assertEqual(winners, 1)

    def test_whole_random_games_stop_on_a_valid_chain(self):
        rng = random.Random(24)
        for size in (9, 11):
            game = HexGame(size)
            cells = game.legal_moves()
            rng.shuffle(cells)
            for cell in cells:
                game.play(*cell)
                if game.winner:
                    break
            self.assertIsNotNone(game.winner)
            for previous, current in zip(game.path, game.path[1:]):
                self.assertIn(current, list(game.neighbors(*previous)))
            self.assertTrue(all(game.board[r][c] == game.winner for r, c in game.path))


class ComputerTests(unittest.TestCase):
    def test_ai_takes_immediate_win_without_mutating_board(self):
        game = HexGame(9)
        for r in range(8):
            game.play(r, 0)
            game.play(r, 8)
        before = [row[:] for row in game.board]
        self.assertEqual(choose_action(game), (8, 0))
        self.assertEqual(game.board, before)

    def test_ai_blocks_opponents_immediate_win(self):
        game = HexGame(9)
        for r in range(8):
            game.play(r, 0)
            game.play(r, 8 if r % 2 == 0 else 6)
        game.play(0, 3)
        self.assertEqual(game.turn, BLUE)
        self.assertEqual(choose_action(game), (8, 0))

    def test_ai_swap_and_normal_move_on_both_sizes(self):
        for size in (9, 11):
            game = HexGame(size)
            self.assertEqual(connection_cost(game, RED), size)
            game.play(size // 2, size // 2)
            self.assertEqual(choose_action(game), "swap")
            game.swap()
            self.assertIn(choose_action(game), game.legal_moves())

    def test_ai_returns_none_after_victory(self):
        game = HexGame(9)
        game.winner = RED
        self.assertIsNone(choose_action(game))


if __name__ == "__main__":
    unittest.main()
