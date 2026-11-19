"""Contrôles d'intégration graphiques, exécutables sans fenêtre."""

import os
import time
import unittest
from concurrent.futures import Future

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from hexgame.app import BoardLayout, HexApp


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.app = HexApp()
        self.app.draw()
        self.app.present()

    def tearDown(self):
        self.app.close()

    def click(self, logical_position):
        x, y = logical_position
        view = self.app.viewport
        scale = view.width / 1180
        position = (round(view.x + x * scale), round(view.y + y * scale))
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                               button=1, pos=position))
        self.app.draw()

    def click_button(self, key):
        self.click(self.app.buttons[key][0].center)

    def test_every_cell_center_is_clickable_in_all_sizes(self):
        for size in (3, 5, 7, 9, 11):
            layout = BoardLayout(size, pygame.Rect(52, 246, 742, 442))
            for r in range(size):
                for c in range(size):
                    self.assertEqual(layout.cell_at(layout.center(r, c)), (r, c))
            self.assertIsNone(layout.cell_at((25, 25)))

    def test_click_swap_undo_and_change_size(self):
        self.assertEqual(self.app.score, 0)
        self.click(self.app.layout.center(4, 4))
        self.assertEqual(len(self.app.game.moves), 1)
        opening_score = self.app.score
        self.assertGreater(opening_score, 0)
        self.click_button("swap")
        self.assertTrue(self.app.game.swapped)
        self.assertEqual(self.app.score, opening_score)
        self.click_button("undo")
        self.assertFalse(self.app.game.swapped)
        self.assertEqual(self.app.score, opening_score)
        self.click_button("size11")
        self.assertEqual(self.app.game.size, 9)
        self.click_button("new")
        self.assertTrue(self.app.confirm_restart)
        self.click_button("confirm_restart")
        self.assertEqual(self.app.game.size, 11)
        self.assertEqual(len(self.app.game.moves), 0)
        self.assertEqual(self.app.score, 0)

    def test_help_blocks_board_and_keyboard_moves(self):
        self.click_button("help")
        self.assertTrue(self.app.show_help)
        self.click(self.app.layout.center(4, 4))
        self.assertEqual(len(self.app.game.moves), 0)
        self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
        self.assertTrue(self.app.show_help)
        self.click_button("close_help")
        self.assertFalse(self.app.show_help)

    def test_custom_size_rejects_even_numbers_and_starts_an_odd_board(self):
        def press(key, digit=""):
            self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, unicode=digit))
            self.app.draw()

        self.click_button("custom_size")
        press(pygame.K_1, "1")
        press(pygame.K_2, "2")
        self.assertFalse(self.app.buttons["new"][1])
        self.click_button("new")
        self.assertEqual(self.app.game.size, 9)
        self.click_button("custom_size")
        press(pygame.K_BACKSPACE)
        press(pygame.K_3, "3")
        press(pygame.K_RETURN)
        self.click_button("new")
        self.assertEqual(self.app.game.size, 13)
        self.click(self.app.layout.center(12, 12))
        self.assertEqual(self.app.game.last_move, (12, 12))

    def test_custom_large_boards_and_column_labels(self):
        for size in (31, 51):
            self.click_button("custom_size")
            self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE))
            for digit in str(size):
                self.app.handle_event(pygame.event.Event(
                    pygame.KEYDOWN, key=getattr(pygame, "K_" + digit), unicode=digit))
            self.app.draw()
            self.click_button("new")
            self.assertEqual(self.app.game.size, size)
        self.click(self.app.layout.center(50, 50))
        self.assertEqual(self.app.game.last_move, (50, 50))
        self.app.mouse = (-100, -100)
        self.app.draw()

    def test_ai_panel_applies_delay_and_holds_a_ready_move_until_due(self):
        self.click_button("ai")
        self.click_button("new")
        self.click(self.app.layout.center(4, 4))
        self.app.ai_future = Future()
        self.app.ai_future.set_result(("swap", {"swap": True, "selected": "swap"}))
        self.app.ai_due = 0
        self.click_button("ai_panel")
        self.app.update()
        self.assertFalse(self.app.game.swapped)
        self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE))
        for digit in "5000":
            self.app.handle_event(pygame.event.Event(
                pygame.KEYDOWN, key=getattr(pygame, "K_" + digit), unicode=digit))
        self.app.draw()
        self.click_button("apply_delay")
        self.assertFalse(self.app.show_ai_panel)
        self.assertEqual(self.app.ai_due, self.app.placed_at + 5000)
        self.app.update()
        self.assertFalse(self.app.game.swapped)
        self.app.ai_due = 0
        self.app.update()
        self.assertTrue(self.app.game.swapped)

    def test_ai_responds_and_undo_restores_human_turn(self):
        self.click_button("ai")
        self.click_button("new")
        self.click(self.app.layout.center(4, 4))
        deadline = time.monotonic() + 5
        while self.app.ai_turn and time.monotonic() < deadline:
            self.app.ai_due = 0
            self.app.update()
            time.sleep(0.005)
        self.assertFalse(self.app.ai_turn)
        self.assertTrue(self.app.game.swapped)
        self.click_button("undo")
        self.assertEqual(len(self.app.game.moves), 0)
        self.assertEqual(self.app.game.current_player, 1)

    def test_restart_while_ai_is_working_ignores_old_result(self):
        self.app.selected_ai = True
        self.app.new_game()
        self.app.game.play(0, 0)
        self.app.ai_due = 0
        self.app.update()
        self.app.new_game()
        self.app.update()
        self.assertEqual(self.app.game.moves, [])
        self.assertIsNone(self.app.ai_future)

    def finish_ai_match(self):
        deadline = time.monotonic() + 5
        while self.app.game.winner is None and time.monotonic() < deadline:
            self.app.ai_due = 0
            self.app.update()
            self.assertFalse(self.app.ai_error)
            time.sleep(0.001)
        self.assertIsNotNone(self.app.game.winner)
        self.assertEqual(self.app.score, 100 if self.app.game.winner == 1 else -100)
        self.app.draw()
        return self.app.player_name(self.app.game.owners[self.app.game.winner])

    def test_duel_uses_two_distinct_selected_players(self):
        self.click_button("size5")
        self.click_button("duel")
        self.click_button("duel0")
        self.click_button("duel1")
        players = tuple(self.app.duel_players)
        self.assertNotEqual(*players)
        self.click_button("new")
        self.assertEqual(self.app.match_players, players)
        self.click(self.app.layout.center(2, 2))
        self.assertFalse(self.app.game.moves)
        self.assertIn(self.finish_ai_match(), players)

    def test_separate_window_shows_both_duelists_and_closes_independently(self):
        self.click_button("size5")
        self.click_button("duel")
        self.click_button("new")
        self.click_button("decision_view")
        view = self.app.decision_view
        self.assertIsNotNone(view)
        self.app.present()
        self.finish_ai_match()
        self.assertEqual(set(self.app.decisions), {"user1", "user2"})
        for action, trace in self.app.decisions.values():
            self.assertIsNotNone(action)
            self.assertTrue(trace.get(trace["selected"]))
        self.app.present()
        self.assertEqual({name for name, color in view.last_state[0]}, {"user1", "user2"})
        self.app.handle_event(pygame.event.Event(pygame.WINDOWCLOSE, window=view.window))
        self.assertIsNone(self.app.decision_view)
        self.assertTrue(self.app.running)

    def test_tournament_advances_semifinal_winners_and_keeps_board_size(self):
        self.click_button("size5")
        self.click_button("tournament")
        self.click_button("new")
        self.assertEqual(self.app.match_players, ("user1", "user4"))
        first = self.finish_ai_match()
        self.click_button("size7")
        self.click_button("next_match")
        self.assertEqual(self.app.match_players, ("user2", "user3"))
        self.assertEqual(self.app.game.size, 5)
        second = self.finish_ai_match()
        self.click_button("next_match")
        self.assertEqual(self.app.match_players, (first, second))
        champion = self.finish_ai_match()
        self.assertEqual(self.app.tournament_results, [first, second, champion])
        self.assertNotIn("next_match", self.app.buttons)

    def test_resized_window_maps_clicks_to_board(self):
        self.app.handle_event(pygame.event.Event(pygame.VIDEORESIZE,
                                               w=720, h=520, size=(720, 520)))
        self.app.present()
        self.click(self.app.layout.center(2, 7))
        self.assertEqual(self.app.game.last_move, (2, 7))
