"""Contrôles d'intégration graphiques, exécutables sans fenêtre."""

import os
import time
import unittest

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
        self.click(self.app.layout.center(4, 4))
        self.assertEqual(len(self.app.game.moves), 1)
        self.click_button("swap")
        self.assertTrue(self.app.game.swapped)
        self.click_button("undo")
        self.assertFalse(self.app.game.swapped)
        self.click_button("size11")
        self.assertEqual(self.app.game.size, 9)
        self.click_button("new")
        self.assertTrue(self.app.confirm_restart)
        self.click_button("confirm_restart")
        self.assertEqual(self.app.game.size, 11)
        self.assertEqual(len(self.app.game.moves), 0)

    def test_help_blocks_board_and_keyboard_moves(self):
        self.click_button("help")
        self.assertTrue(self.app.show_help)
        self.click(self.app.layout.center(4, 4))
        self.assertEqual(len(self.app.game.moves), 0)
        self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
        self.assertTrue(self.app.show_help)
        self.click_button("close_help")
        self.assertFalse(self.app.show_help)

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

    def test_resized_window_maps_clicks_to_board(self):
        self.app.handle_event(pygame.event.Event(pygame.VIDEORESIZE,
                                               w=720, h=520, size=(720, 520)))
        self.app.present()
        self.click(self.app.layout.center(2, 7))
        self.assertEqual(self.app.game.last_move, (2, 7))
