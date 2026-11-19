"""Interface Pygame, dessinée sans image ni police à télécharger."""

import copy
import math
from concurrent.futures import ThreadPoolExecutor

import pygame
import pygame.gfxdraw

from IA import PLAYERS, SEMIFINALS

from .ai import choose_action, trace_decision
from .decision_view import DecisionView, column_label
from .engine import BLUE, RED, HexGame
from .evaluation import position_score

WIDTH, HEIGHT = 1180, 800
BG = (15, 22, 33)
PANEL = (22, 32, 46)
PANEL_LIGHT = (30, 43, 60)
LINE = (48, 64, 82)
TEXT = (235, 241, 246)
MUTED = (147, 165, 184)
FAINT = (102, 124, 146)
CORAL = (245, 119, 110)
AZURE = (101, 177, 249)
MINT = (158, 221, 197)
COLORS = {RED: CORAL, BLUE: AZURE}
MODES = {"local": "À deux", "ai": "Ordi à règles",
         "duel": "Duel IA", "tournament": "Tournoi IA"}
CUSTOM_SIZE_MIN = 3


def mix(first, second, amount):
    return tuple(round(a + (b - a) * amount) for a, b in zip(first, second))


def polygon(surface, color, points):
    points = [(round(x), round(y)) for x, y in points]
    pygame.gfxdraw.filled_polygon(surface, points, color)
    pygame.gfxdraw.aapolygon(surface, points, color)


def hex_points(center, radius):
    x, y = center
    return [(x + math.cos(math.radians(-90 + i * 60)) * radius,
             y + math.sin(math.radians(-90 + i * 60)) * radius) for i in range(6)]


class BoardLayout:
    """La géométrie et le clic partagent le même repère axial."""

    def __init__(self, size, rect):
        self.size = size
        self.radius = min((rect.width - 94) / (math.sqrt(3) * (1.5 * (size - 1) + 1)),
                          (rect.height - 74) / (1.5 * (size - 1) + 2))
        self.step_x = math.sqrt(3) * self.radius
        span_x = self.step_x * 1.5 * (size - 1)
        span_y = 1.5 * self.radius * (size - 1)
        self.origin = (rect.centerx - span_x / 2, rect.centery - span_y / 2)

    def center(self, row, col):
        return (self.origin[0] + self.step_x * (col + row / 2),
                self.origin[1] + self.radius * 1.5 * row)

    def cell_at(self, position):
        x, y = position
        row = round((y - self.origin[1]) / (1.5 * self.radius))
        col = round((x - self.origin[0]) / self.step_x - row / 2)
        if not (0 <= row < self.size and 0 <= col < self.size):
            return None
        cx, cy = self.center(row, col)
        dx, dy = abs(x - cx), abs(y - cy)
        radius = self.radius * 0.94
        if dx <= math.sqrt(3) * radius / 2 and dy <= radius - dx / math.sqrt(3):
            return row, col
        return None


class HexApp:
    window_title = "HEX — Jeu de connexion"

    def __init__(self, size=9, versus_ai=False):
        pygame.display.init()
        pygame.font.init()
        desktop = pygame.display.get_desktop_sizes()[0]
        window = (min(WIDTH, max(820, desktop[0] - 100)),
                  min(HEIGHT, max(560, desktop[1] - 100)))
        self.screen = pygame.display.set_mode(window, pygame.RESIZABLE)
        pygame.display.set_caption(self.window_title)
        icon = pygame.Surface((48, 48), pygame.SRCALPHA)
        polygon(icon, MINT, hex_points((24, 24), 21))
        polygon(icon, BG, hex_points((24, 24), 12))
        pygame.display.set_icon(icon)
        self.closed = False
        self.canvas = pygame.Surface((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.fonts = {}
        self.font_path = pygame.font.match_font("arial,dejavusans,freesans")
        self.bold_path = pygame.font.match_font("arial,dejavusans,freesans", bold=True)
        self.game = HexGame(size)
        self.score = position_score(self.game)
        self.selected_size = size
        self.custom_size_text = ""
        self.custom_size_selected = False
        self.editing_size = False
        self.versus_ai = versus_ai
        self.selected_ai = versus_ai
        self.selected_arena = None
        self.duel_players = ["user1", "user2"]
        self.match_players = None
        self.tournament_results = None
        self.ai_error = ""
        self.layout = BoardLayout(size, pygame.Rect(52, 246, 742, 442))
        self.buttons = {}
        self.mouse = (-100, -100)
        self.running = True
        self.show_help = False
        self.show_ai_panel = False
        self.ai_delay = 400
        self.delay_text = ""
        self.confirm_restart = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hex-ai")
        self.ai_future = None
        self.decision_view = None
        self.decisions = {}
        self.ai_due = 0
        self.notice = ""
        self.notice_until = 0
        self.started_at = pygame.time.get_ticks()
        self.finished_at = None
        self.placed_at = 0
        self.viewport = pygame.Rect(0, 0, WIDTH, HEIGHT)

    def font(self, size, bold=False):
        key = size, bold
        if key not in self.fonts:
            self.fonts[key] = pygame.font.Font(self.bold_path if bold else self.font_path, size)
            if bold and self.bold_path is None:
                self.fonts[key].set_bold(True)
        return self.fonts[key]

    def text(self, value, x, y, size=16, color=TEXT, bold=False, center=False):
        rendered = self.font(size, bold).render(value, True, color)
        rect = rendered.get_rect()
        if center:
            rect.center = (x, y)
        else:
            rect.topleft = (x, y)
        self.canvas.blit(rendered, rect)
        return rect

    def panel(self, rect, color=PANEL, radius=20, border=True):
        pygame.draw.rect(self.canvas, color, rect, border_radius=radius)
        if border:
            pygame.draw.rect(self.canvas, LINE, rect, width=1, border_radius=radius)

    def button(self, key, rect, label, selected=False, primary=False, enabled=True):
        rect = pygame.Rect(rect)
        self.buttons[key] = (rect, enabled)
        hover = rect.collidepoint(self.mouse) and enabled
        color = MINT if primary else (PANEL_LIGHT if selected or hover else PANEL)
        if not enabled:
            color = mix(PANEL, BG, 0.3)
        self.panel(rect, color, 10, border=not primary)
        if selected and not primary:
            pygame.draw.rect(self.canvas, MINT, rect, width=1, border_radius=10)
        foreground = BG if primary else (MINT if selected else TEXT)
        if not enabled:
            foreground = FAINT
        self.text(label, rect.centerx, rect.centery, 15, foreground, True, center=True)

    def player_name(self, player):
        if self.match_players:
            return self.match_players[player - 1]
        if self.versus_ai:
            return "Vous" if player == 1 else "Ordinateur"
        return "Joueur {}".format(player)

    @property
    def ai_turn(self):
        return self.game.winner is None and (
            self.match_players is not None
            or (self.versus_ai and self.game.current_player == 2)
        )

    def toast(self, message):
        self.notice = message
        self.notice_until = pygame.time.get_ticks() + 3200

    def new_game(self):
        self.versus_ai = self.selected_ai
        self.tournament_results = [] if self.selected_arena == "tournament" else None
        self.match_players = {
            "duel": tuple(self.duel_players),
            "tournament": SEMIFINALS[0],
        }.get(self.selected_arena)
        self.start_board(self.selected_size)

    def start_board(self, size):
        if self.ai_future is not None:
            self.ai_future.cancel()
            self.ai_future = None
        self.game = HexGame(size)
        self.decisions.clear()
        self.score = position_score(self.game)
        self.layout = BoardLayout(self.game.size, pygame.Rect(52, 246, 742, 442))
        self.started_at = pygame.time.get_ticks()
        self.finished_at = None
        self.confirm_restart = False
        self.ai_due = 0
        self.ai_error = ""
        self.notice = ""

    def next_match(self):
        results = self.tournament_results
        if not results or len(results) >= 3 or self.game.winner is None:
            return
        self.match_players = SEMIFINALS[1] if len(results) == 1 else tuple(results)
        self.start_board(self.game.size)

    def cycle_duel_player(self, slot):
        candidates = [name for name in PLAYERS if name != self.duel_players[1 - slot]]
        current = candidates.index(self.duel_players[slot])
        self.duel_players[slot] = candidates[(current + 1) % len(candidates)]

    @property
    def custom_size(self):
        value = int(self.custom_size_text or "0")
        if value >= CUSTOM_SIZE_MIN and value % 2 == 1:
            return value
        return None

    def request_restart(self):
        self.editing_size = False
        if self.custom_size_selected:
            size = self.custom_size
            if size is None:
                return
            self.selected_size = size
        if self.game.moves and self.game.winner is None:
            self.confirm_restart = True
        else:
            self.new_game()

    def after_action(self):
        self.score = position_score(self.game)
        self.placed_at = pygame.time.get_ticks()
        self.ai_due = self.placed_at + self.ai_delay
        if self.game.winner is not None:
            self.finished_at = self.placed_at
            if self.tournament_results is not None:
                self.tournament_results.append(self.player_name(self.game.owners[self.game.winner]))

    def undo(self):
        if self.match_players:
            return
        if self.ai_future is not None:
            self.ai_future.cancel()
            self.ai_future = None
        if not self.game.undo():
            return
        # En solo, revenir à une décision humaine, y compris après un échange.
        while self.versus_ai and self.game.current_player == 2 and self.game.history:
            self.game.undo()
        self.score = position_score(self.game)
        self.finished_at = None
        self.decisions.clear()
        self.placed_at = pygame.time.get_ticks()
        self.ai_due = self.placed_at + self.ai_delay
        self.toast("Dernier tour annulé.")

    def update(self):
        now = pygame.time.get_ticks()
        if (not self.ai_turn or self.show_help or self.show_ai_panel
                or self.confirm_restart or self.ai_error):
            return
        name = self.player_name(self.game.current_player)
        if self.ai_future is None and now >= self.ai_due:
            strategy = PLAYERS[name] if self.match_players else choose_action
            self.ai_future = self.executor.submit(trace_decision, strategy, copy.deepcopy(self.game))
        if now >= self.ai_due and self.ai_future is not None and self.ai_future.done():
            future = self.ai_future
            self.ai_future = None
            try:
                action, trace = future.result()
            except Exception:
                self.ai_error = "{} a rencontré une erreur.".format(name)
                return
            if isinstance(action, str) and action == "swap" and self.game.swap():
                self.toast("{} prend les rouges.".format(name))
            elif (isinstance(action, tuple) and len(action) == 2
                  and all(type(coord) is int for coord in action)
                  and action in self.game.legal_moves()):
                self.game.play(*action)
            else:
                self.ai_error = "{} a proposé un coup invalide.".format(name)
                return
            self.decisions[name] = (action, trace)
            self.after_action()

    def activate(self, key):
        if key == "help":
            self.show_help = True
        elif key == "decision_view" and self.decision_view is None:
            try:
                self.decision_view = DecisionView(self.font)
            except pygame.error:
                self.toast("Impossible d’ouvrir la fenêtre de réflexion.")
        elif key == "ai_panel":
            self.delay_text = str(self.ai_delay)
            self.show_ai_panel = True
        elif key == "close_ai_panel":
            self.show_ai_panel = False
        elif key == "apply_delay" and self.delay_text:
            self.ai_delay = int(self.delay_text)
            if self.game.history:
                self.ai_due = self.placed_at + self.ai_delay
            self.show_ai_panel = False
        elif key == "close_help":
            self.show_help = False
        elif key == "cancel_restart":
            self.confirm_restart = False
        elif key == "confirm_restart":
            self.new_game()
        elif key in ("size5", "size7", "size9", "size11"):
            self.selected_size = int(key[4:])
            self.custom_size_selected = False
        elif key == "custom_size":
            self.custom_size_selected = True
            self.editing_size = True
        elif key in MODES:
            self.selected_ai = key == "ai"
            self.selected_arena = key if key in ("duel", "tournament") else None
        elif key in ("duel0", "duel1"):
            self.cycle_duel_player(int(key[-1]))
        elif key == "next_match":
            self.next_match()
        elif key == "new":
            self.request_restart()
        elif key == "undo":
            self.undo()
        elif key == "swap" and not self.ai_turn and self.game.swap():
            self.after_action()
            self.toast("Couleurs échangées : {} joue les bleus.".format(
                self.player_name(self.game.current_player)))

    def logical_mouse(self, position):
        scale = self.viewport.width / WIDTH
        return ((position[0] - self.viewport.x) / scale,
                (position[1] - self.viewport.y) / scale)

    def edit_size(self, event):
        digit = getattr(event, "unicode", "")
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.custom_size is not None:
                self.editing_size = False
        elif event.key == pygame.K_ESCAPE:
            self.custom_size_selected = False
            self.editing_size = False
        elif event.key == pygame.K_BACKSPACE:
            self.custom_size_text = self.custom_size_text[:-1]
        elif event.key == pygame.K_DELETE:
            self.custom_size_text = ""
        elif digit and digit in "0123456789":
            self.custom_size_text += digit

    def edit_delay(self, event):
        digit = getattr(event, "unicode", "")
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.activate("apply_delay")
        elif event.key == pygame.K_ESCAPE:
            self.activate("close_ai_panel")
        elif event.key == pygame.K_BACKSPACE:
            self.delay_text = self.delay_text[:-1]
        elif event.key == pygame.K_DELETE:
            self.delay_text = ""
        elif digit and digit in "0123456789":
            self.delay_text += digit

    def handle_event(self, event):
        window = getattr(event, "window", None)
        if (self.decision_view is not None
                and getattr(window, "id", window) == self.decision_view.window.id):
            if event.type == pygame.WINDOWCLOSE or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.close_decision_view()
            return
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode((max(400, event.w), max(300, event.h)),
                                                  pygame.RESIZABLE)
        elif event.type == pygame.KEYDOWN:
            if self.show_ai_panel:
                self.edit_delay(event)
            elif self.editing_size:
                self.edit_size(event)
            elif event.key == pygame.K_ESCAPE:
                self.show_help = False
                self.confirm_restart = False
            elif self.show_help or self.confirm_restart:
                if event.key == pygame.K_RETURN and self.confirm_restart:
                    self.new_game()
            elif event.key in (pygame.K_h, pygame.K_F1):
                self.show_help = True
            elif event.key == pygame.K_n:
                self.request_restart()
            elif event.key in (pygame.K_u, pygame.K_BACKSPACE):
                self.undo()
            elif event.key == pygame.K_e:
                self.activate("swap")
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.mouse = self.logical_mouse(event.pos)
            self.editing_size = False
            for key, (rect, enabled) in self.buttons.items():
                if enabled and rect.collidepoint(self.mouse):
                    self.activate(key)
                    return
            if self.show_help or self.show_ai_panel or self.confirm_restart or self.ai_turn:
                return
            cell = self.layout.cell_at(self.mouse)
            if cell and self.game.play(*cell):
                self.after_action()

    def draw_header(self):
        polygon(self.canvas, MINT, hex_points((48, 51), 21))
        polygon(self.canvas, BG, hex_points((48, 51), 12))
        self.text("HEX", 84, 22, 39, bold=True)
        self.text("Chaque pierre crée un lien.", 85, 72, 15, MUTED)
        self.button("decision_view", (658, 33, 160, 40), "Réflexion des IA",
                    enabled=self.decision_view is None)
        self.button("ai_panel", (832, 33, 154, 40), "Délai IA")
        self.button("help", (998, 33, 154, 40), "Comment jouer ?")
        pygame.draw.line(self.canvas, LINE, (28, 105), (1152, 105))

    def draw_board_edges(self):
        """Quatre rives en escalier, chacune le long des hexagones extérieurs."""
        size, radius = self.game.size, self.layout.radius
        top, bottom, left, right = [], [], [], []
        for i in range(size):
            pts = hex_points(self.layout.center(0, i), radius * 1.055)
            top.extend((pts[5], pts[0], pts[1]))
            pts = hex_points(self.layout.center(size - 1, i), radius * 1.055)
            bottom.extend((pts[4], pts[3], pts[2]))
            pts = hex_points(self.layout.center(i, 0), radius * 1.055)
            left.extend((pts[5], pts[4], pts[3]))
            pts = hex_points(self.layout.center(i, size - 1), radius * 1.055)
            right.extend((pts[0], pts[1], pts[2]))
        for points, color in ((top, CORAL), (bottom, CORAL), (left, AZURE), (right, AZURE)):
            pygame.draw.lines(self.canvas, mix(color, PANEL, 0.20), False, points, 4)
            pygame.draw.aalines(self.canvas, color, False, points)

    def draw_evaluation(self):
        bar = pygame.Rect(429, 176, 250, 10)
        red_width = round(bar.width * (100 + self.score) / 200)
        pygame.draw.rect(self.canvas, AZURE, bar)
        if red_width:
            pygame.draw.rect(self.canvas, CORAL, (bar.x, bar.y, red_width, bar.height))
        pygame.draw.line(self.canvas, BG, bar.midtop, (bar.centerx, bar.bottom - 1))
        if self.game.winner is not None:
            label = "+100 · Victoire rouge" if self.game.winner == RED else "−100 · Victoire bleue"
        elif self.score == 0:
            label = "Évaluation  0 · Équilibre"
        else:
            color = "rouge" if self.score > 0 else "bleu"
            label = "Évaluation  {:+d} · Avantage {}".format(self.score, color)
        self.text(label, bar.centerx, 157, 14, TEXT, True, center=True)

    def draw_board(self):
        self.panel((28, 122, 790, 624))
        self.text("LE PLATEAU", 52, 143, 12, MUTED, True)
        self.text("Reliez vos deux rives.", 52, 165, 22, bold=True)
        self.draw_evaluation()
        self.panel((710, 143, 82, 34), PANEL_LIGHT, 9, False)
        self.text("{} × {}".format(self.game.size, self.game.size), 751, 160,
                  16, MINT, True, True)
        color = self.game.winner or self.game.turn
        pygame.draw.circle(self.canvas, COLORS[color], (61, 210), 4)
        if self.game.winner:
            message = "{} remporte la partie !".format(self.player_name(self.game.owners[color]))
        elif self.ai_turn:
            message = self.ai_error or "{} réfléchit…".format(
                self.player_name(self.game.current_player))
        else:
            name = self.player_name(self.game.current_player)
            message = "À vous de jouer." if name == "Vous" else "{} : à vous de jouer.".format(name)
        self.text(message, 74, 199, 15, COLORS[color])
        self.draw_board_edges()
        hover = self.layout.cell_at(self.mouse)
        path = set(self.game.path)
        for r in range(self.game.size):
            for c in range(self.game.size):
                center = self.layout.center(r, c)
                value = self.game.board[r][c]
                radius = self.layout.radius * 0.91
                fill = (37, 52, 70) if (r + c) % 2 == 0 else (35, 49, 66)
                if value:
                    fill = mix(COLORS[value], PANEL, 0.06 if (r, c) in path else 0.22)
                elif hover == (r, c) and not self.ai_turn and not self.game.winner:
                    fill = mix(COLORS[self.game.turn], PANEL, 0.57)
                pts = hex_points(center, radius)
                polygon(self.canvas, fill, pts)
                outline = mix(fill, TEXT, 0.12 if value else 0.05)
                pygame.gfxdraw.aapolygon(self.canvas, [(round(x), round(y)) for x, y in pts], outline)
                if (r, c) == self.game.last_move:
                    pygame.draw.circle(self.canvas, mix(TEXT, fill, 0.18),
                                       (round(center[0]), round(center[1])), 4)
                if hover == (r, c) and not value and not self.ai_turn and not self.game.winner:
                    self.text("+", center[0], center[1] - 1, 21,
                              COLORS[self.game.turn], center=True)
        if self.game.path:
            points = [self.layout.center(r, c) for r, c in self.game.path]
            pygame.draw.lines(self.canvas, (255, 242, 222), False, points, 3)
            for x, y in points:
                pygame.draw.circle(self.canvas, (255, 242, 222), (round(x), round(y)), 4)
        # Repères : lettres des colonnes, numéros des lignes.
        label_step = max(1, math.ceil(22 / (1.5 * self.layout.radius)))
        for i in range(0, self.game.size, label_step):
            x, y = self.layout.center(0, i)
            self.text(column_label(i), x, y - self.layout.radius - 17, 12, FAINT, center=True)
            x, y = self.layout.center(i, 0)
            self.text(str(i + 1), x - self.layout.step_x * 0.5 - 22, y,
                      12, FAINT, center=True)
        cell_label = ""
        if hover:
            cell_label = "{}{}".format(column_label(hover[1]), hover[0] + 1)
        elif self.game.last_move:
            row, col = self.game.last_move
            cell_label = "Dernier coup : {}{}".format(column_label(col), row + 1)
        self.text(cell_label, 651, 711, 13, FAINT)
        if self.tournament_results is not None:
            results = self.tournament_results
            semifinals = [
                "{} / {} → {}".format(first, second, results[i] if i < len(results) else "…")
                for i, (first, second) in enumerate(SEMIFINALS)
            ]
            summary = "Demi-finales   " + "  ·  ".join(semifinals)
            if len(results) == 3:
                summary += "  ·  Champion : " + results[-1]
            self.text(summary, 52, 711, 14, MINT, True)

    def draw_player(self, color, y):
        player = self.game.owners[color]
        active = (self.game.winner or self.game.turn) == color
        rect = pygame.Rect(866, y, 266, 66)
        self.panel(rect, mix(COLORS[color], PANEL, 0.90) if active else PANEL,
                   12, border=False)
        if active:
            pygame.draw.rect(self.canvas, mix(COLORS[color], PANEL, 0.40), rect,
                             width=1, border_radius=12)
        polygon(self.canvas, COLORS[color], hex_points((890, y + 32), 11))
        self.text(self.player_name(player), 912, y + 11, 17, TEXT, True)
        direction = "Haut → bas" if color == RED else "Gauche → droite"
        self.text(direction, 912, y + 36, 13, MUTED)
        if active:
            pygame.draw.circle(self.canvas, COLORS[color], (1115, y + 32), 4)

    def draw_sidebar(self):
        self.panel((842, 122, 310, 624))
        self.text("LA PARTIE", 866, 143, 12, MUTED, True)
        self.draw_player(RED, 175)
        self.draw_player(BLUE, 249)
        now = self.finished_at if self.finished_at is not None else pygame.time.get_ticks()
        seconds = (now - self.started_at) // 1000
        self.text("{:02d}:{:02d}".format(seconds // 60, seconds % 60), 866, 337,
                  21, TEXT, True)
        self.text("TEMPS", 866, 367, 10, FAINT, True)
        self.text(str(len(self.game.moves)), 1046, 337, 21, TEXT, True)
        self.text("COUPS", 1046, 367, 10, FAINT, True)
        pygame.draw.line(self.canvas, LINE, (866, 397), (1128, 397))
        self.text("PROCHAINE PARTIE", 866, 418, 11, MUTED, True)
        self.text("Taille du plateau", 866, 442, 14, TEXT)
        for i, size in enumerate((5, 7, 9, 11)):
            self.button("size" + str(size), (866 + i * 67, 462, 61, 30),
                        "{}×{}".format(size, size),
                        not self.custom_size_selected and self.selected_size == size)
        value = self.custom_size_text[-5:] or "…"
        if self.editing_size:
            value = self.custom_size_text[-5:] + "|"
        self.button("custom_size", (866, 498, 62, 28), value, self.custom_size_selected)
        invalid = self.custom_size_selected and self.custom_size is None
        hint = "Saisissez un impair ≥ 3" if invalid else "Taille libre · impair ≥ 3"
        self.text(hint, 936, 504, 12, CORAL if invalid else MUTED)
        selected_mode = self.selected_arena or ("ai" if self.selected_ai else "local")
        for i, (mode, label) in enumerate(MODES.items()):
            self.button(mode, (866 + i % 2 * 135, 532 + i // 2 * 34, 127, 29),
                        label, mode == selected_mode)
        if self.selected_arena == "duel":
            for slot, player in enumerate(self.duel_players):
                self.button("duel" + str(slot), (866 + slot * 135, 600, 127, 30),
                            "{} ↻".format(player))
        self.button("new", (866, 638, 262, 40), "Nouvelle partie", primary=True,
                    enabled=not invalid)
        if self.tournament_results and self.game.winner and len(self.tournament_results) < 3:
            label = "Seconde demi-finale" if len(self.tournament_results) == 1 else "Jouer la finale"
            self.button("next_match", (866, 686, 262, 38), label)
        elif not self.match_players:
            self.button("undo", (866, 686, 262, 38), "Annuler le dernier tour",
                        enabled=bool(self.game.history))

    def draw_footer(self):
        if self.game.can_swap:
            self.text("Règle d'échange : le second joueur peut prendre les rouges.",
                      30, 766, 13, MUTED)
            self.button("swap", (630, 756, 188, 33), "Échanger les couleurs", enabled=not self.ai_turn)
        elif self.game.winner:
            if self.tournament_results and len(self.tournament_results) < 3:
                message = "Demi-finale terminée. Le prochain match est disponible."
            else:
                message = "Chaîne gagnante complétée. Lancez une nouvelle partie pour rejouer."
            self.text(message, 30, 766, 13, MINT)
        else:
            self.text("ROUGE  haut ↔ bas", 30, 766, 12, CORAL, True)
            self.text("BLEU  gauche ↔ droite", 232, 766, 12, AZURE, True)
        self.text("HEX  /  {} CASES".format(self.game.size ** 2), 1015, 766, 11, FAINT, True)
        if self.notice and pygame.time.get_ticks() < self.notice_until:
            self.panel((102, 694, 530, 42), (39, 62, 67), 10)
            self.text(self.notice, 367, 715, 14, MINT, center=True)

    def draw_ai_panel(self):
        self.panel((290, 267, 600, 262), PANEL, 22)
        self.text("Délai IA", 320, 290, 28, bold=True)
        self.text("Délai avant le prochain coup IA (ms)", 320, 348, 16, bold=True)
        self.panel((320, 378, 140, 37), PANEL_LIGHT, 8)
        self.text(self.delay_text[-10:] + "|", 334, 386, 17)
        self.text("0 = sans attente · Suppr efface le champ", 475, 387, 15, MUTED)
        self.button("close_ai_panel", (320, 450, 250, 44), "Fermer")
        self.button("apply_delay", (590, 450, 270, 44), "Appliquer", primary=True,
                    enabled=bool(self.delay_text))

    def draw_modal(self):
        veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        veil.fill((4, 10, 17, 210))
        self.canvas.blit(veil, (0, 0))
        # Seuls les boutons de la fenêtre modale restent interactifs.
        self.buttons.clear()
        if self.show_ai_panel:
            self.draw_ai_panel()
        elif self.confirm_restart:
            self.panel((320, 261, 540, 278), PANEL, 22)
            self.text("Recommencer la partie ?", 350, 290, 26, TEXT, True)
            self.text("La partie en cours sera remplacée par un plateau", 350, 345, 17, MUTED)
            self.text("{} × {}, {}.".format(self.selected_size, self.selected_size,
                      MODES[self.selected_arena or ("ai" if self.selected_ai else "local")]),
                      350, 373, 17, MUTED)
            self.button("cancel_restart", (350, 451, 220, 46), "Continuer la partie")
            self.button("confirm_restart", (590, 451, 240, 46), "Recommencer", primary=True)
        else:
            self.panel((267, 133, 646, 554), PANEL, 22)
            self.text("Une règle. Mille chemins.", 302, 161, 29, TEXT, True)
            self.text("COMMENT JOUER AU HEX", 303, 211, 12, MINT, True)
            lines = [
                ("01", "À tour de rôle, cliquez sur une case vide."),
                ("02", "Rouge relie le haut au bas du plateau."),
                ("03", "Bleu relie la gauche à la droite du plateau."),
                ("04", "Une chaîne de cases voisines de votre couleur gagne."),
            ]
            for i, (number, message) in enumerate(lines):
                y = 250 + i * 41
                self.text(number, 303, y, 15, MINT, True)
                self.text(message, 343, y, 16, TEXT)
            self.text("Après le premier coup, le second joueur peut échanger", 303, 433, 15, MUTED)
            self.text("les couleurs. La pierre reste en place ; bleu joue ensuite.", 303, 457, 15, MUTED)
            self.text("N : nouvelle partie  ·  U : annuler  ·  E : échanger  ·  H : aide",
                      303, 578, 13, FAINT)
            self.button("close_help", (303, 621, 574, 43), "C'est parti", primary=True)

    def draw(self):
        self.buttons.clear()
        self.canvas.fill(BG)
        self.draw_header()
        self.draw_board()
        self.draw_sidebar()
        self.draw_footer()
        if self.show_help or self.show_ai_panel or self.confirm_restart:
            self.draw_modal()

    def present(self):
        sw, sh = self.screen.get_size()
        scale = min(sw / WIDTH, sh / HEIGHT)
        size = (max(1, round(WIDTH * scale)), max(1, round(HEIGHT * scale)))
        self.viewport = pygame.Rect((sw - size[0]) // 2, (sh - size[1]) // 2, *size)
        self.screen.fill(BG)
        scaled = pygame.transform.smoothscale(self.canvas, size)
        self.screen.blit(scaled, self.viewport)
        pygame.display.flip()
        if self.decision_view is not None:
            players = tuple((self.player_name(self.game.owners[color]), COLORS[color])
                            for color in (RED, BLUE) if self.match_players
                            or (self.versus_ai and self.game.owners[color] == 2))
            active = self.player_name(self.game.current_player) if self.ai_turn else None
            self.decision_view.present(players, active, self.decisions)

    def close_decision_view(self):
        if self.decision_view is not None:
            view = self.decision_view
            self.decision_view = None
            view.close()

    def run(self):
        try:
            self.draw()
            self.present()
            while self.running:
                for event in pygame.event.get():
                    self.handle_event(event)
                    if not self.running:
                        break
                if not self.running:
                    break
                self.mouse = self.logical_mouse(pygame.mouse.get_pos())
                self.update()
                self.draw()
                self.present()
                self.clock.tick(60)
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.close_decision_view()
        self.executor.shutdown(wait=True)
        pygame.quit()
