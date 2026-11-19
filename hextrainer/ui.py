"""Interface de l'entraîneur, dans le style du jeu et avec ses composants.

Trois colonnes :
  * le plateau : dernière partie d'entraînement (rejouable coup par coup) ou bac
    à sable, avec la politique du réseau en surimpression ;
  * le réseau : ses couches, ses activations, et ce que « regarde » chaque neurone ;
  * la session : réglages, lancement, statistiques.
L'entraînement tourne dans un autre processus (`worker.py`) pour rester rapide.
"""

import math
import traceback

import numpy as np
import pygame

from hexgame.app import (AZURE, BG, COLORS, CORAL, FAINT, LINE, MINT, MUTED, PANEL,
                         PANEL_LIGHT, TEXT, BoardLayout, hex_points, mix)
from hexgame.engine import BLUE, EMPTY, RED

from . import registry
from .api import Explanation, Observation, is_legal, normalize_decision
from .board import SWAP, FastHex, cell_label, move_label
from .dataset import list_datasets
from .display import HiDpiWindow
from .session import OPPONENTS, Journal, TrainConfig
from .worker import Worker

CW, CH = 1440, 900
CELL = (37, 52, 70)
POS = MINT
NEG = (232, 128, 178)
AMBER = (240, 190, 110)
SIZES = (5, 7, 9, 11)
MIN_SIZE, MAX_SIZE = 3, 25
SPEEDS = (("Turbo", 0.0), ("Rapide", 0.06), ("Lent", 0.4))
OPPONENT_HINTS = {
    "autojeu": ("Le réseau joue contre lui-même :", "il progresse avec son adversaire."),
    "regles": ("Contre l'ordi à règles du jeu,", "un professeur dont il peut imiter les coups."),
    "aleatoire": ("Contre un joueur qui joue au hasard :", "idéal pour vérifier les bases."),
    "parties": ("Rejoue des parties enregistrées", "(dossier parties/, format .jsonl)."),
}
BOARD_PANEL = pygame.Rect(28, 122, 520, 518)
CHART_PANEL = pygame.Rect(28, 656, 520, 216)
NET_PANEL = pygame.Rect(564, 122, 540, 750)
SIDE_PANEL = pygame.Rect(1120, 122, 292, 750)
DIAGRAM = pygame.Rect(580, 262, 508, 432)


def shade(value, scale, base=CELL):
    """Vert pour le positif, rose pour le négatif, plus vif quand c'est fort."""
    if scale <= 0 or abs(value) < 1e-9:
        return base
    t = min(1.0, abs(value) / scale)
    return mix(base, POS if value > 0 else NEG, 0.12 + 0.88 * math.sqrt(t))


def mini_layout(n_rows, n_cols, rect):
    """Rayon et origine pour un petit plateau en losange, comme le grand."""
    radius = min(rect.width / (math.sqrt(3) * (n_cols - 1 + (n_rows - 1) / 2 + 1)),
                 rect.height / (1.5 * (n_rows - 1) + 2))
    step = math.sqrt(3) * radius
    span_x = step * (n_cols - 1 + (n_rows - 1) / 2)
    span_y = 1.5 * radius * (n_rows - 1)
    origin = (rect.centerx - span_x / 2, rect.centery - span_y / 2)
    centers = np.array([(origin[0] + step * (c + r / 2), origin[1] + 1.5 * radius * r)
                        for r in range(n_rows) for c in range(n_cols)])
    return radius, centers


def fmt_int(value):
    return "{:,}".format(int(value)).replace(",", " ")


class LayerView:
    """Position à l'écran de chaque valeur d'une couche (pour dessiner et survoler)."""

    def __init__(self, layer, values, centers, size, kind, slots=()):
        self.layer = layer
        self.values = values          # tableau d'origine (forme conservée)
        self.flat = values.reshape(-1)
        self.centers = centers        # (M, 2) : M premières valeurs aplaties
        self.size = size              # taille d'un élément
        self.kind = kind              # "nodes", "single" ou "boards"
        self.slots = slots            # cartes : (nom, rect)
        self.scale = float(np.abs(self.flat).max()) if self.flat.size else 1.0

    def hit(self, point):
        if not len(self.centers):
            return None
        d = np.hypot(self.centers[:, 0] - point[0], self.centers[:, 1] - point[1])
        i = int(d.argmin())
        return i if d[i] <= max(self.size * 0.62, 3) else None


class TrainerApp:
    window_title = "HEX — Centre d'entraînement"

    def __init__(self, size=7, network=None):
        icon = pygame.Surface((48, 48), pygame.SRCALPHA)
        for radius, color in ((21, MINT), (12, BG)):
            pygame.draw.polygon(icon, color, hex_points((24, 24), radius))
        self.screen = HiDpiWindow(self.window_title, (CW, CH), icon)
        self.d = self.screen.painter  # Toutes les primitives de dessin, à l'échelle.
        self.font_path = pygame.font.match_font("arial,dejavusans,freesans")
        self.bold_path = pygame.font.match_font("arial,dejavusans,freesans", bold=True)
        self.mono_path = pygame.font.match_font("menlo,monaco,dejavusansmono,couriernew")
        self.clock = pygame.time.Clock()
        self.buttons = {}
        self.mouse = (-100, -100)
        self.running = True
        self.closed = False
        self.show_help = False
        self.notice, self.notice_until = "", 0
        self.networks = registry.discover()
        self.network = network if network in self.networks else (
            "exemple_mlp" if "exemple_mlp" in self.networks else
            (self.networks[0] if self.networks else None))
        self.size = size
        self.custom_size = size if size not in SIZES else 13
        self.opponent = "autojeu"
        self.swap_rule = True
        self.speed = 0
        self.datasets, self.dataset_index = [], 0
        self.worker = None
        self.task = None
        self.stats = {}
        self.progress = None
        self.error = None
        self.journal = Journal()
        # Plateau affiché.
        self.sandbox = False
        self.follow = True
        self.view_moves, self.view_cursor, self.view_info = [], 0, None
        self.view_swap = True
        self.last_game = None
        self.sandbox_moves = []
        self.versus = None      # Bac à sable contre l'IA : numéro du joueur humain (1 ou 2)
        self.auto_due = None    # Instant du prochain coup automatique du réseau
        self._view_cache = None
        # Réseau côté interface (pour l'affichage) et son explication.
        self.brain, self.brain_error, self.brain_loaded = None, None, False
        self.brain_version = 0
        self._explain_cache = (None, None)
        self.layer_views = []
        self.hover_node = None
        self.board_layout = None
        self.load_brain()
        self.refresh_datasets()

    # --- Affichage générique (mêmes composants que le jeu, rendus nets) --------
    def font(self, size, bold=False):
        return self.d.font(self.bold_path if bold else self.font_path, size, bold)

    def text(self, value, x, y, size=16, color=TEXT, bold=False, center=False):
        rendered = self.font(size, bold).render(value, True, color)
        return self.d.blit_text(rendered, "center" if center else "topleft", (x, y))

    def text_right(self, value, x, y, size=16, color=TEXT, bold=False):
        rendered = self.font(size, bold).render(value, True, color)
        return self.d.blit_text(rendered, "topright", (x, y))

    def mono_text(self, value, x, y, color=MUTED):
        rendered = self.d.font(self.mono_path, 12).render(value, True, color)
        return self.d.blit_text(rendered, "topleft", (x, y))

    def fit(self, value, size, width, bold=False):
        font, limit = self.font(size, bold), width * self.d.scale
        if font.size(value)[0] <= limit:
            return value
        while value and font.size(value + "…")[0] > limit:
            value = value[:-1]
        return value + "…"

    def panel(self, rect, color=PANEL, radius=20, border=True):
        self.d.rect(color, rect, border_radius=radius)
        if border:
            self.d.rect(LINE, rect, width=1, border_radius=radius)

    def button(self, key, rect, label, selected=False, primary=False, enabled=True):
        rect = pygame.Rect(rect)
        self.buttons[key] = (rect, enabled)
        hover = rect.collidepoint(self.mouse) and enabled
        color = MINT if primary else (PANEL_LIGHT if selected or hover else PANEL)
        if not enabled:
            color = mix(PANEL, BG, 0.3)
        self.panel(rect, color, 10, border=not primary)
        if selected and not primary:
            self.d.rect(MINT, rect, width=1, border_radius=10)
        foreground = BG if primary else (MINT if selected else TEXT)
        if not enabled:
            foreground = FAINT
        self.text(label, rect.centerx, rect.centery, 15, foreground, True, center=True)

    def toast(self, message):
        self.notice = message
        self.notice_until = pygame.time.get_ticks() + 3200

    def logical_mouse(self, position):
        return self.screen.logical_mouse(position)

    def present(self):
        self.screen.present(BG)

    def draw_board_edges(self, game, layout):
        """Les quatre rives colorées, comme dans le jeu."""
        size, radius = game.size, layout.radius
        top, bottom, left, right = [], [], [], []
        for i in range(size):
            pts = hex_points(layout.center(0, i), radius * 1.055)
            top.extend((pts[5], pts[0], pts[1]))
            pts = hex_points(layout.center(size - 1, i), radius * 1.055)
            bottom.extend((pts[4], pts[3], pts[2]))
            pts = hex_points(layout.center(i, 0), radius * 1.055)
            left.extend((pts[5], pts[4], pts[3]))
            pts = hex_points(layout.center(i, size - 1), radius * 1.055)
            right.extend((pts[0], pts[1], pts[2]))
        for points, color in ((top, CORAL), (bottom, CORAL), (left, AZURE), (right, AZURE)):
            self.d.lines(mix(color, PANEL, 0.20), False, points, 4)
            self.d.aalines(color, False, points)

    def run(self):
        try:
            while self.running:
                for event in pygame.event.get():
                    self.handle_event(event)
                    if not self.running:
                        break
                if not self.running:
                    break
                self.mouse = self.logical_mouse(pygame.mouse.get_pos())
                self.update()
                self.screen.layout()
                self.draw()
                self.present()
                self.clock.tick(60)
        finally:
            self.close()

    # --- Réseau côté interface ------------------------------------------------
    @property
    def training(self):
        return self.worker is not None and self.task == "train"

    def load_brain(self, reload=False):
        self.brain, self.brain_error, self.brain_loaded = None, None, False
        self.journal = Journal()
        self.brain_version += 1
        if self.network is None:
            return
        try:
            self.brain = registry.create(self.network, self.size, reload=reload)
            self.brain_loaded = registry.load_checkpoint(self.brain, self.network, self.size)
        except Exception:
            self.brain = None
            self.brain_error = traceback.format_exc()
        self.journal = Journal.load(registry.checkpoint_dir(self.network, self.size))

    def reload_weights(self):
        if self.brain is None:
            return
        try:
            self.brain_loaded = registry.load_checkpoint(self.brain, self.network, self.size)
            self.brain_version += 1
        except Exception:
            pass  # Sauvegarde en cours d'écriture : on réessaiera à la suivante.

    def refresh_datasets(self):
        self.datasets = list_datasets(self.size)
        self.dataset_index = min(self.dataset_index, max(0, len(self.datasets) - 1))

    def settings_changed(self, reload=False):
        self.sandbox_moves = []
        self.view_moves, self.view_cursor, self.view_info = [], 0, None
        self.last_game = None
        self.stats = {}
        self.load_brain(reload)
        self.refresh_datasets()

    # --- Plateau affiché ------------------------------------------------------
    def view_game(self):
        if self.sandbox:
            key = ("bac", tuple(self.sandbox_moves), self.size, self.swap_rule)
            moves, size, swap = self.sandbox_moves, self.size, self.swap_rule
        else:
            size = self.view_info["taille"] if self.view_info else self.size
            moves = self.view_moves[:self.view_cursor]
            key = ("vue", tuple(moves), size, self.view_swap)
            swap = self.view_swap
        if self._view_cache is None or self._view_cache[0] != key:
            try:
                game = FastHex.from_moves(size, moves, swap)
            except ValueError:
                game = FastHex(size, swap)
            self._view_cache = (key, game)
        return self._view_cache[1]

    def show_game(self, data):
        self.view_info = data
        self.view_moves = list(map(lambda m: tuple(m) if isinstance(m, list) else m,
                                   data["coups"]))
        self.view_swap = data.get("echange", True)
        self.view_cursor = len(self.view_moves)

    def player_names(self, game):
        if self.sandbox and self.versus:
            return {color: "Vous" if game.owners[color] == self.versus else "Le réseau"
                    for color in (RED, BLUE)}
        if self.sandbox or not self.view_info or "joueurs" not in self.view_info:
            return {RED: "Rouge", BLUE: "Bleu"}
        names = self.view_info["joueurs"]
        return {color: names[game.owners[color] - 1] for color in (RED, BLUE)}

    # --- Explication du réseau ------------------------------------------------
    def explained_game(self):
        """La position expliquée : celle affichée, ou celle d'avant le coup gagnant."""
        game = self.view_game()
        if game.winner is not None and game.moves:
            return FastHex.from_moves(game.size, game.moves[:-1], game.swap_rule), True
        return game, False

    def explanation(self):
        self.view_game()
        key = (self.brain_version, self._view_cache[0])
        if self._explain_cache[0] == key:
            return self._explain_cache[1]
        game, before_end = self.explained_game()
        result = None
        if self.brain is not None and game.size == self.brain.size and game.winner is None:
            obs = Observation.from_game(game)
            try:
                result = self.brain.explain(obs) if hasattr(self.brain, "explain") else None
                if result is None:
                    decision = normalize_decision(self.brain.act(obs), obs)
                    result = Explanation(policy=decision.policy, value=decision.value)
            except Exception:
                result = Explanation(notes=["Erreur dans explain() :",
                                            traceback.format_exc().strip().splitlines()[-1]])
            if before_end:
                result.notes = ["Partie finie : voici son raisonnement juste avant le coup "
                                "gagnant."] + list(result.notes)
        self._explain_cache = (key, (result, game))
        return result, game

    # --- Processus d'entraînement ------------------------------------------------
    def config(self):
        dataset = None
        if self.opponent == "parties" and self.datasets:
            dataset = str(self.datasets[self.dataset_index][0])
        return TrainConfig(network=self.network, size=self.size, opponent=self.opponent,
                           dataset=dataset, swap_rule=self.swap_rule, eval_every=500,
                           eval_games=20, save_every=3.0, move_delay=SPEEDS[self.speed][1])

    def start_training(self):
        if self.worker is not None:
            return
        if self.brain is None:
            self.error = self.brain_error or "Aucun réseau dans le dossier reseaux/."
            return
        if self.opponent == "parties" and not self.datasets:
            self.toast("Aucune partie {0}×{0} : générez-en d'abord.".format(self.size))
            return
        self.sandbox = False
        self.follow = True
        self.worker = Worker.train(self.config())
        self.task = "train"
        self.toast("Entraînement lancé : {}.".format(OPPONENTS[self.opponent].lower()))

    def stop_worker(self):
        if self.worker is None:
            return
        for event in self.worker.stop():
            self.handle_worker_event(*event)
        self.worker, self.task, self.progress = None, None, None

    def start_generation(self, count=200):
        if self.worker is not None:
            return
        self.worker = Worker.generate(self.size, count)
        self.task = "generate"
        self.progress = (0, count)

    def handle_worker_event(self, kind, data):
        if kind == "hello":
            self.journal = Journal(data["journal"]["points"], data["journal"]["evaluations"],
                                   data["journal"]["parties"])
        elif kind == "stats":
            self.stats = data
        elif kind == "point":
            self.journal.add_point(data)
        elif kind == "eval":
            self.journal.add_eval(data)
        elif kind in ("game", "position"):
            if self.follow:
                self.show_game(data)
            self.last_game = data
        elif kind == "checkpoint":
            self.journal.games = data["parties"]
            self.reload_weights()
        elif kind == "progress":
            self.progress = (data["fait"], data["total"])
        elif kind == "generated":
            self.toast("{} parties ajoutées à parties/{}.".format(
                data["parties"], data["fichier"].split("/")[-1]))
            self.refresh_datasets()
        elif kind == "error":
            self.error = data

    @property
    def network_turn(self):
        """Vrai quand, en duel contre l'IA, c'est au réseau de jouer."""
        if not (self.sandbox and self.versus) or self.brain is None:
            return False
        game = self.view_game()
        return (game.winner is None and game.size == self.brain.size
                and game.current_player != self.versus)

    def update_versus(self):
        if not self.network_turn or self.show_help or self.error:
            self.auto_due = None
            return
        now = pygame.time.get_ticks()
        if self.auto_due is None:
            self.auto_due = now + 350  # Petit délai : on voit le coup arriver.
        elif now >= self.auto_due:
            self.auto_due = None
            self.network_move(announce=False)

    def cycle_versus(self):
        self.versus = {None: 1, 1: 2, 2: None}[self.versus]
        self.sandbox = True
        self.auto_due = None
        if self.versus:
            self.sandbox_moves = []
            self.toast("Duel : {}. Le réseau répond tout seul.".format(
                "vous commencez" if self.versus == 1 else "le réseau commence"))

    def update(self):
        self.update_versus()
        if self.worker is None:
            return
        for event in self.worker.poll():
            self.handle_worker_event(*event)
        if self.worker.finished:
            self.worker, self.task, self.progress = None, None, None
            self.refresh_datasets()

    # --- Actions ------------------------------------------------------------------
    def select_size(self, size):
        size = max(MIN_SIZE, min(MAX_SIZE, size))
        if size % 2 == 0:
            size += 1
        if size != self.size:
            self.size = size
            self.settings_changed()

    def cycle_network(self, step):
        self.networks = registry.discover()
        if not self.networks:
            self.network = None
        else:
            index = self.networks.index(self.network) if self.network in self.networks else 0
            self.network = self.networks[(index + step) % len(self.networks)]
        self.settings_changed()

    def step_view(self, target):
        self.follow = False
        self.view_cursor = max(0, min(len(self.view_moves), target))

    def network_move(self, announce=True):
        game = self.view_game()
        if self.brain is None or game.winner is not None or game.size != self.brain.size:
            return
        obs = Observation.from_game(game)
        try:
            decision = normalize_decision(self.brain.act(obs), obs)
        except Exception:
            self.error = traceback.format_exc()
            return
        if not is_legal(decision.move, obs):
            self.toast("Le réseau propose un coup illégal : {}.".format(move_label(decision.move)))
            if self.versus:  # En duel, on ne bloque pas la partie pour autant.
                decision.move = obs.legal_moves()[0]
            else:
                return
        self.sandbox_moves.append(decision.move)
        if decision.move == SWAP and self.versus:
            self.toast("Le réseau échange : il prend les rouges, vous jouez les bleus.")
        elif announce:
            self.toast("Le réseau joue {}.".format(
            "l'échange" if decision.move == SWAP else cell_label(*decision.move)))

    def activate(self, key):
        locked = self.worker is not None
        if key == "help":
            self.show_help = True
        elif key == "close_help":
            self.show_help = False
        elif key == "close_error":
            self.error = None
        elif key == "start":
            self.stop_worker() if self.training else self.start_training()
        elif key == "cancel_task":
            self.stop_worker()
        elif key.startswith("speed"):
            self.speed = int(key[5:])
            if self.training:
                self.worker.send("delay", SPEEDS[self.speed][1])
        elif key == "follow_mode":
            self.sandbox = False
        elif key == "sandbox_mode":
            self.sandbox = True
        elif key == "first":
            self.step_view(0)
        elif key == "prev":
            self.step_view(self.view_cursor - 1)
        elif key == "next":
            self.step_view(self.view_cursor + 1)
        elif key == "last":
            self.step_view(len(self.view_moves))
        elif key == "live":
            self.follow = True
            if self.last_game:
                self.show_game(self.last_game)
        elif key == "net_move":
            self.network_move()
        elif key == "sb_swap" and self.view_game().can_swap and not self.network_turn:
            self.sandbox_moves.append(SWAP)
        elif key == "sb_undo" and self.sandbox_moves:
            self.sandbox_moves.pop()
            # En duel, revenir jusqu'à votre tour (comme « Annuler » dans le jeu).
            while self.versus and self.sandbox_moves and self.network_turn:
                self.sandbox_moves.pop()
            self.auto_due = None
        elif key == "versus":
            self.cycle_versus()
        elif key == "sb_clear":
            self.sandbox_moves = []
        elif key == "reload":
            self.networks = registry.discover()
            self.settings_changed(reload=True)
            self.toast("Réseau rechargé depuis reseaux/{}.".format(self.network)
                       if self.brain else "Le réseau n'a pas pu être chargé.")
        elif locked:
            return
        elif key in ("net_prev", "net_next"):
            self.cycle_network(-1 if key == "net_prev" else 1)
        elif key.startswith("size"):
            self.select_size(int(key[4:]))
        elif key == "custom":
            self.select_size(self.custom_size)
        elif key in ("custom_minus", "custom_plus"):
            self.custom_size = max(MIN_SIZE, min(MAX_SIZE, self.custom_size
                                                 + (2 if key == "custom_plus" else -2)))
            self.select_size(self.custom_size)
        elif key.startswith("opp_"):
            self.opponent = key[4:]
        elif key in ("data_prev", "data_next") and self.datasets:
            self.dataset_index = (self.dataset_index + (1 if key == "data_next" else -1)) \
                % len(self.datasets)
        elif key == "generate":
            self.start_generation()
        elif key == "swap_rule":
            self.swap_rule = not self.swap_rule

    def handle_event(self, event):
        if event.type in (pygame.QUIT, pygame.WINDOWCLOSE):
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.show_help = False
                self.error = None
            elif self.show_help or self.error:
                return
            elif event.key in (pygame.K_h, pygame.K_F1):
                self.show_help = True
            elif event.key == pygame.K_SPACE:
                self.activate("start")
            elif event.key == pygame.K_LEFT:
                self.activate("sb_undo" if self.sandbox else "prev")
            elif event.key == pygame.K_RIGHT and not self.sandbox:
                self.activate("next")
            elif event.key == pygame.K_b:
                self.sandbox = not self.sandbox
            elif event.key == pygame.K_j and self.sandbox:
                self.network_move()
            elif event.key == pygame.K_a:
                self.cycle_versus()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.mouse = self.logical_mouse(event.pos)
            for key, (rect, enabled) in list(self.buttons.items()):
                if enabled and rect.collidepoint(self.mouse):
                    self.activate(key)
                    return
            if self.show_help or self.error or self.board_layout is None:
                return
            cell = self.board_layout.cell_at(self.mouse)
            game = self.view_game()
            if cell is None or game.winner is not None or game.at(*cell) != EMPTY:
                return
            if self.network_turn:
                return  # Le réseau réfléchit.
            if not self.sandbox:
                # Explorer à partir de la position affichée.
                self.sandbox = True
                self.sandbox_moves = list(self.view_moves[:self.view_cursor]) \
                    if game.size == self.size else []
                self.toast("Bac à sable : vous jouez à partir de cette position.")
                if game.size != self.size:
                    return
            self.sandbox_moves.append(cell)

    # --- Dessin : en-tête -------------------------------------------------------
    def draw_header(self):
        self.d.polygon(MINT, hex_points((48, 51), 21))
        self.d.polygon(BG, hex_points((48, 51), 12))
        self.text("HEX", 84, 22, 39, bold=True)
        self.text("Centre d'entraînement des réseaux de neurones.", 85, 72, 15, MUTED)
        self.text("ENTRAÎNEUR", 700, 43, 12, FAINT, True, center=True)
        self.button("help", (1222, 33, 190, 40), "Comment ça marche ?")
        self.d.line(LINE, (28, 105), (1412, 105))

    # --- Dessin : plateau ---------------------------------------------------------
    def draw_board_panel(self):
        self.panel(BOARD_PANEL)
        game = self.view_game()
        names = self.player_names(game)
        self.text("LE PLATEAU", 52, 143, 12, MUTED, True)
        self.button("follow_mode", (318, 136, 102, 30), "Entraînement", not self.sandbox)
        self.button("sandbox_mode", (426, 136, 102, 30), "Bac à sable", self.sandbox)
        if self.sandbox:
            title = "Bac à sable"
        elif self.view_info:
            title = "Partie n° {}".format(fmt_int(self.view_info.get("numero", 0)))
        else:
            title = "Aucune partie pour l'instant"
        self.text(title, 52, 165, 21, bold=True)
        versus_label = {None: "Affronter l'IA (A)", 1: "Duel : vous commencez",
                        2: "Duel : l'IA commence"}[self.versus]
        self.button("versus", (318, 172, 210, 30), versus_label, bool(self.versus and self.sandbox),
                    enabled=self.brain is not None)
        color = game.winner or game.turn
        self.d.circle(COLORS[color], (61, 217), 4)
        if game.winner and self.sandbox and self.versus:
            message = "Vous gagnez !" if names[color] == "Vous" else "Le réseau gagne."
        elif self.network_turn:
            message = "Le réseau réfléchit…"
        elif self.sandbox and self.versus:
            message = "À vous de jouer."
        elif game.winner:
            message = "{} gagne.".format(names[color])
        elif not self.sandbox and not self.view_info:
            message = "Lancez l'entraînement ou passez au bac à sable."
        else:
            message = "{} au trait.".format(names[color])
        if game.can_swap:
            message += " Échange possible."
        self.text(self.fit(message, 14, 290), 74, 208, 14, COLORS[color])
        layout = BoardLayout(game.size, pygame.Rect(40, 228, 496, 364))
        self.board_layout = layout
        explanation, _ = self.explanation()
        # Sur une partie finie, la politique concerne la position d'avant : on ne la peint pas.
        policy = explanation.policy if explanation is not None and game.winner is None else None
        hover = layout.cell_at(self.mouse)
        self.draw_main_board(game, layout, policy, hover)
        # Survol : nom de la case et probabilité selon le réseau.
        if hover:
            label = cell_label(*hover)
            if policy is not None and game.at(*hover) == EMPTY:
                label += " · {:.1f} %".format(100 * float(policy[hover]))
            self.text_right(label, 528, 209, 13, MUTED)
        y = 600
        if self.sandbox:
            self.button("net_move", (44, y, 150, 32), "Le réseau joue (J)",
                        enabled=self.brain is not None and game.winner is None)
            self.button("sb_swap", (200, y, 104, 32), "Échanger",
                        enabled=game.can_swap and not self.network_turn)
            self.button("sb_undo", (310, y, 104, 32), "Annuler", enabled=bool(self.sandbox_moves))
            self.button("sb_clear", (420, y, 108, 32), "Vider", enabled=bool(self.sandbox_moves))
        else:
            total = len(self.view_moves)
            self.button("first", (44, y, 40, 32), "<<", enabled=self.view_cursor > 0)
            self.button("prev", (88, y, 40, 32), "<", enabled=self.view_cursor > 0)
            self.button("next", (132, y, 40, 32), ">", enabled=self.view_cursor < total)
            self.button("last", (176, y, 40, 32), ">>", enabled=self.view_cursor < total)
            self.text("Coup {} / {}".format(self.view_cursor, total), 232, y + 8, 14, MUTED)
            self.button("live", (398, y, 130, 32), "En direct", self.follow)

    def draw_main_board(self, game, layout, policy, hover):
        self.draw_board_edges(game, layout)
        path = set(game.winning_path())
        interactive = game.winner is None
        top = set()
        pmax = 0.0
        if policy is not None:
            empty = np.asarray(game.cells).reshape(game.size, game.size) == EMPTY
            flat = np.where(empty, policy, 0.0)
            # Une politique presque uniforme reste pâle : seul un vrai choix ressort.
            pmax = max(float(flat.max()), 4.0 / max(1, int(empty.sum())))
            order = np.argsort(flat, axis=None)[::-1][:3]
            top = {divmod(int(i), game.size) for i in order if flat.flat[i] > 0.005}
        label_size = max(9, min(13, round(layout.radius * 0.62)))
        for r in range(game.size):
            for c in range(game.size):
                center = layout.center(r, c)
                value = game.at(r, c)
                fill = CELL if (r + c) % 2 == 0 else (35, 49, 66)
                if value:
                    fill = mix(COLORS[value], PANEL, 0.06 if (r, c) in path else 0.22)
                elif policy is not None and pmax > 0:
                    t = min(1.0, max(0.0, float(policy[r, c]) / pmax))
                    fill = mix(fill, MINT, 0.75 * t)
                if hover == (r, c) and not value and interactive:
                    fill = mix(fill, COLORS[game.turn], 0.45)
                pts = hex_points(center, layout.radius * 0.91)
                self.d.polygon(fill, pts)
                self.d.aapolygon(mix(fill, TEXT, 0.12 if value else 0.05), pts)
                if (r, c) == game.last_move:
                    self.d.circle(mix(TEXT, fill, 0.18),
                                       (round(center[0]), round(center[1])), 4)
                elif (r, c) in top and layout.radius > 11:
                    self.text("{:.0f}".format(100 * float(policy[r, c])), center[0], center[1],
                              label_size, BG, True, center=True)
        if path:
            points = [layout.center(r, c) for r, c in game.winning_path()]
            if len(points) > 1:
                self.d.lines((255, 242, 222), False, points, 3)
            for x, y in points:
                self.d.circle((255, 242, 222), (round(x), round(y)), 4)
        if game.size <= 15:
            for i in range(game.size):
                x, y = layout.center(0, i)
                self.text(chr(65 + i), x, y - layout.radius - 15, 11, FAINT, center=True)
                x, y = layout.center(i, 0)
                self.text(str(i + 1), x - layout.step_x * 0.5 - 18, y, 11, FAINT, center=True)

    # --- Dessin : courbes ------------------------------------------------------------
    def draw_charts(self):
        self.panel(CHART_PANEL)
        self.text("PROGRÈS", 52, 674, 12, MUTED, True)
        evals = self.journal.evals
        series = []
        for key, label, color in (("vs_regles", "vs règles", MINT),
                                  ("vs_aleatoire", "vs hasard", AMBER)):
            pts = [(e["parties"], e[key]) for e in evals if key in e]
            series.append((label, color, pts))
        self.draw_chart(pygame.Rect(44, 698, 236, 150), "Victoires en évaluation",
                        series, percent=True)
        points = self.journal.points
        keys = []
        for point in points:
            for key in point.get("mesures", {}):
                if key not in keys:
                    keys.append(key)
        colors = (MINT, AMBER, NEG, AZURE)
        series = [(key.replace("_", " "), colors[i % 4],
                   [(p["parties"], p["mesures"][key]) for p in points if key in p.get("mesures", {})])
                  for i, key in enumerate(keys[:4])]
        if not series and points:
            series = [("rouge gagne", CORAL, [(p["parties"], p.get("rouge", 0)) for p in points])]
        self.draw_chart(pygame.Rect(296, 698, 236, 150), "Mesures de learn()", series,
                        normalize=True)

    def draw_chart(self, rect, title, series, percent=False, normalize=False):
        self.text(title, rect.x, rect.y, 13, TEXT, True)
        legend_y = rect.y + 20
        x = rect.x
        for label, color, pts in series:
            if not pts:
                continue
            last = pts[-1][1]
            text = "{} {}".format(label, "{:.0f} %".format(100 * last) if percent
                                  else "{:.3g}".format(last))
            self.d.circle(color, (x + 4, legend_y + 7), 3)
            width = self.text(text, x + 11, legend_y, 11, MUTED).width
            x += width + 20
            if x > rect.right - 40:
                x = rect.x
                legend_y += 15
        area = pygame.Rect(rect.x, legend_y + 22, rect.width, rect.bottom - legend_y - 22)
        self.d.rect(mix(PANEL, BG, 0.4), area, border_radius=6)
        all_pts = [p for _, _, pts in series for p in pts]
        if len(all_pts) < 2:
            self.text("En attente de données…", area.centerx, area.centery, 12, FAINT,
                      center=True)
            return
        x0 = min(p[0] for p in all_pts)
        x1 = max(max(p[0] for p in all_pts), x0 + 1)
        if percent:
            for frac in (0.0, 0.5, 1.0):
                yy = area.bottom - 6 - frac * (area.height - 12)
                self.d.line(LINE, (area.x + 4, yy), (area.right - 4, yy))
        for label, color, pts in series:
            if len(pts) < 2:
                continue
            ys = [p[1] for p in pts]
            lo, hi = (0.0, 1.0) if percent else (min(ys), max(ys))
            if hi - lo < 1e-9:
                lo, hi = lo - 0.5, hi + 0.5
            coords = [(area.x + 6 + (px - x0) / (x1 - x0) * (area.width - 12),
                       area.bottom - 6 - (py - lo) / (hi - lo) * (area.height - 12))
                      for px, py in pts]
            self.d.aalines(color, False, coords)
        self.text_right("{} parties".format(fmt_int(x1)), area.right, area.bottom + 2, 10, FAINT)
        if normalize:
            self.text("chaque courbe à sa propre échelle", area.x, area.bottom + 2, 10, FAINT)

    # --- Dessin : réseau -----------------------------------------------------------
    def draw_network(self):
        self.panel(NET_PANEL)
        self.text("LE RÉSEAU", 588, 143, 12, MUTED, True)
        self.button("reload", (968, 136, 120, 30), "Recharger le code",
                    enabled=self.worker is None and self.network is not None)
        if self.network is None:
            self.text("Aucun réseau trouvé.", 588, 170, 21, bold=True)
            self.text("Déposez un dossier dans reseaux/ (voir reseaux/README.md).",
                      588, 205, 14, MUTED)
            return
        name = registry.display_name(self.brain, self.network) if self.brain else self.network
        self.text(self.fit(name, 21, 480, True), 588, 165, 21, bold=True)
        if self.brain is None:
            self.text("Erreur au chargement du réseau :", 588, 205, 14, NEG)
            lines = (self.brain_error or "").strip().splitlines()[-6:]
            for i, line in enumerate(lines):
                self.mono_text(line[:78], 588, 232 + i * 18)
            return
        description = getattr(self.brain, "description", "") or "reseaux/{}".format(self.network)
        self.text(self.fit(description, 13, 492), 588, 194, 13, MUTED)
        explanation, explained = self.explanation()
        self.draw_value_gauge(explanation, explained)
        self.layer_views = []
        self.hover_node = None
        if explanation is None:
            game = explained
            if game is not None and game.size != self.brain.size:
                message = "Cette partie n'est pas à la taille du réseau."
            else:
                message = "Rien à afficher."
            self.text(message, DIAGRAM.centerx, DIAGRAM.centery, 14, FAINT, center=True)
        elif explanation.layers:
            self.draw_layers(explanation.layers)
        else:
            self.text("Ce réseau ne détaille pas ses couches (méthode explain).",
                      DIAGRAM.centerx, DIAGRAM.centery, 14, FAINT, center=True)
        self.draw_inspector(explanation)

    def draw_value_gauge(self, explanation, game):
        y = 222
        bar = pygame.Rect(588, y + 20, 492, 12)
        self.text("Qui gagne, selon le réseau ?", 588, y, 13, TEXT, True)
        value = explanation.value if explanation is not None else None
        if value is None:
            self.d.rect(PANEL_LIGHT, bar, border_radius=6)
            self.text_right("pas d'estimation", 1080, y, 12, FAINT)
            return
        value = max(-1.0, min(1.0, float(value)))
        red = (1 + value) / 2 if game.turn == RED else (1 - value) / 2
        split = bar.x + round(bar.width * red)
        self.d.rect(AZURE, bar, border_radius=6)
        self.d.rect(CORAL, (bar.x, bar.y, split - bar.x, bar.height),
                         border_top_left_radius=6, border_bottom_left_radius=6)
        self.text_right("Rouge {:.0f} %  ·  Bleu {:.0f} %".format(100 * red, 100 - 100 * red),
                        1080, y, 12, MUTED)

    def layout_layer(self, layer, box):
        values = np.asarray(layer.values, dtype=float)
        if values.ndim == 1 and values.size == 1:
            radius = min(box.width, box.height, 64) / 2 * 0.7
            return LayerView(layer, values, np.array([box.center], float), radius * 2, "single")
        if values.ndim == 1:
            count = values.size
            best = (0, 1)
            for cols in range(1, count + 1):
                rows = math.ceil(count / cols)
                side = min(box.width / cols, box.height / rows)
                if side > best[0]:
                    best = (side, cols)
            side, cols = best
            side = min(side, 22)
            rows = math.ceil(count / cols)
            x0 = box.centerx - cols * side / 2 + side / 2
            y0 = box.centery - rows * side / 2 + side / 2
            centers = np.array([(x0 + (i % cols) * side, y0 + (i // cols) * side)
                                for i in range(count)])
            return LayerView(layer, values, centers, side, "nodes")
        maps = values[None] if values.ndim == 2 else values.reshape(-1, *values.shape[-2:])
        shown = min(len(maps), 6)
        slot_h = box.height / shown
        centers, slots, radius = [], [], 0
        names = list(getattr(layer, "channels", ()) or ())
        for k in range(shown):
            label_h = 14 if (names or shown > 1) else 0
            slot = pygame.Rect(box.x, round(box.y + k * slot_h), box.width, round(slot_h))
            inner = pygame.Rect(slot.x + 2, slot.y + label_h, slot.width - 4,
                                slot.height - label_h - 4)
            radius, pts = mini_layout(maps.shape[1], maps.shape[2], inner)
            centers.append(pts)
            name = names[k] if k < len(names) else ("carte {}".format(k + 1) if shown > 1 else "")
            slots.append((name, slot))
        return LayerView(layer, values, np.concatenate(centers), radius * 2, "boards", slots)

    def draw_layers(self, layers):
        count = len(layers)
        # Les couches en forme de plateau ont besoin de plus de place qu'un seul neurone.
        weights = []
        for layer in layers:
            shape = np.shape(layer.values)
            weights.append(1.5 if len(shape) >= 2 else (0.8 if np.prod(shape) <= 1 else 1.0))
        unit = DIAGRAM.width / sum(weights)
        x = DIAGRAM.x
        for i, layer in enumerate(layers):
            box = pygame.Rect(round(x), DIAGRAM.y, round(unit * weights[i]), DIAGRAM.height)
            x += unit * weights[i]
            values = np.asarray(layer.values)
            self.text(self.fit(layer.name, 12, box.width - 6, True), box.centerx, box.y + 6,
                      12, TEXT, True, center=True)
            self.text("×".join(str(s) for s in values.shape), box.centerx, box.y + 22, 10,
                      FAINT, center=True)
            body = pygame.Rect(box.x + 6, box.y + 36, box.width - 12, box.height - 40)
            self.layer_views.append(self.layout_layer(layer, body))
        # Survol d'un neurone.
        for i, view in enumerate(self.layer_views):
            hit = view.hit(self.mouse)
            if hit is not None and DIAGRAM.collidepoint(self.mouse):
                self.hover_node = (i, hit)
        # Connexions d'abord, neurones par-dessus.
        for i in range(1, count):
            self.draw_connections(i)
        for view in self.layer_views:
            self.draw_layer(view)
        if self.hover_node is not None:
            i, j = self.hover_node
            x, y = self.layer_views[i].centers[j]
            size = max(6, self.layer_views[i].size)
            self.d.circle(TEXT, (round(x), round(y)), round(size * 0.6), 1)

    def source_index(self, i):
        """Couche d'où partent les poids de la couche i (Layer.source, sinon la précédente)."""
        source = getattr(self.layer_views[i].layer, "source", None)
        if source:
            for k in range(i - 1, -1, -1):
                if self.layer_views[k].layer.name == source:
                    return k
        return i - 1

    def connection_weights(self, i):
        view, previous = self.layer_views[i], self.layer_views[self.source_index(i)]
        weights = view.layer.weights
        if weights is None:
            return None
        weights = np.asarray(weights)
        if weights.shape != (view.flat.size, previous.flat.size):
            return None
        return weights

    def draw_connections(self, i):
        src = self.source_index(i)
        view, previous = self.layer_views[i], self.layer_views[src]
        weights = self.connection_weights(i)
        if weights is None:
            return
        shown_prev = len(previous.centers)
        if self.hover_node is not None and self.hover_node[0] == i:
            targets, per_target, strong = [self.hover_node[1]], 18, True
        elif self.hover_node is not None and self.hover_node[0] == src:
            # Neurone survolé dans la couche précédente : ses sorties les plus fortes.
            j = self.hover_node[1]
            contrib = weights[:len(view.centers), j] * previous.flat[j]
            order = np.argsort(np.abs(contrib))[::-1][:18]
            peak = float(np.abs(contrib[order[0]])) if len(order) else 0
            for k in order:
                if peak <= 0 or abs(contrib[k]) < 1e-9:
                    continue
                self.connection_line(previous.centers[j], view.centers[k], contrib[k] / peak, True)
            return
        else:
            activity = np.abs(view.flat[:len(view.centers)])
            targets, per_target, strong = list(np.argsort(activity)[::-1][:3]), 6, False
        for j in targets:
            contrib = weights[j, :shown_prev] * previous.flat[:shown_prev]
            order = np.argsort(np.abs(contrib))[::-1][:per_target]
            peak = float(np.abs(contrib[order[0]])) if len(order) else 0
            for k in order:
                if peak <= 0 or abs(contrib[k]) < 1e-9:
                    continue
                self.connection_line(previous.centers[k], view.centers[j], contrib[k] / peak,
                                     strong)

    def connection_line(self, start, end, strength, strong):
        color = POS if strength > 0 else NEG
        amount = abs(strength)
        color = mix(PANEL, color, (0.35 + 0.65 * amount) if strong else 0.12 + 0.3 * amount)
        width = 1 + round(2 * amount) if strong else 1
        if width > 1:
            self.d.line(color, start, end, width)
        else:
            self.d.aaline(color, start, end)

    def draw_layer(self, view):
        scale = view.scale or 1.0
        if view.kind == "single":
            value = float(view.flat[0])
            x, y = view.centers[0]
            self.d.circle(shade(value, max(scale, 1.0)), (round(x), round(y)),
                               round(view.size / 2))
            self.d.circle(LINE, (round(x), round(y)), round(view.size / 2), 1)
            self.text("{:+.2f}".format(value), x, y, 13, TEXT, True, center=True)
        elif view.kind == "nodes":
            side = max(2, view.size - 2)
            for (x, y), value in zip(view.centers, view.flat):
                rect = pygame.Rect(0, 0, side, side)
                rect.center = (round(x), round(y))
                self.d.rect(shade(value, scale), rect,
                                 border_radius=max(1, round(side / 4)))
        else:
            for name, slot in view.slots:
                if name:
                    self.text(self.fit(name, 10, slot.width), slot.centerx, slot.y + 6, 10,
                              FAINT, center=True)
            radius = view.size / 2
            for (x, y), value in zip(view.centers, view.flat):
                points = hex_points((x, y), radius * 0.92)
                if radius < 3:
                    self.d.circle(shade(value, scale), (round(x), round(y)),
                                       max(1, round(radius)))
                else:
                    self.d.polygon(shade(value, scale), points)

    def draw_inspector(self, explanation):
        area = pygame.Rect(588, 706, 492, 150)
        self.d.line(LINE, (588, 698), (1080, 698))
        if self.hover_node is None:
            lines = ["Survolez un neurone : ses connexions les plus fortes s'allument,",
                     "et pour la 1re couche cachée vous voyez quelles cases il regarde.",
                     "Sur le plateau, la teinte verte montre où le réseau veut jouer."]
            if explanation is not None:
                lines = list(explanation.notes)[:2] + lines
            for i, line in enumerate(lines[:5]):
                self.text(self.fit(line, 13, area.width), area.x, area.y + 4 + i * 22, 13,
                          MUTED if i >= len(lines) - 3 else TEXT)
            self.text("vert : positif / excite    rose : négatif / inhibe", area.x,
                      area.bottom - 14, 11, FAINT)
            return
        i, j = self.hover_node
        view = self.layer_views[i]
        layer = view.layer
        value = float(view.flat[j])
        where = "neurone {}".format(j)
        if view.kind == "boards":
            h, w = view.values.shape[-2:]
            k, rest = divmod(j, h * w)
            where = "case {}".format(cell_label(*divmod(rest, w)))
            if view.slots and len(view.slots) > 1:
                where += " · {}".format(view.slots[k][0])
        self.text(self.fit("{} · {}".format(layer.name, where), 15, 330, True), area.x,
                  area.y, 15, TEXT, True)
        self.text_right("valeur {:+.3f}".format(value), area.right, area.y + 2, 13, MINT)
        if layer.note:
            self.text(self.fit(layer.note, 12, area.width), area.x, area.y + 22, 12, FAINT)
        weights = self.connection_weights(i) if i > 0 else None
        if weights is None:
            return
        previous = self.layer_views[self.source_index(i)]
        row = weights[j]
        if previous.kind == "boards":
            # Champ récepteur : poids vers chaque case de chaque carte d'entrée.
            fields = row.reshape(previous.values.shape)
            fields = fields[None] if fields.ndim == 2 else fields
            self.text("Ce qu'il regarde (poids depuis « {} »)".format(previous.layer.name),
                      area.x, area.y + 42, 12, MUTED)
            shown = min(len(fields), 4)
            width = area.width / shown
            scale = float(np.abs(fields[:shown]).max()) or 1.0
            names = list(getattr(previous.layer, "channels", ()) or ())
            for k in range(shown):
                slot = pygame.Rect(round(area.x + k * width), area.y + 60, round(width) - 8, 86)
                radius, centers = mini_layout(fields.shape[1], fields.shape[2],
                                              slot.inflate(0, -14).move(0, 7))
                for (x, y), v in zip(centers, fields[k].reshape(-1)):
                    self.d.polygon(shade(v, scale), hex_points((x, y), radius * 0.92))
                if k < len(names):
                    self.text(names[k], slot.centerx, slot.y, 10, FAINT, center=True)
        else:
            contrib = row * previous.flat
            order = np.argsort(np.abs(contrib))[::-1][:4]
            self.text("Plus fortes influences (poids × activation) :", area.x, area.y + 42, 12,
                      MUTED)
            for n, k in enumerate(order):
                self.text("{} n° {} : {:+.3f}".format(previous.layer.name, int(k), contrib[k]),
                          area.x + 8, area.y + 62 + n * 20, 13,
                          POS if contrib[k] > 0 else NEG)

    # --- Dessin : session ----------------------------------------------------------
    def draw_session(self):
        self.panel(SIDE_PANEL)
        x, w = 1144, 244
        locked = self.worker is not None
        self.text("SESSION", x, 143, 12, MUTED, True)
        self.text("Réseau (dossier reseaux/)", x, 166, 13, TEXT)
        self.button("net_prev", (x, 188, 34, 34), "<", enabled=not locked and len(self.networks) > 1)
        self.button("net_next", (x + w - 34, 188, 34, 34), ">",
                    enabled=not locked and len(self.networks) > 1)
        self.text(self.fit(self.network or "aucun", 15, w - 84, True), x + w / 2, 205, 15,
                  MINT, True, center=True)
        if self.brain is None:
            status, color = "Erreur de chargement", NEG
        elif self.brain_loaded:
            status, color = "Poids repris · {} parties jouées".format(
                fmt_int(self.journal.games)), MUTED
        else:
            status, color = "Poids neufs (aucune sauvegarde {0}×{0})".format(self.size), MUTED
        self.text(self.fit(status, 12, w), x, 230, 12, color)

        self.text("Taille du plateau", x, 254, 13, TEXT)
        for i, size in enumerate(SIZES):
            self.button("size{}".format(size), (x + i * 62, 276, 56, 32), "{0}×{0}".format(size),
                        self.size == size, enabled=not locked)
        custom = self.size not in SIZES
        self.button("custom", (x, 314, 92, 32), "Impaire :", custom, enabled=not locked)
        self.button("custom_minus", (x + 100, 314, 40, 32), "−", enabled=not locked)
        self.text("{0}×{0}".format(self.custom_size), x + 172, 330, 15,
                  MINT if custom else TEXT, True, center=True)
        self.button("custom_plus", (x + 204, 314, 40, 32), "+", enabled=not locked)

        self.text("Adversaire d'entraînement", x, 358, 13, TEXT)
        for n, key in enumerate(("autojeu", "regles", "aleatoire", "parties")):
            self.button("opp_" + key, (x + (n % 2) * 125, 380 + (n // 2) * 38, 119, 32),
                        OPPONENTS[key] if key != "parties" else "Parties",
                        self.opponent == key, enabled=not locked)
        y = 460
        if self.opponent == "parties":
            enabled = not locked and len(self.datasets) > 1
            self.button("data_prev", (x, y, 30, 28), "<", enabled=enabled)
            self.button("data_next", (x + w - 30, y, 30, 28), ">", enabled=enabled)
            if self.datasets:
                path, count = self.datasets[self.dataset_index]
                label = "{} ({})".format(path.name, fmt_int(count))
            else:
                label = "aucune partie {0}×{0}".format(self.size)
            self.text(self.fit(label, 12, w - 72), x + w / 2, y + 14, 12, MUTED, center=True)
            self.button("generate", (x, y + 34, w, 30), "Générer 200 parties (règles)",
                        enabled=not locked)
        else:
            for k, line in enumerate(OPPONENT_HINTS[self.opponent]):
                self.text(line, x, y + 4 + k * 19, 12, MUTED)
        self.text("Règle d'échange", x, 538, 13, TEXT)
        self.button("swap_rule", (x + w - 76, 532, 76, 30), "Oui" if self.swap_rule else "Non",
                    self.swap_rule, enabled=not locked)
        self.text("Vitesse", x, 572, 13, TEXT)
        for n, (label, _) in enumerate(SPEEDS):
            self.button("speed{}".format(n), (x + n * 83, 594, 78, 30), label, self.speed == n)

        if self.task == "generate":
            done, total = self.progress or (0, 1)
            self.button("cancel_task", (x, 640, w, 44), "Génération {} / {} · arrêter".format(
                done, total), primary=True)
        else:
            self.button("start", (x, 640, w, 44),
                        "Mettre en pause (espace)" if self.training else "Lancer l'entraînement",
                        primary=True, enabled=self.brain is not None)
        self.d.line(LINE, (x, 698), (x + w, 698))
        stats = self.stats
        games = stats.get("parties", self.journal.games)
        self.text(fmt_int(games), x, 708, 21, TEXT, True)
        self.text("PARTIES", x, 736, 10, FAINT, True)
        rate = stats.get("parties_par_s") if self.training else None
        self.text("{:.0f}".format(rate) if rate is not None else "—", x + 130, 708, 21, TEXT, True)
        self.text("PARTIES / S", x + 130, 736, 10, FAINT, True)
        last = self.journal.evals[-1] if self.journal.evals else None
        if last:
            self.text("Dernière évaluation (sans exploration) :", x, 758, 12, MUTED)
            self.text("{:.0f} % vs règles  ·  {:.0f} % vs hasard".format(
                100 * last.get("vs_regles", 0), 100 * last.get("vs_aleatoire", 0)),
                x, 776, 14, MINT, True)
        metrics = stats.get("mesures") or {}
        for n, (key, value) in enumerate(list(metrics.items())[:3]):
            self.text(self.fit("{} : {:.4g}".format(key.replace("_", " "), value), 12, w),
                      x, 800 + n * 17, 12, FAINT)
        if stats.get("illegaux"):
            self.text("Coups illégaux remplacés : {}".format(stats["illegaux"]), x, 851, 12, NEG)

    # --- Dessin : pied de page, fenêtres ------------------------------------------
    def draw_footer(self):
        self.text("ESPACE lancer/pause  ·  ← → rejouer  ·  B bac à sable  ·  J le réseau joue"
                  "  ·  H aide", 30, 880, 11, FAINT)
        self.text_right("Sauvegardes : reseaux/{}/sauvegardes/taille_{}".format(
            self.network or "…", self.size), 1412, 880, 11, FAINT)
        if self.notice and pygame.time.get_ticks() < self.notice_until:
            self.panel((80, 560, 410, 34), (39, 62, 67), 10)
            self.text(self.fit(self.notice, 13, 396), 285, 577, 13, MINT, center=True)

    def draw_modal(self):
        self.d.veil((4, 10, 17, 210))
        self.buttons.clear()
        if self.error:
            self.panel((220, 180, 1000, 540), PANEL, 22)
            self.text("Le réseau a rencontré une erreur", 254, 208, 26, TEXT, True)
            self.text("L'entraînement est arrêté. Corrigez le code puis « Recharger le code ».",
                      254, 252, 15, MUTED)
            lines = str(self.error).strip().splitlines()[-18:]
            for i, line in enumerate(lines):
                self.mono_text(line[:140], 254, 292 + i * 19,
                               NEG if i == len(lines) - 1 else MUTED)
            self.button("close_error", (254, 650, 932, 44), "Fermer", primary=True)
            return
        self.panel((300, 120, 840, 660), PANEL, 22)
        self.text("Entraîner un réseau au Hex", 334, 148, 29, TEXT, True)
        self.text("COMMENT ÇA MARCHE", 335, 198, 12, MINT, True)
        lines = [
            ("01", "Déposez votre réseau dans reseaux/<nom>/__init__.py (copiez reseaux/_modele)."),
            ("02", "Choisissez la taille, l'adversaire, puis lancez : il joue et apprend à la chaîne."),
            ("03", "Autojeu : contre lui-même · Règles : contre l'ordi du jeu · Parties : fichiers."),
            ("04", "Turbo joue à pleine vitesse ; Rapide et Lent montrent chaque coup."),
            ("05", "Le plateau teinte en vert les cases que le réseau veut jouer."),
            ("06", "Au centre, ses couches : survolez un neurone pour voir ce qu'il regarde."),
            ("07", "Bac à sable : posez des pierres, ou « Affronter l'IA » (A) pour un duel."),
        ]
        for i, (number, message) in enumerate(lines):
            y = 236 + i * 40
            self.text(number, 335, y, 15, MINT, True)
            self.text(message, 375, y, 15, TEXT)
        self.text("Poids et courbes sont sauvegardés toutes les 3 s dans", 335, 532, 14, MUTED)
        self.text("reseaux/<nom>/sauvegardes/taille_<n>/ : on reprend où on s'était arrêté.",
                  335, 554, 14, MUTED)
        self.text("Sans interface, à pleine vitesse :  python3 trainer.py --console --adversaire regles",
                  335, 590, 14, MUTED)
        self.text("ESPACE lancer/pause · ← → rejouer · B bac à sable · J le réseau joue · H aide",
                  335, 668, 13, FAINT)
        self.button("close_help", (335, 706, 770, 43), "C'est parti", primary=True)

    def draw(self):
        self.buttons.clear()
        self.d.fill(BG)
        self.draw_header()
        self.draw_board_panel()
        self.draw_charts()
        self.draw_network()
        self.draw_session()
        self.draw_footer()
        if self.show_help or self.error:
            self.draw_modal()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.stop_worker()
        finally:
            self.screen.destroy()
            pygame.quit()
