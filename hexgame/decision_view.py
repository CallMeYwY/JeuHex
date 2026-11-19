"""Fenêtre SDL qui affiche les étapes de décision du match en cours."""

import pygame
from pygame._sdl2.video import Renderer, Texture, Window

STAGES = (("swap", "Échange"), ("tactics", "Tactique"), ("paths", "Chemins"))
BG = (15, 22, 33)
CARD = (22, 32, 46)
EDGE = (62, 77, 96)
TEXT = (235, 241, 246)
MUTED = (147, 165, 184)
ACTIVE = (158, 221, 197)


def column_label(index):
    label = ""
    while index >= 0:
        index, letter = divmod(index, 26)
        label = chr(65 + letter) + label
        index -= 1
    return label


def action_label(action):
    if action == "swap":
        return "Échange des couleurs"
    row, col = action
    return "{}{}".format(column_label(col), row + 1)


class DecisionView:
    def __init__(self, font):
        self.canvas = pygame.Surface((860, 390))
        self.font = font
        self.window = Window("HEX — Réflexion des IA", size=self.canvas.get_size())
        try:
            self.renderer = Renderer(self.window)
        except pygame.error:
            self.window.destroy()
            raise
        self.texture = None
        self.last_state = None

    def text(self, value, pos, size=16, color=TEXT, bold=False, center=False):
        surface = self.font(size, bold).render(value, True, color)
        rect = surface.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.canvas.blit(surface, rect)

    def draw_player(self, x, name, color, active, decision):
        pygame.draw.rect(self.canvas, CARD, (x, 77, 398, 284), border_radius=14)
        pygame.draw.circle(self.canvas, color, (x + 25, 107), 6)
        self.text(name, (x + 40, 93), 22, bold=True)
        if active:
            self.text("Au tour de jouer", (x + 249, 99), color=ACTIVE)

        trace = decision[1] if decision else {}
        pygame.draw.line(self.canvas, EDGE, (x + 67, 161), (x + 327, 161), 2)
        for index, (stage, label) in enumerate(STAGES):
            center = (x + 67 + index * 130, 161)
            used = bool(trace.get(stage))
            chosen = trace.get("selected") == stage
            if chosen:
                pygame.draw.circle(self.canvas, color, center, 20, 2)
            pygame.draw.circle(self.canvas, color if used else EDGE, center, 12)
            self.text(label, (center[0], 193), 15, TEXT if used else MUTED, center=True)

        if decision is None:
            self.text("En attente du premier coup", (x + 20, 230), color=MUTED)
            return
        action, trace = decision
        self.text("Dernier choix · " + action_label(action), (x + 20, 224), 18, bold=True)
        if "top" in trace:
            self.text("{} cases évaluées · meilleurs scores".format(trace["tested"]),
                      (x + 20, 260), color=MUTED)
            for index, (score, cell) in enumerate(trace["top"]):
                left = x + 20 + 73 * index
                self.text(action_label(cell), (left, 293), 15, bold=True)
                self.text("{:+.2f}".format(score), (left, 315), 14, MUTED)
        elif not trace:
            self.text("Détails indisponibles pour cette stratégie.", (x + 20, 260), color=MUTED)

    def present(self, players, active, decisions):
        state = (players, active, tuple(decisions.items()))
        if self.last_state != state:
            self.canvas.fill(BG)
            self.text("Réflexion des IA", (24, 17), 27, bold=True)
            self.text("Le cercle indique la règle qui a déterminé le dernier choix.",
                      (24, 50), 15, MUTED)
            if not players:
                self.text("Lancez un duel, un tournoi ou une partie contre l’ordinateur.",
                          (24, 112), 19, MUTED)
            for index, (name, color) in enumerate(players):
                self.draw_player(24 + 414 * index, name, color, active == name,
                                 decisions.get(name))
            self.texture = Texture.from_surface(self.renderer, self.canvas)
            self.last_state = state
        self.texture.draw()
        self.renderer.present()

    def close(self):
        self.texture = None
        self.renderer = None
        self.window.destroy()
