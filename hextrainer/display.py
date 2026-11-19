"""Affichage net sur écran Retina.

Le jeu dessine sur une toile de taille fixe, puis la redimensionne pour la
fenêtre : sur un écran Retina, macOS l'agrandit encore ×2, d'où un rendu flou.
Ici, la fenêtre est ouverte en haute résolution (`allow_high_dpi`) et tout est
dessiné directement à la taille réelle en pixels : l'interface garde des
coordonnées « logiques » (1440 × 900) que le `Painter` multiplie par l'échelle.
"""

import pygame
import pygame.gfxdraw

RADII = ("border_radius", "border_top_left_radius", "border_top_right_radius",
         "border_bottom_left_radius", "border_bottom_right_radius")


class Painter:
    """Primitives de dessin en coordonnées logiques, rendues à l'échelle."""

    def __init__(self, logical_size):
        self.logical = logical_size
        self.scale = 0.0
        self.surface = None
        self._fonts = {}

    def resize(self, scale):
        if abs(scale - self.scale) < 1e-4 and self.surface is not None:
            return
        self.scale = scale
        width, height = self.logical
        self.surface = pygame.Surface((max(1, round(width * scale)),
                                       max(1, round(height * scale))))
        self._fonts.clear()

    # --- Conversions ------------------------------------------------------------
    def v(self, value):
        return value * self.scale

    def pt(self, point):
        return (point[0] * self.scale, point[1] * self.scale)

    def ipt(self, point):
        return (round(point[0] * self.scale), round(point[1] * self.scale))

    def r(self, rect):
        rect = pygame.Rect(rect)
        s = self.scale
        left, top = round(rect.x * s), round(rect.y * s)
        return pygame.Rect(left, top, round(rect.right * s) - left, round(rect.bottom * s) - top)

    def w(self, width):
        return max(1, round(width * self.scale))

    # --- Formes -------------------------------------------------------------------
    def fill(self, color):
        self.surface.fill(color)

    def rect(self, color, rect, width=0, **radii):
        scaled = {k: round(v * self.scale) for k, v in radii.items() if k in RADII}
        pygame.draw.rect(self.surface, color, self.r(rect), self.w(width) if width else 0,
                         **scaled)

    def line(self, color, start, end, width=1):
        pygame.draw.line(self.surface, color, self.pt(start), self.pt(end), self.w(width))

    def lines(self, color, closed, points, width=1):
        pygame.draw.lines(self.surface, color, closed, [self.pt(p) for p in points],
                          self.w(width))

    def aaline(self, color, start, end):
        if self.scale >= 1.5:  # Trait fin mais visible à haute résolution.
            pygame.draw.line(self.surface, color, self.pt(start), self.pt(end), self.w(1))
        pygame.draw.aaline(self.surface, color, self.pt(start), self.pt(end))

    def aalines(self, color, closed, points):
        points = [self.pt(p) for p in points]
        if self.scale >= 1.5:
            pygame.draw.lines(self.surface, color, closed, points, self.w(1))
        pygame.draw.aalines(self.surface, color, closed, points)

    def circle(self, color, center, radius, width=0):
        pygame.draw.circle(self.surface, color, self.pt(center), max(1, radius * self.scale),
                           self.w(width) if width else 0)

    def polygon(self, color, points):
        points = [self.ipt(p) for p in points]
        pygame.gfxdraw.filled_polygon(self.surface, points, color)
        pygame.gfxdraw.aapolygon(self.surface, points, color)

    def aapolygon(self, color, points):
        pygame.gfxdraw.aapolygon(self.surface, [self.ipt(p) for p in points], color)

    def veil(self, rgba):
        layer = pygame.Surface(self.surface.get_size(), pygame.SRCALPHA)
        layer.fill(rgba)
        self.surface.blit(layer, (0, 0))

    # --- Texte -------------------------------------------------------------------
    def font(self, path, size, bold=False):
        key = path, max(6, round(size * self.scale)), bold
        if key not in self._fonts:
            font = pygame.font.Font(path, key[1])
            if bold and path is None:
                font.set_bold(True)
            self._fonts[key] = font
        return self._fonts[key]

    def blit_text(self, rendered, anchor, point):
        """Place un texte déjà rendu ; renvoie son rectangle en coordonnées logiques."""
        rect = rendered.get_rect(**{anchor: self.ipt(point)})
        self.surface.blit(rendered, rect)
        s = self.scale
        return pygame.Rect(round(rect.x / s), round(rect.y / s), round(rect.width / s),
                           round(rect.height / s))


class HiDpiWindow:
    """Fenêtre haute résolution qui affiche la toile du `Painter` sans la flouter."""

    def __init__(self, title, logical_size, icon=None):
        pygame.display.init()
        pygame.font.init()
        self.logical = logical_size
        desktop = pygame.display.get_desktop_sizes()[0]
        size = (min(logical_size[0], max(900, desktop[0] - 60)),
                min(logical_size[1], max(560, desktop[1] - 90)))
        self.window = pygame.Window(title, size, resizable=True, allow_high_dpi=True)
        if icon is not None:
            self.window.set_icon(icon)
        self.painter = Painter(logical_size)
        self.offset = (0, 0)
        self.ratio = 1.0
        self.layout()

    def layout(self):
        """Échelle et marges pour la taille actuelle de la fenêtre (en pixels réels)."""
        surface = self.window.get_surface()
        width, height = surface.get_size()
        points = self.window.size[0] or 1
        self.ratio = width / points
        scale = min(width / self.logical[0], height / self.logical[1])
        self.painter.resize(scale)
        canvas = self.painter.surface.get_size()
        self.offset = ((width - canvas[0]) // 2, (height - canvas[1]) // 2)
        return surface

    def present(self, background):
        """Affiche la toile ; appeler `layout()` avant de dessiner l'image suivante."""
        surface = self.window.get_surface()
        surface.fill(background)
        surface.blit(self.painter.surface, self.offset)
        self.window.flip()

    def logical_mouse(self, position):
        scale = self.painter.scale or 1.0
        return ((position[0] * self.ratio - self.offset[0]) / scale,
                (position[1] * self.ratio - self.offset[1]) / scale)

    def destroy(self):
        self.window.destroy()
