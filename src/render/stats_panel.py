"""
Stats panel renderer — draws the 400x1080 HUD sidebar on the right.
"""

import pygame
import math
from src.render.colors import Color

_BG          = (15,  15,  20)
_BORDER      = (50,  50,  60)
_LABEL       = (130, 130, 145)
_VALUE       = (230, 235, 255)
_ACCENT_G    = (80,  220, 120)
_ACCENT_Y    = (255, 210,  60)
_ACCENT_R    = (255,  90,  80)
_ACCENT_B    = (90,  170, 255)
_BAR_BG      = (35,  35,  45)

_FONT_CACHE: dict = {}

def _font(name: str, size: int) -> pygame.font.Font:
    key = (name, size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = pygame.font.SysFont(name, size, bold=False)
        except Exception:
            _FONT_CACHE[key] = pygame.font.SysFont("monospace", size)
    return _FONT_CACHE[key]

def _label_font(): return _font("segoeui", 18)
def _value_font(): return _font("segoeui", 30)
def _big_font():   return _font("segoeui", 38)
def _small_font(): return _font("segoeui", 15)

class StatsPanel:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))

    # pragma: no cover
    def draw(self, screen: pygame.Surface, x_offset: int, y_offset: int,
             generation: int, total_cars: int, alive: int,
             time_left: float, time_limit: float,
             best_fitness: float, median_fitness: float,
             max_speed: float, best_net_nodes: int, best_net_conns: int) -> None:
        
        s = self.surface
        s.fill(_BG)
        
        # Left border separating track and sidebar
        pygame.draw.line(s, _BORDER, (0, 0), (0, self.height), 3)
        
        # NN visualizer space label
        title = _label_font().render("BEST AI NEURAL NETWORK", True, _LABEL)
        s.blit(title, (self.width // 2 - title.get_width() // 2, 20))
        
        # Draw a subtle box for the NN area
        pygame.draw.rect(s, _BAR_BG, (20, 60, self.width - 40, 420), border_radius=8, width=2)

        # Stats start below the NN visualizer
        start_y = 510
        cx = self.width // 2
        spacing = 65  # Tighter vertical spacing
        
        # METRICS
        self._draw_cell(s, cx, start_y, "GENERATION", str(generation), _ACCENT_B, True)
        self._draw_cell(s, cx, start_y + spacing*1 + 10, "ALIVE CARS", f"{alive} / {total_cars}", _ACCENT_G)
        self._draw_cell(s, cx, start_y + spacing*2, "TIME LEFT", f"{time_left:.1f}s / {time_limit:.1f}s", _ACCENT_Y)
        self._draw_cell(s, cx, start_y + spacing*3, "BEST FITNESS", self._fmt(best_fitness), _VALUE)
        self._draw_cell(s, cx, start_y + spacing*4, "MEDIAN FITNESS", self._fmt(median_fitness), _VALUE)
        
        speed_str = f"{max_speed:.1f} px/s" if max_speed > 0 else "0.0 px/s"
        self._draw_cell(s, cx, start_y + spacing*5, "LIVE TOP SPEED", speed_str, _ACCENT_R)
        
        net_str = f"{best_net_nodes} Nodes / {best_net_conns} Conns" if best_net_nodes > 0 else "—"
        self._draw_cell(s, cx, start_y + spacing*6, "BRAIN COMPLEXITY", net_str, _VALUE)

        screen.blit(s, (x_offset, y_offset))

    def _fmt(self, val):
        if math.isfinite(val): return f"{val:,.1f}"
        return "—"
        
    def _draw_cell(self, surf, cx, y, label, value, val_color, big=False):
        lbl = _label_font().render(label, True, _LABEL)
        val = (_big_font() if big else _value_font()).render(value, True, val_color)
        surf.blit(lbl, (cx - lbl.get_width() // 2, y))
        surf.blit(val, (cx - val.get_width() // 2, y + 22))
