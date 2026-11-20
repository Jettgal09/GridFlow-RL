"""
gridflow/render.py — Pygame drawing of a Network onto a surface.

Pure drawing: it never changes the simulation. Cars are blue when moving and
red when stopped; stop lines show each approach's signal; numbers at the edge
are cars waiting to enter the grid.
"""

from __future__ import annotations

import pygame

from gridflow.sim import DC, DR, LEFT, RIGHT, STRAIGHT, Network

BG = (24, 26, 31)
ROAD = (58, 61, 70)
BOX = (70, 73, 84)
LANE_MARK = (95, 99, 110)
CAR_MOVING = (86, 156, 255)
CAR_STOPPED = (240, 96, 84)
GREEN = (60, 210, 110)
RED = (215, 60, 60)
AMBER = (245, 190, 60)
TEXT = (230, 232, 238)
MUTED = (150, 155, 168)

LANE_ORDER = {LEFT: 0, STRAIGHT: 1, RIGHT: 2}  # from road centre outward (right-hand traffic)


class Renderer:
    def __init__(self, net: Network, cell_px: int = 5, lane_px: int = 6, margin_cells: int | None = None):
        self.net = net
        self.c = cell_px
        self.w = lane_px
        self.box = 6 * lane_px
        self.margin = (margin_cells if margin_cells is not None else net.L) * cell_px
        self.spacing = self.box + net.L * cell_px
        self.width = 2 * self.margin + net.cols * self.box + (net.cols - 1) * net.L * cell_px
        self.height = 2 * self.margin + net.rows * self.box + (net.rows - 1) * net.L * cell_px
        self.font = pygame.font.SysFont("consolas", 12)

    def centre(self, node):
        r, cc = divmod(node, self.net.cols)
        return (self.margin + self.box / 2 + cc * self.spacing,
                self.margin + self.box / 2 + r * self.spacing)

    def _lane_xy(self, node, h, turn, dist):
        """Point at `dist` px before the stop line, in the centre of the given lane."""
        cx, cy = self.centre(node)
        k = LANE_ORDER[turn]
        lat = (k + 0.5) * self.w
        dx, dy = DC[h], DR[h]          # travel direction on screen (y grows downward)
        rx, ry = -dy, dx               # right-hand side of travel direction
        d = dist + self.box / 2
        return cx - dx * d + rx * lat, cy - dy * d + ry * lat

    def draw(self, surf: pygame.Surface, ox: int = 0, oy: int = 0):
        net = self.net
        surf.fill(BG, (ox, oy, self.width, self.height))
        half = self.box / 2
        # roads
        for node in range(net.n_nodes):
            cx, cy = self.centre(node)
            r, c = divmod(node, net.cols)
            x0 = ox + (cx - half - (self.margin if c == 0 else 0))
            x1 = ox + (cx + half + (self.margin if c == net.cols - 1 else self.spacing - self.box))
            pygame.draw.rect(surf, ROAD, (x0, oy + cy - half, x1 - x0, self.box))
            y0 = oy + (cy - half - (self.margin if r == 0 else 0))
            y1 = oy + (cy + half + (self.margin if r == net.rows - 1 else self.spacing - self.box))
            pygame.draw.rect(surf, ROAD, (ox + cx - half, y0, self.box, y1 - y0))
        for node in range(net.n_nodes):
            cx, cy = self.centre(node)
            pygame.draw.rect(surf, BOX, (ox + cx - half, oy + cy - half, self.box, self.box))
            pygame.draw.line(surf, LANE_MARK, (ox + cx - half, oy + cy), (ox + cx - half - 4 * self.c, oy + cy))
        # signals (stop lines) and cars
        for node in range(net.n_nodes):
            for h in range(4):
                if net.trans[node]:
                    col = AMBER if net.phase[node] == (h & 1) else RED
                else:
                    col = GREEN if net.phase[node] == (h & 1) else RED
                a = self._lane_xy(node, h, LEFT, 0)
                b = self._lane_xy(node, h, RIGHT, 0)
                dx = DC[h]
                pygame.draw.line(surf, col, (ox + a[0], oy + a[1]), (ox + b[0], oy + b[1]), 3)
                for ln in net.approach[node][h]:
                    for idx, p in enumerate(ln.pos):
                        dist = (ln.length - 1 - p + 0.5) * self.c
                        x, y = self._lane_xy(node, h, ln.turn, dist)
                        # a car is stopped if the cell ahead is occupied or it is at a red stop line
                        ahead = ln.pos[idx - 1] if idx > 0 else None
                        stopped = (ahead is not None and ahead == p + 1) or (
                            p == ln.length - 1 and not net.is_green(node, h))
                        car_col = CAR_STOPPED if stopped else CAR_MOVING
                        lw = (self.c - 1) if dx else (self.w - 2)
                        lh = (self.w - 2) if dx else (self.c - 1)
                        pygame.draw.rect(surf, car_col, (ox + x - lw / 2, oy + y - lh / 2, lw, lh))
        # entry queues
        for s, (node, h) in enumerate(net.sources):
            b = len(net.entry_q[s])
            if b:
                x, y = self._lane_xy(node, h, STRAIGHT, net.L * self.c + 8)
                txt = self.font.render(str(b), True, AMBER)
                surf.blit(txt, (ox + x - txt.get_width() / 2, oy + y - txt.get_height() / 2))


def draw_hud(surf, font, x, y, title, net: Network, width):
    """A compact stats panel under a simulation view."""
    m = net.metrics()
    lines = [
        (title, TEXT),
        (f"time {net.t // 60:02d}:{net.t % 60:02d}   cars on grid {net.vehicles_in_network():4d}", MUTED),
        (f"waiting now {net.stopped_now:4d}   queued outside {net.total_backlog():4d}", MUTED),
        (f"avg wait / car {m['wait_per_vehicle_s']:6.1f} s", TEXT),
        (f"finished trips {m['completed']:6d}", MUTED),
    ]
    pygame.draw.rect(surf, (32, 35, 42), (x, y, width, 18 * len(lines) + 10))
    for i, (s, col) in enumerate(lines):
        surf.blit(font.render(s, True, col), (x + 10, y + 6 + 18 * i))
