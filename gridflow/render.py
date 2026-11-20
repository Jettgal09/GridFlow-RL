"""
gridflow/render.py — Pygame drawing of a Network onto a surface.

Pure drawing: it never changes the simulation.

What you see
  - City blocks between streets; asphalt roads with a double yellow centre line,
    dashed lane dividers, solid edge lines and painted turn arrows.
  - Each approach has 3 lanes (from the centre line outward): left-turn,
    straight, right-turn. Traffic drives on the right.
  - Cars: blue when moving, red when stopped (queueing).
  - The bar across each approach at the junction is its signal:
    green / amber (switching) / red.
  - Amber numbers at the map edge = cars waiting to enter the city.
"""

from __future__ import annotations

import pygame

from gridflow.sim import DC, DR, LEFT, RIGHT, STRAIGHT, Network

BG = (18, 19, 23)
BLOCK = (36, 44, 40)
BLOCK_EDGE = (46, 56, 51)
ROAD = (54, 57, 64)
BOX = (62, 65, 73)
WHITE = (205, 208, 214)
DASH = (150, 154, 162)
YELLOW = (214, 178, 64)
ARROW = (120, 124, 133)
CAR_MOVING = (96, 165, 255)
CAR_STOPPED = (238, 92, 80)
CAR_EDGE = (20, 22, 26)
GREEN = (60, 215, 110)
RED = (220, 58, 58)
AMBER = (245, 190, 60)
TEXT = (230, 232, 238)
MUTED = (150, 155, 168)

LANE_ORDER = {LEFT: 0, STRAIGHT: 1, RIGHT: 2}  # from centre line outward (right-hand traffic)


class Renderer:
    def __init__(self, net: Network, cell_px: int = 7, lane_px: int = 8, margin_cells: int = 12):
        self.net = net
        self.c = cell_px
        self.w = lane_px
        self.box = 6 * lane_px                      # junction = width of a 6-lane road
        self.margin = margin_cells * cell_px        # how much of the entry roads is shown
        self.seg = net.L * cell_px                  # length of a road between junctions
        self.spacing = self.box + self.seg
        self.width = 2 * self.margin + net.cols * self.box + (net.cols - 1) * self.seg
        self.height = 2 * self.margin + net.rows * self.box + (net.rows - 1) * self.seg
        self.font = pygame.font.SysFont("consolas", 12, bold=True)

    # ------------------------------------------------------------ geometry
    def centre(self, node):
        r, c = divmod(node, self.net.cols)
        return (self.margin + self.box / 2 + c * self.spacing,
                self.margin + self.box / 2 + r * self.spacing)

    def point(self, node, h, dist, lat):
        """Screen point `dist` px before node's stop line on approach h, `lat` px to the right of the centre line."""
        cx, cy = self.centre(node)
        dx, dy = DC[h], DR[h]          # direction of travel on screen (y grows downward)
        rx, ry = -dy, dx               # driver's right-hand side
        d = dist + self.box / 2
        return cx - dx * d + rx * lat, cy - dy * d + ry * lat

    def approach_len(self, node, h):
        return self.seg if self.net.upstream[node][h] is not None else self.margin

    # ------------------------------------------------------------- drawing
    def _static(self, surf, ox, oy):
        net = self.net
        surf.fill(BLOCK, (ox, oy, self.width, self.height))
        half = self.box / 2
        # asphalt
        for node in range(net.n_nodes):
            cx, cy = self.centre(node)
            r, c = divmod(node, net.cols)
            x0 = cx - half - (self.margin if c == 0 else 0)
            x1 = cx + half + (self.margin if c == net.cols - 1 else self.seg)
            pygame.draw.rect(surf, ROAD, (ox + x0, oy + cy - half, x1 - x0, self.box))
            y0 = cy - half - (self.margin if r == 0 else 0)
            y1 = cy + half + (self.margin if r == net.rows - 1 else self.seg)
            pygame.draw.rect(surf, ROAD, (ox + cx - half, oy + y0, self.box, y1 - y0))
        # markings, drawn along every approach (shared roads are simply drawn twice)
        w = self.w
        for node in range(net.n_nodes):
            for h in range(4):
                length = self.approach_len(node, h)
                # double yellow centre line
                for off in (-1, 1):
                    a = self.point(node, h, 0, off); b = self.point(node, h, length, off)
                    pygame.draw.line(surf, YELLOW, (ox + a[0], oy + a[1]), (ox + b[0], oy + b[1]), 1)
                # solid edge lines
                for lat in (-3 * w + 1, 3 * w - 1):
                    a = self.point(node, h, 0, lat); b = self.point(node, h, length, lat)
                    pygame.draw.line(surf, DASH, (ox + a[0], oy + a[1]), (ox + b[0], oy + b[1]), 1)
                # dashed lane dividers, both directions; solid near the stop line (no lane changes)
                for lat in (-2 * w, -w, w, 2 * w):
                    d = 0
                    while d < length:
                        seg_len = 5 if d > 4 * self.c else 4 * self.c
                        a = self.point(node, h, d, lat); b = self.point(node, h, min(d + seg_len, length), lat)
                        pygame.draw.line(surf, DASH, (ox + a[0], oy + a[1]), (ox + b[0], oy + b[1]), 1)
                        d += seg_len + 6
                # turn arrows painted in each lane
                for turn, k in LANE_ORDER.items():
                    self._arrow(surf, ox, oy, node, h, turn, 3.2 * self.c, (k + 0.5) * w)
        # junction boxes
        for node in range(net.n_nodes):
            cx, cy = self.centre(node)
            pygame.draw.rect(surf, BOX, (ox + cx - half, oy + cy - half, self.box, self.box))

    def _arrow(self, surf, ox, oy, node, h, turn, dist, lat):
        dx, dy = DC[h], DR[h]
        rx, ry = -dy, dx
        px, py = self.point(node, h, dist, lat)
        tail = (px - dx * 6, py - dy * 6)
        if turn == STRAIGHT:
            tip = (px + dx * 5, py + dy * 5)
            pts = [tail, tip]
            head_dir = (dx, dy)
        else:
            s = -1 if turn == LEFT else 1
            elbow = (px + dx * 1, py + dy * 1)
            tip = (elbow[0] + s * rx * 4, elbow[1] + s * ry * 4)
            pts = [tail, elbow, tip]
            head_dir = (s * rx, s * ry)
        pts = [(ox + x, oy + y) for x, y in pts]
        pygame.draw.lines(surf, ARROW, False, pts, 1)
        hx, hy = head_dir
        nx, ny = -hy, hx
        t = pts[-1]
        pygame.draw.polygon(surf, ARROW, [t, (t[0] - hx * 3 + nx * 2, t[1] - hy * 3 + ny * 2),
                                          (t[0] - hx * 3 - nx * 2, t[1] - hy * 3 - ny * 2)])

    def draw(self, surf: pygame.Surface, ox: int = 0, oy: int = 0):
        net = self.net
        prev_clip = surf.get_clip()
        surf.set_clip(pygame.Rect(ox, oy, self.width, self.height))
        self._static(surf, ox, oy)
        car_len, car_w = self.c - 1, self.w - 4
        for node in range(net.n_nodes):
            for h in range(4):
                green = net.is_green(node, h)
                # signal bar across the 3 lanes, just inside the junction
                if net.trans[node]:
                    col = AMBER if net.phase[node] == (h & 1) else RED
                else:
                    col = GREEN if green else RED
                a = self.point(node, h, -2, 1); b = self.point(node, h, -2, 3 * self.w - 1)
                pygame.draw.line(surf, col, (ox + a[0], oy + a[1]), (ox + b[0], oy + b[1]), 3)
                dx = DC[h]
                for ln in net.approach[node][h]:
                    lat = (LANE_ORDER[ln.turn] + 0.5) * self.w
                    for idx, p in enumerate(ln.pos):
                        dist = (ln.length - 1 - p + 0.5) * self.c
                        if dist > self.approach_len(node, h) + self.c:
                            break  # beyond the visible part of an entry road
                        x, y = self.point(node, h, dist, lat)
                        ahead = ln.pos[idx - 1] if idx > 0 else None
                        stopped = (ahead is not None and ahead == p + 1) or p == ln.length - 1
                        body = CAR_STOPPED if stopped else CAR_MOVING
                        lw, lh = (car_len, car_w) if dx else (car_w, car_len)
                        rect = pygame.Rect(0, 0, lw, lh)
                        rect.center = (ox + x, oy + y)
                        pygame.draw.rect(surf, body, rect, border_radius=2)
                        pygame.draw.rect(surf, CAR_EDGE, rect, 1, border_radius=2)
        # entry queues
        for s, (node, h) in enumerate(net.sources):
            b = len(net.entry_q[s])
            if b:
                x, y = self.point(node, h, self.margin - 8, 1.5 * self.w)
                txt = self.font.render(f"+{b}", True, AMBER)
                surf.blit(txt, (ox + x - txt.get_width() / 2, oy + y - txt.get_height() / 2))
        surf.set_clip(prev_clip)


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
    pygame.draw.rect(surf, (30, 33, 40), (x, y, width, 18 * len(lines) + 10), border_radius=4)
    for i, (s, col) in enumerate(lines):
        surf.blit(font.render(s, True, col), (x + 10, y + 6 + 18 * i))
