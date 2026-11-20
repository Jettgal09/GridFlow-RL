"""
Side-by-side live demo: two controllers, identical traffic.

  uv run python demo.py                                   # actuated vs RL on 3x3 surge
  uv run python demo.py --grid 4 --scenario incident      # zero-shot 4x4 (never trained on)
  uv run python demo.py --left fixed_time --right rl
  uv run python demo.py --gif docs/demo.gif --seconds 900 # headless recording

Keys: SPACE pause, UP/DOWN speed, ESC quit.
"""

import argparse
import os
import sys

if "--gif" in sys.argv:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from gridflow import scenarios as S
from gridflow.controllers import Actuated, FixedTime, MaxPressure, PolicyController
from gridflow.render import Renderer, draw_hud
from gridflow.sim import Network

HUD_H = 110


def build(name, model_path):
    if name == "fixed_time":
        return FixedTime(15), "Fixed-time (15 s greens)"
    if name == "actuated":
        return Actuated(5, 40, 4), "Vehicle-actuated"
    if name == "max_pressure":
        return MaxPressure(20), "Max-pressure"
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    return PolicyController(model, name="rl"), "Shared RL policy (neighbour-aware)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--scenario", default="surge", choices=S.EVAL_SCENARIOS)
    ap.add_argument("--left", default="actuated")
    ap.add_argument("--right", default="rl")
    ap.add_argument("--model", default="runs/neighbour_s0/model.zip")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--start", type=int, default=0, help="fast-forward to this second before showing")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--seconds", type=int, default=3600)
    ap.add_argument("--gif_every", type=int, default=3)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    pygame.init()
    font = pygame.font.SysFont("consolas", 14)
    sides = []
    for name in (args.left, args.right):
        net = Network(args.grid, args.grid, S.make(args.scenario), seed=args.seed)
        ctrl, title = build(name, args.model)
        ctrl.reset(net)
        sides.append((net, ctrl, title, Renderer(net, cell_px=5 if args.grid <= 4 else 3)))
    rw, rh = sides[0][3].width, sides[0][3].height
    gap = 16
    W, H = 2 * rw + gap, rh + HUD_H + 30
    screen = pygame.display.set_mode((W, H)) if not args.gif else pygame.Surface((W, H))
    pygame.display.set_caption("GridFlow-RL — live comparison")
    clock = pygame.time.Clock()

    def advance():
        for net, ctrl, _, _ in sides:
            if net.t % ctrl.interval == 0:
                ctrl.decide(net)
            net.step()

    for _ in range(args.start):
        advance()

    frames, paused, speed, running = [], False, 2, True
    while running and sides[0][0].t < args.start + args.seconds:
        if not args.gif:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_UP:
                    speed = min(speed * 2, 64)
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_DOWN:
                    speed = max(speed // 2, 1)
        if not paused:
            for _ in range(1 if args.gif else speed):
                advance()
        screen.fill((18, 19, 23))
        title = f"{args.grid}x{args.grid} grid  |  scenario: {args.scenario}  |  same traffic on both sides"
        screen.blit(font.render(title, True, (200, 204, 214)), (10, 6))
        for k, (net, _, name, rnd) in enumerate(sides):
            x = k * (rw + gap)
            rnd.draw(screen, x, 26)
            draw_hud(screen, font, x, 26 + rh + 4, name, net, rw)
        if args.gif:
            if sides[0][0].t % args.gif_every == 0:
                frames.append(pygame.image.tostring(screen, "RGB"))
        else:
            pygame.display.flip()
            clock.tick(args.fps)

    if args.gif:
        from PIL import Image
        imgs = [Image.frombytes("RGB", (W, H), f).quantize(colors=64) for f in frames]
        os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=50, loop=0, optimize=True)
        print("saved", args.gif, len(imgs), "frames")
    pygame.quit()


if __name__ == "__main__":
    main()
