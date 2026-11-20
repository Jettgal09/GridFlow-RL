"""
Cars waiting over time during the disruption scenarios, one line per controller.

  uv run python scripts/timeseries.py --model runs/neighbour_s0/model.zip
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from gridflow import scenarios as S
from gridflow.controllers import Actuated, FixedTime, MaxPressure, PolicyController, run_episode
from gridflow.sim import Network

COLOR = {"Fixed-time": "#9aa0a6", "Actuated": "#f2a541", "Max-pressure": "#d9534f",
         "RL (neighbour-aware)": "#1f6fd1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/neighbour_s0/model.zip")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--seed", type=int, default=10_002)
    ap.add_argument("--out", default="results/fig_timeseries.png")
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    ctrls = {"Fixed-time": lambda: FixedTime(15), "Actuated": lambda: Actuated(5, 40, 4),
             "Max-pressure": lambda: MaxPressure(20),
             "RL (neighbour-aware)": lambda: PolicyController(model)}
    scen = ["rush_hour", "surge", "incident"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, sc in zip(axes, scen):
        for name, mk in ctrls.items():
            trace = []
            run_episode(Network(args.grid, args.grid, S.make(sc)), mk(), seed=args.seed,
                        on_step=lambda n: trace.append(n.stopped_now))
            per_min = np.asarray(trace).reshape(-1, 60).mean(1)
            ax.plot(np.arange(len(per_min)), per_min, label=name, color=COLOR[name], lw=1.8)
        if sc == "incident":
            ax.axvspan(15, 40, color="k", alpha=0.07); ax.text(27.5, ax.get_ylim()[1] * 0.92, "road blocked", ha="center")
        if sc == "surge":
            ax.axvspan(20, 35, color="k", alpha=0.07); ax.text(27.5, ax.get_ylim()[1] * 0.92, "surge", ha="center")
        ax.set_title(f"{sc.replace('_', ' ')} ({args.grid}x{args.grid}, test seed {args.seed})")
        ax.set_xlabel("minute"); ax.set_ylabel("cars waiting (inside + queued outside)")
        ax.set_yscale("symlog", linthresh=50)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
