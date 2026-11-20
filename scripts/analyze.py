"""
Turn results/eval.csv into tables and figures.

  uv run python scripts/analyze.py

Primary metric: wait_per_vehicle_s = all vehicle-seconds spent stopped (inside the
grid AND queued outside it) / number of vehicles that arrived. Lower is better.
It counts cars that never got in, so a controller cannot look good by blocking.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridflow.scenarios import EVAL_SCENARIOS

OUT = "results"
ORDER = ["fixed_time", "actuated", "max_pressure", "max_pressure+shield",
         "rl_neighbour", "rl_local+shield", "rl_neighbour+shield"]
LABEL = {"fixed_time": "Fixed-time", "actuated": "Actuated", "max_pressure": "Max-pressure",
         "max_pressure+shield": "Max-pressure + shield",
         "rl_neighbour": "RL, no shield", "rl_local+shield": "RL local-only + shield",
         "rl_neighbour+shield": "GridFlow-RL (ours)"}
COLOR = {"fixed_time": "#9aa0a6", "actuated": "#f2a541", "max_pressure": "#d9534f",
         "max_pressure+shield": "#f08a85", "rl_neighbour": "#9cc3ef",
         "rl_local+shield": "#6aa0d8", "rl_neighbour+shield": "#1f4fbf"}
MAIN = "rl_neighbour+shield"
CURVE = {"neighbour": "#1f4fbf", "local": "#7fb3e6"}
BASES = ["fixed_time", "actuated", "max_pressure", "max_pressure+shield"]
SEEN = {"2x2", "3x3"}
METRIC = "wait_per_vehicle_s"
rng = np.random.default_rng(0)


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def boot_ci(d, n=5000):
    d = np.asarray(d, float)
    m = rng.choice(d, (n, len(d))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def per_episode(df):
    """Average RL over its training seeds so every controller has one value per (grid, scenario, test seed)."""
    return (df.groupby(["controller", "grid", "scenario", "test_seed"], as_index=False)
              .mean(numeric_only=True))


def main():
    df = pd.read_csv(os.path.join(OUT, "eval.csv"))
    ctrls = [c for c in ORDER if c in df.controller.unique()]
    ep = per_episode(df)
    grids = sorted(df.grid.unique(), key=lambda g: int(g.split("x")[0]))
    lines = []

    # ---- Table 1: main results per grid x scenario
    lines.append(f"# Results — {METRIC} (mean ± 95% CI over 10 test seeds; RL averaged over training seeds)\n")
    for g in grids:
        tag = "seen in training" if g in SEEN else "NEVER seen in training (zero-shot)"
        lines.append(f"\n## Grid {g} ({tag})\n")
        lines.append("| scenario | " + " | ".join(LABEL[c] for c in ctrls) + " |")
        lines.append("|---" * (len(ctrls) + 1) + "|")
        for sc in EVAL_SCENARIOS:
            row = []
            vals = {c: ep[(ep.controller == c) & (ep.grid == g) & (ep.scenario == sc)][METRIC] for c in ctrls}
            best = min(ctrls, key=lambda c: vals[c].mean())
            for c in ctrls:
                s = f"{vals[c].mean():.1f} ± {ci95(vals[c]):.1f}"
                row.append(f"**{s}**" if c == best else s)
            lines.append(f"| {sc} | " + " | ".join(row) + " |")

    # ---- Table 2: paired difference ours vs best classical controller per cell
    if MAIN in ctrls:
        lines.append("\n## GridFlow-RL vs the best classical controller in each cell\n")
        lines.append("Paired difference in wait per vehicle (negative = ours better), bootstrap 95% CI "
                     "over the 10 test seeds. 'Best classical' is chosen per cell, after the fact — "
                     "the strictest possible comparison.\n")
        lines.append("| grid | scenario | best classical | classical | ours | diff | 95% CI | verdict |")
        lines.append("|---|---|---|---|---|---|---|---|")
        wins = losses = ties = 0
        for g in grids:
            for sc in EVAL_SCENARIOS:
                cell = ep[(ep.grid == g) & (ep.scenario == sc)]
                base = {c: cell[cell.controller == c].sort_values("test_seed")[METRIC].values
                        for c in BASES if c in ctrls}
                bb = min(base, key=lambda c: base[c].mean())
                rl = cell[cell.controller == MAIN].sort_values("test_seed")[METRIC].values
                d = rl - base[bb]
                lo, hi = boot_ci(d)
                verdict = "ours better" if hi < 0 else ("ours worse" if lo > 0 else "no clear difference")
                wins += hi < 0; losses += lo > 0; ties += (lo <= 0 <= hi)
                lines.append(f"| {g} | {sc} | {LABEL[bb]} | {base[bb].mean():.1f} | {rl.mean():.1f} | "
                             f"{d.mean():+.1f} | [{lo:+.1f}, {hi:+.1f}] | {verdict} |")
        lines.append(f"\nOurs clearly better in {wins}, clearly worse in {losses}, no clear difference in {ties} "
                     f"of {wins + losses + ties} cells.\n")

    # ---- Table 3: ablations
    abl = [("rl_neighbour", "Remove the shield"), ("rl_local+shield", "Remove the neighbour view")]
    if MAIN in ctrls:
        lines.append("\n## Ablations: what does each ingredient contribute?\n")
        lines.append("Mean wait per vehicle over all scenarios (s). Positive change = worse without it.\n")
        lines.append("| grid | full method | " + " | ".join(n for _, n in abl) + " |")
        lines.append("|---|---|" + "---|" * len(abl))
        for g in grids:
            full = ep[(ep.grid == g) & (ep.controller == MAIN)][METRIC].mean()
            row = [f"{full:.1f}"]
            for c, _ in abl:
                v = ep[(ep.grid == g) & (ep.controller == c)][METRIC].mean()
                row.append(f"{v:.1f} ({100 * (v - full) / full:+.0f}%)")
            lines.append(f"| {g} | " + " | ".join(row) + " |")

    # ---- Table 4: robustness across training seeds
    rl = df[df.controller.str.startswith("rl_")]
    if len(rl):
        lines.append("\n## Spread across training seeds (all grids and scenarios pooled)\n")
        lines.append("| controller | training seed | mean wait per vehicle (s) |")
        lines.append("|---|---|---|")
        for (c, s), v in rl.groupby(["controller", "train_seed"])[METRIC]:
            lines.append(f"| {LABEL[c]} | {s} | {v.mean():.1f} |")

    # ---- Secondary metrics on 3x3
    lines.append("\n## Secondary metrics, 3x3, all scenarios pooled\n")
    lines.append("| controller | mean travel time (s) | throughput (veh/h) | max cars queued outside | switches per junction per hour |")
    lines.append("|---|---|---|---|---|")
    for c in ctrls:
        v = ep[(ep.grid == "3x3") & (ep.controller == c)]
        lines.append(f"| {LABEL[c]} | {v.mean_travel_time_s.mean():.1f} | {v.throughput_vph.mean():.0f} | "
                     f"{v.max_backlog.mean():.0f} | {v.switches_per_node_h.mean():.0f} |")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    # ---- Figure 1: per-scenario bars on 3x3
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=False)
    for ax, g in zip(axes, ["3x3", grids[-1]]):
        x = np.arange(len(EVAL_SCENARIOS)); wdt = 0.8 / len(ctrls)
        for k, c in enumerate(ctrls):
            m = [ep[(ep.controller == c) & (ep.grid == g) & (ep.scenario == sc)][METRIC].mean() for sc in EVAL_SCENARIOS]
            e = [ci95(ep[(ep.controller == c) & (ep.grid == g) & (ep.scenario == sc)][METRIC]) for sc in EVAL_SCENARIOS]
            ax.bar(x + (k - (len(ctrls) - 1) / 2) * wdt, m, wdt, yerr=e, label=LABEL[c], color=COLOR[c], capsize=2)
        ax.set_xticks(x, EVAL_SCENARIOS)
        ax.set_ylabel("wait per vehicle (s)  — lower is better")
        tag = "trained on this size" if g in SEEN else "never trained on this size"
        ax.set_title(f"{g} grid ({tag})")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_scenarios.png"), dpi=150); plt.close(fig)

    # ---- Figure 2: generalisation across grid sizes (median over scenarios to resist outliers)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    gx = [int(g.split("x")[0]) for g in grids]
    for c in ctrls:
        m = [ep[(ep.controller == c) & (ep.grid == g)].groupby("scenario")[METRIC].mean().median() for g in grids]
        ax.plot(gx, m, "o-", label=LABEL[c], color=COLOR[c], lw=2)
    ax.axvspan(1.5, 3.5, color="#1f6fd1", alpha=0.07)
    ax.text(2.5, ax.get_ylim()[1] * 0.95, "training sizes", ha="center", va="top", fontsize=9, color="#1f6fd1")
    ax.set_xticks(gx, grids); ax.set_xlabel("grid size"); ax.set_ylabel("median over scenarios of wait per vehicle (s)")
    ax.set_title("One policy, any grid: zero-shot transfer to larger networks")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_grid_sizes.png"), dpi=150); plt.close(fig)

    # ---- Figure 3: learning curves
    curves = sorted(glob.glob("runs/*_s*/curve.csv"))
    if curves:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for p in curves:
            v = os.path.basename(os.path.dirname(p)).split("_s")[0]
            d = pd.read_csv(p)
            d = d[d.grid_nodes == 9]
            ax.plot(d.timesteps / 1e6, d.wait_per_vehicle_s.rolling(40, min_periods=5).median(),
                    color=CURVE[v], alpha=0.7, lw=1.2)
        for v in ("neighbour", "local"):
            ax.plot([], [], color=CURVE[v], label={"neighbour": "with neighbour view", "local": "local only"}[v])
        ax.set_xlabel("training samples (millions)"); ax.set_ylabel("wait per vehicle (s), 3x3 training episodes")
        ax.set_title("Learning curves (one line per training seed; random demand)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_learning.png"), dpi=150); plt.close(fig)
    print("figures saved to", OUT)


if __name__ == "__main__":
    main()
