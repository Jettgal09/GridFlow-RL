"""
Train the shared signal-control policy with PPO.

  uv run python scripts/train.py --seed 0                    # neighbour-aware (default)
  uv run python scripts/train.py --seed 0 --variant local    # ablation: no downstream info

Output: runs/<variant>_s<seed>/model.zip and curve.csv (one row per finished
training episode). The FINAL model is kept — no best-checkpoint cherry-picking.
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from gridflow.marl import SharedPolicyVecEnv


class CurveLogger(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["timesteps", "grid_nodes", "wait_per_vehicle_s", "throughput_vph", "final_backlog"])

    def _on_step(self):
        env = self.training_env
        for k, m in env.finished_episodes:
            self.w.writerow([self.num_timesteps, env.sims[k].n_nodes,
                             round(m["wait_per_vehicle_s"], 2), round(m["throughput_vph"], 1),
                             m["final_backlog"]])
        env.finished_episodes.clear()
        self.f.flush()
        return True

    def _on_training_end(self):
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", choices=["neighbour", "local"], default="neighbour")
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--dt", type=int, default=5, help="seconds between decisions")
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--reward", choices=["queue", "pressure"], default="queue")
    ap.add_argument("--share", type=float, default=0.0,
                    help="weight of neighbours' mean queue in each junction's reward (cooperation)")
    ap.add_argument("--grids", default="3x3,3x3,3x3,3x3,2x2,2x2,2x2,2x2",
                    help="training grids (rows x cols); 4x4 and 6x6 are kept for zero-shot testing")
    args = ap.parse_args()

    torch.set_num_threads(1)
    out = os.path.join(args.out, f"{args.variant}_s{args.seed}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    grids = [tuple(int(x) for x in g.split("x")) for g in args.grids.split(",")]
    env = SharedPolicyVecEnv(grid_sizes=grids, downstream=(args.variant == "neighbour"), seed=args.seed,
                             decision_interval=args.dt, reward=args.reward, share=args.share)
    model = PPO(
        "MlpPolicy", env,
        n_steps=128,            # 128 decisions x 52 junctions = 6656 samples per update
        batch_size=env.num_envs * 128 // 8,
        n_epochs=8,
        learning_rate=3e-4,
        gamma=args.gamma,       # 0.97 with 5 s decisions -> looks ~3 minutes ahead
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=args.seed,
        verbose=0,
        device="cpu",
    )
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=CurveLogger(os.path.join(out, "curve.csv")))
    model.save(os.path.join(out, "model"))
    print(f"[{args.variant} s{args.seed}] done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
