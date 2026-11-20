"""
Train the shared signal-control policy with PPO.

  uv run python scripts/train.py --seed 0                    # neighbour-aware (default)
  uv run python scripts/train.py --seed 0 --variant local    # ablation: no downstream info

Output: runs/<variant>_s<seed>/model.zip and curve.csv (one row per finished
training episode). The FINAL model is kept — no best-checkpoint cherry-picking.
"""

import argparse
import csv
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
        return True

    def _on_training_end(self):
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", choices=["neighbour", "local"], default="neighbour")
    ap.add_argument("--timesteps", type=int, default=1_500_000)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    torch.set_num_threads(1)
    out = os.path.join(args.out, f"{args.variant}_s{args.seed}")
    os.makedirs(out, exist_ok=True)

    env = SharedPolicyVecEnv(downstream=(args.variant == "neighbour"), seed=args.seed)
    model = PPO(
        "MlpPolicy", env,
        n_steps=128,            # 128 decisions x 52 junctions = 6656 samples per update
        batch_size=832,
        n_epochs=8,
        learning_rate=3e-4,
        gamma=0.95,             # decisions every 10 s -> looks ~3 minutes ahead
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
