"""
Evaluate every controller on every scenario and grid size, on identical traffic.

  uv run python scripts/evaluate.py --runs runs --out results/eval.csv

- Test seeds (10_000+) never overlap training seeds.
- Every controller sees exactly the same arrivals and routes for a given test seed
  (paired comparison), so differences come from control, not luck.
- 2x2 and 3x3 were seen in training; 4x4 and 6x6 were NEVER seen (zero-shot).
- Rush hour, surge and incident were NEVER in the training distribution.
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from gridflow import scenarios as S
from gridflow.controllers import Actuated, FixedTime, MaxPressure, run_episode
from gridflow.sim import Network

GRIDS = [(2, 2), (3, 3), (4, 4), (6, 6)]
TEST_SEEDS = [10_000 + i for i in range(10)]

# Tuned once on separate tuning seeds (500-502) across all scenarios on 3x3; see README.
BASELINES = {
    "fixed_time": lambda: FixedTime(15),
    "actuated": lambda: Actuated(5, 40, 4),
    "max_pressure": lambda: MaxPressure(20),
}


def make_controller(spec):
    kind, arg = spec
    if kind == "baseline":
        return BASELINES[arg]()
    import torch
    from gridflow.policy_io import load_controller
    torch.set_num_threads(1)
    return load_controller(arg)


def job(task):
    spec, grids, scenarios = task
    ctrl = make_controller(spec)
    train_seed = int(os.path.basename(os.path.dirname(spec[1])).split("_s")[1]) if spec[0] == "rl" else -1
    rows = []
    for (r, c) in grids:
        for sc in scenarios:
            net = Network(r, c, S.make(sc))
            for s in TEST_SEEDS:
                m = run_episode(net, ctrl, seed=s, steps=3600)
                rows.append(dict(controller=ctrl.name, train_seed=train_seed, grid=f"{r}x{c}",
                                 scenario=sc, test_seed=s, **m))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/eval.csv")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    specs = [("baseline", k) for k in BASELINES]
    specs += [("rl", p) for p in sorted(glob.glob(os.path.join(args.runs, "*_s*", "model.zip")))]
    # one task per (controller, grid, scenario) for good load balancing
    tasks = [(sp, [g], [sc]) for sp in specs for g in GRIDS for sc in S.EVAL_SCENARIOS]
    print(f"{len(specs)} controllers x {len(GRIDS)} grids x {len(S.EVAL_SCENARIOS)} scenarios "
          f"x {len(TEST_SEEDS)} seeds = {len(tasks) * len(TEST_SEEDS)} episodes")

    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(job, tasks, chunksize=1)):
            rows.extend(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tasks)} tasks")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
