"""
gridflow/controllers.py â€” signal controllers and the per-intersection features
used by the learning agent.

Every controller has the same interface:
  interval          how often (seconds) it is asked for a decision
  reset(net)        called at the start of an episode
  decide(net)       sets phases via net.set_phase(node, phase)

Classical baselines
  FixedTime     fixed 50/50 cycle, the way most real signals still run
  Actuated      vehicle-actuated: extend green while cars keep arriving
  MaxPressure   serve the phase with the largest (upstream - downstream) imbalance;
                a well-known adaptive rule with a stability proof (Varaiya 2013)
"""

from __future__ import annotations

import numpy as np

from gridflow.sim import Network, phase_of, turn_heading

OBS_DIM = 24
NEAR_CELLS = 8  # "about to arrive" zone in front of the stop line (60 m)


def lane_blocked(net: Network, ln) -> bool:
    """Is the front car at the stop line unable to cross because the lane it needs is full?"""
    if not ln.pos or ln.pos[0] != ln.length - 1:
        return False
    h2 = turn_heading(ln.heading, ln.turn)
    d = net.downstream[ln.node][h2]
    if d is None:
        return False
    tgt = net._target_lane(d, h2, ln.vid[0])
    return bool(tgt.pos) and tgt.pos[-1] == 0


# --------------------------------------------------------------------------- features
def node_features(net: Network, i: int, downstream: bool = True) -> list:
    """
    The 24 numbers one junction's agent sees. All in [0, 1]. Same layout at
    every junction, whatever the grid size — that is what lets ONE policy
    control ANY grid. Approaches are ordered N, E, S, W.

      [0:4]   cars on each approach (incl. entry queue) / capacity of its 3 lanes
      [4:8]   stopped cars on each approach, same scaling
      [8:12]  cars in the last 60 m before each stop line (about to arrive)
      [12:16] share of each approach's lanes whose front car is stuck because
              the lane it needs downstream is full (a green there is wasted)
      [16:20] how full each OUTGOING road is (= the neighbour's incoming road):
              the agent's view of congestion at its neighbours.
              Zeroed in the 'local-only' ablation.
      [20:22] current phase, one-hot
      [22]    time in current green / 60 s
      [23]    1 while switching (yellow / all-red)
    """
    cap = 3.0 * net.L
    f = [0.0] * OBS_DIM
    for h in range(4):
        n, q = net.approach_counts(i, h)
        f[h] = min(1.0, n / cap)
        f[4 + h] = min(1.0, q / cap)
        near = blocked = 0
        for ln in net.approach[i][h]:
            lim = ln.length - NEAR_CELLS
            for p in ln.pos:
                if p < lim:
                    break
                near += 1
            blocked += lane_blocked(net, ln)
        f[8 + h] = near / (3.0 * NEAR_CELLS)
        f[12 + h] = blocked / 3.0
        if downstream:
            f[16 + h] = net.road_occupancy(i, h)
    f[20 + net.phase[i]] = 1.0
    f[22] = min(1.0, net.tip[i] / 60.0)
    f[23] = 1.0 if net.trans[i] > 0 else 0.0
    return f


def all_features(net: Network, downstream: bool = True) -> np.ndarray:
    return np.array([node_features(net, i, downstream) for i in range(net.n_nodes)],
                    dtype=np.float32)


def node_queue(net: Network, i: int, backlog_cap: int = 50) -> float:
    """Stopped cars on node i's incoming lanes, plus its entry queue (capped)."""
    tot = 0.0
    for h in range(4):
        for ln in net.approach[i][h]:
            tot += ln.n_stopped
        s = net.source_of.get((i, h))
        if s is not None:
            tot += min(len(net.entry_q[s]), backlog_cap)
    return tot


def node_pressure(net: Network, i: int) -> float:
    """Total absolute pressure at a junction (the PressLight reward uses minus this)."""
    p = phase_pressure(net, i)
    return abs(p[0]) + abs(p[1])


def phase_pressure(net: Network, i: int) -> list:
    """
    Max-pressure (Varaiya 2013), per movement:
        pressure(lane) = density of that lane - density of the road it feeds
    Entry-queue cars count toward the boundary lanes they will join, and a
    road leaving the grid has density 0. Phase pressure = sum over its lanes.
    """
    L = net.L
    p = [0.0, 0.0]
    for h in range(4):
        s = net.source_of.get((i, h))
        extra = len(net.entry_q[s]) / 3.0 if s is not None else 0.0
        ph = h & 1
        for ln in net.approach[i][h]:
            up = (len(ln.pos) + extra) / L
            down = net.road_occupancy(i, turn_heading(h, ln.turn))
            p[ph] += up - down
    return p


# ------------------------------------------------------------------------ controllers
class Controller:
    name = "base"
    interval = 5

    def reset(self, net: Network):
        pass

    def decide(self, net: Network):
        raise NotImplementedError


class FixedTime(Controller):
    """Alternate phases every `green` seconds, regardless of traffic."""

    interval = 1

    def __init__(self, green: int = 30):
        self.green = green
        self.name = f"fixed_{green}s"

    def decide(self, net):
        for i in range(net.n_nodes):
            if net.trans[i] == 0 and net.tip[i] >= self.green:
                net.set_phase(i, 1 - net.phase[i])


class Actuated(Controller):
    """
    Classic vehicle-actuated control. Hold green at least `min_green`. After that,
    switch when (a) nobody is approaching on the green roads within `detect`
    cells of the stop line ("gap-out"), or (b) green has reached `max_green`
    ("max-out") â€” but only if someone is actually waiting on red.
    """

    interval = 1
    name = "actuated"

    def __init__(self, min_green=10, max_green=60, detect=6):
        self.min_green, self.max_green, self.detect = min_green, max_green, detect

    def decide(self, net):
        for i in range(net.n_nodes):
            if net.trans[i] > 0 or net.tip[i] < self.min_green:
                continue
            cur = net.phase[i]
            green_near = False
            red_demand = False
            for h in range(4):
                if (h & 1) == cur:
                    for ln in net.approach[i][h]:
                        if ln.pos and ln.pos[0] >= ln.length - self.detect:
                            green_near = True
                else:
                    n, _ = net.approach_counts(i, h)
                    if n:
                        red_demand = True
            if red_demand and (not green_near or net.tip[i] >= self.max_green):
                net.set_phase(i, 1 - cur)


class MaxPressure(Controller):
    """Every `interval` seconds, give green to the phase with the highest pressure."""

    name = "max_pressure"

    def __init__(self, interval: int = 5):
        self.interval = interval

    def decide(self, net):
        for i in range(net.n_nodes):
            p = phase_pressure(net, i)
            best = 0 if p[0] >= p[1] else 1
            if p[best] > p[net.phase[i]]:
                net.set_phase(i, best)


class PolicyController(Controller):
    """Runs a trained shared policy: the same network decides for every intersection."""

    def __init__(self, model, name="rl_shared", downstream=True, interval=5, deterministic=True):
        self.model = model
        self.name = name
        self.downstream = downstream
        self.interval = interval
        self.deterministic = deterministic

    def decide(self, net):
        obs = all_features(net, self.downstream)
        actions, _ = self.model.predict(obs, deterministic=self.deterministic)
        for i, a in enumerate(np.asarray(actions).reshape(-1)):
            net.set_phase(i, int(a))


# ----------------------------------------------------------------------------- runner
def run_episode(net: Network, ctrl: Controller, seed: int, steps: int = 3600,
                on_step=None) -> dict:
    """Reset the network with `seed`, run `steps` seconds under `ctrl`, return metrics."""
    net.reset(seed)
    ctrl.reset(net)
    for _ in range(steps):
        if net.t % ctrl.interval == 0:
            ctrl.decide(net)
        net.step()
        if on_step is not None:
            on_step(net)
    return net.metrics()
