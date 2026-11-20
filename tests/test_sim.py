"""Physics and bookkeeping checks for the simulator. Run: python -m pytest -q"""

import numpy as np

from gridflow import scenarios as S
from gridflow.controllers import FixedTime, MaxPressure, run_episode
from gridflow.sim import E, N, STRAIGHT, Network


class Const(S.Scenario):
    """Fixed rate on chosen headings only, everyone goes straight."""

    def __init__(self, rate, headings=(N, E, 2, 3)):
        super().__init__(turn_probs=(1.0, 0.0, 0.0))
        self.rate, self.hs = rate, headings

    def rates(self, t):
        return np.array([self.rate if m[0] in self.hs else 0.0 for m in self.meta])


def test_vehicle_conservation():
    """Every car created is finished, on the road, or still queued to enter."""
    for sc in S.EVAL_SCENARIOS:
        net = Network(3, 3, S.make(sc))
        m = run_episode(net, FixedTime(30), seed=3, steps=1200)
        assert m["spawned"] == m["completed"] + net.vehicles_in_network() + net.total_backlog()


def test_saturation_flow_is_half_a_car_per_second():
    """A permanently green, saturated straight lane serves ~0.5 veh/s (1800 veh/h)."""
    net = Network(1, 1, Const(1.0, headings=(N,)))
    net.reset(0)
    for _ in range(2000):
        net.step()  # phase 0 (N-S) stays green forever: nobody switches it
    rate = net.completed / 2000
    assert 0.45 < rate <= 0.5


def test_red_light_stops_everyone():
    net = Network(1, 1, Const(0.3, headings=(E,)))
    net.reset(0)
    for _ in range(500):
        net.step()  # E-W never gets green
    assert net.completed == 0
    assert net.vehicles_in_network() > 0


def test_spillback_blocks_upstream():
    """If the downstream lane is full, cars wait at a green light instead of entering."""
    net = Network(1, 2, Const(0.0))  # two junctions, left (0) -> right (1)
    net.reset(0)
    full = net.approach[1][E][STRAIGHT]
    for p in range(full.length):  # fill the right junction's eastbound straight lane
        full.pos.insert(0, p); full.vid.insert(0, net._new_vehicle())
    src = net.approach[0][E][STRAIGHT]
    vid = net._new_vehicle()
    net.v_turns[vid] = [STRAIGHT] * 16
    src.pos.append(src.length - 1); src.vid.append(vid)
    net.set_phase(0, 1); net.set_phase(1, 0)  # left junction E-W green, right junction red for E
    for _ in range(50):
        net.step()
    assert src.pos and src.pos[0] == src.length - 1  # still waiting at the stop line


def test_same_seed_same_traffic():
    """Arrivals and routes do not depend on the controller (paired comparison)."""
    a = Network(2, 2, S.make("uniform")); run_episode(a, FixedTime(30), seed=11, steps=900)
    b = Network(2, 2, S.make("uniform")); run_episode(b, MaxPressure(10), seed=11, steps=900)
    assert a.spawned == b.spawned
    assert a.v_create == b.v_create
    assert a.v_turns == b.v_turns


def test_deterministic():
    m1 = run_episode(Network(2, 2, S.make("rush_hour")), MaxPressure(10), seed=5, steps=900)
    m2 = run_episode(Network(2, 2, S.make("rush_hour")), MaxPressure(10), seed=5, steps=900)
    assert m1 == m2
