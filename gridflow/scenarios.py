"""
gridflow/scenarios.py — traffic demand patterns and disruptions.

A scenario answers two questions for the simulator:
  rates(t)               -> arrival probability per second at every entry point
  apply_events(net, t)   -> switch disruptions (incidents) on or off

Rates are vehicles per second arriving at each boundary approach. Each lane
serves at most 0.5 veh/s while green. With 60% of cars going straight and a
fixed 50/50 split, an approach saturates at roughly 0.38 veh/s.
"""

from __future__ import annotations

import numpy as np

from gridflow.sim import E, N, S, W

DEFAULT_TURNS = (0.60, 0.20, 0.20)  # straight, left, right


class Scenario:
    name = "base"

    def __init__(self, turn_probs=DEFAULT_TURNS):
        self.turn_probs = tuple(turn_probs)
        self.net = None

    def bind(self, net):
        self.net = net
        self.meta = net.source_meta
        self.heading = np.array([m[0] for m in self.meta])

    def on_reset(self, seed: int):
        pass

    def rates(self, t: int) -> np.ndarray:
        raise NotImplementedError

    def apply_events(self, net, t: int):
        pass

    def describe(self) -> str:
        return self.name


class Uniform(Scenario):
    """Same arrival rate at every entry."""

    def __init__(self, rate=0.30, **kw):
        super().__init__(**kw)
        self.rate = rate
        self.name = f"uniform_{rate:.2f}"

    def rates(self, t):
        return np.full(len(self.meta), self.rate)


class Arterial(Scenario):
    """A busy main road on one axis, quiet side streets on the other."""

    def __init__(self, main=0.42, side=0.10, axis="EW", **kw):
        super().__init__(**kw)
        self.main, self.side, self.axis = main, side, axis
        self.name = f"arterial_{axis}"

    def bind(self, net):
        super().bind(net)
        main_axis = 1 if self.axis == "EW" else 0
        self._r = np.where(self.heading % 2 == main_axis, self.main, self.side)

    def rates(self, t):
        return self._r


class RushHour(Scenario):
    """
    One hour with a morning peak: demand ramps from light to heavy and back,
    and the peak is directional (more traffic heading north and east, i.e. into
    the 'city centre' at the top-right).
    """

    def __init__(self, low=0.10, peak=0.38, horizon=3600, **kw):
        super().__init__(**kw)
        self.low, self.peak, self.horizon = low, peak, horizon
        self.name = "rush_hour"

    def bind(self, net):
        super().bind(net)
        self._bias = np.where(np.isin(self.heading, (N, E)), 1.0, 0.6)

    def level(self, t):
        x = t / self.horizon
        # trapezoid: ramp up 0-0.3, hold 0.3-0.6, ramp down 0.6-1.0
        if x < 0.3:
            f = x / 0.3
        elif x < 0.6:
            f = 1.0
        else:
            f = max(0.0, 1 - (x - 0.6) / 0.4)
        return self.low + (self.peak - self.low) * f

    def rates(self, t):
        return self.level(t) * self._bias


class Incident(Scenario):
    """
    Base demand plus a crash that blocks most of one busy road for a while.
    While active, the front car on that road can leave only every k-th second
    (k=3 -> the road keeps about a third of its capacity).
    """

    def __init__(self, base: Scenario | None = None, t_start=900, t_end=2400, k=3, **kw):
        base = base or Arterial(main=0.34, side=0.14)
        super().__init__(turn_probs=base.turn_probs)
        self.base, self.t_start, self.t_end, self.k = base, t_start, t_end, k
        self.name = "incident"

    def bind(self, net):
        super().bind(net)
        self.base.bind(net)
        # Block the eastbound approach of a central junction (an internal road
        # whenever the grid has more than one column).
        r = net.rows // 2
        c = min(max(1, net.cols // 2), net.cols - 1)
        self.node = r * net.cols + c
        self.lanes = net.approach[self.node][E]

    def rates(self, t):
        return self.base.rates(t)

    def apply_events(self, net, t):
        if t == self.t_start:
            for ln in self.lanes:
                ln.incident_k = self.k
        elif t == self.t_end:
            for ln in self.lanes:
                ln.incident_k = 0


class Surge(Scenario):
    """Uniform background, then a sudden 15-minute flood from the west (e.g. a stadium emptying)."""

    def __init__(self, base_rate=0.15, surge_rate=0.60, t_start=1200, t_end=2100, **kw):
        super().__init__(**kw)
        self.base_rate, self.surge_rate = base_rate, surge_rate
        self.t_start, self.t_end = t_start, t_end
        self.name = "surge"

    def rates(self, t):
        r = np.full(len(self.meta), self.base_rate)
        if self.t_start <= t < self.t_end:
            r[self.heading == E] = self.surge_rate
        return r


class RandomTraining(Scenario):
    """
    The training distribution. Every episode draws a fresh pattern: either
    uniform demand at a random level, or an arterial on a random axis with
    random main/side rates. Rush hour, incidents and surges are deliberately
    NOT in here — they are held out to test generalisation.
    """

    name = "random_train"

    def __init__(self, **kw):
        super().__init__(**kw)
        self._r = None

    def bind(self, net):
        super().bind(net)

    def on_reset(self, seed):
        rng = np.random.default_rng(seed + 7_919)
        if self.net is None:
            return
        n = len(self.meta)
        if rng.random() < 0.5:
            self._r = np.full(n, rng.uniform(0.05, 0.36))
        else:
            main_axis = int(rng.integers(2))
            main, side = rng.uniform(0.20, 0.46), rng.uniform(0.03, 0.16)
            self._r = np.where(self.heading % 2 == main_axis, main, side)
        # small per-entry jitter so no two entries are identical
        self._r = np.clip(self._r * rng.uniform(0.8, 1.2, n), 0.0, 0.9)

    def rates(self, t):
        return self._r


def make(name: str, **kw) -> Scenario:
    """Factory for the named evaluation scenarios."""
    table = {
        "light": lambda: Uniform(0.15),
        "uniform": lambda: Uniform(0.30),
        "arterial": lambda: Arterial(),
        "rush_hour": lambda: RushHour(),
        "incident": lambda: Incident(),
        "surge": lambda: Surge(),
        "train": lambda: RandomTraining(),
    }
    sc = table[name]()
    sc.name = name
    return sc


EVAL_SCENARIOS = ["light", "uniform", "arterial", "rush_hour", "surge", "incident"]
