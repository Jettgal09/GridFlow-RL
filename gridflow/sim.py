"""
gridflow/sim.py — a small, fast, cell-based traffic simulator for a grid of
signalised intersections.

MODEL IN ONE PARAGRAPH
  Every approach to a junction has three lanes — left, straight, right — and a
  car picks its lane from its next turn. Each lane is a row of cells; one cell
  holds at most one car. Time advances in steps of 1 second. A car moves forward
  one cell per step if the cell in front of it was empty at the START of the step
  (a cellular-automaton rule). That single rule gives realistic behaviour:
  a stopped queue discharges at one car every 2 s per lane (1800 veh/h, the
  textbook saturation flow), queues form behind red lights, and a full
  downstream lane blocks cars from entering it (spillback). Cars that cannot
  enter the grid because their lane's first cell is occupied wait in an entry
  queue outside it, and that waiting is counted — a controller cannot "hide"
  demand by blocking entry (the reward hack seen in the earlier project).

UNITS
  1 step = 1 s. 1 cell = 7.5 m. Free speed = 1 cell/s = 27 km/h (urban).

GEOMETRY
  Junctions ("nodes") sit on a rows x cols grid. A car heading N arrives from
  the south. Roads between neighbouring junctions are internal; roads on the
  boundary are fed by an entry queue. A car leaving the grid is finished the
  moment it crosses the boundary junction.

SIGNALS
  Two phases: 0 = North-South green, 1 = East-West green. Changing phase costs a
  fixed transition (yellow + all-red, default 3 s) during which nobody crosses.
  Left turns are treated as protected (no conflict with oncoming traffic) — a
  stated simplification.

RANDOMNESS
  Two independent random streams: arrivals, and turning choices. Turning choices
  are pre-drawn per vehicle when it is created. Arrivals never depend on the
  controller and each vehicle's route is fixed at birth, so two controllers run
  with the same seed face EXACTLY the same traffic (paired comparison).
"""

from __future__ import annotations

from collections import deque

import numpy as np

# Headings, clockwise. Index arithmetic below depends on this order.
N, E, S, W = 0, 1, 2, 3
HEADING_NAMES = "NESW"
DR = (-1, 0, 1, 0)  # row change when travelling in heading h
DC = (0, 1, 0, -1)  # col change

STRAIGHT, LEFT, RIGHT = 0, 1, 2
TURN_NAMES = ("straight", "left", "right")
MAX_TURN_DRAWS = 16  # after this many junctions a car just goes straight (so it always exits)


def turn_heading(h: int, turn: int) -> int:
    """Heading after a turn (right-hand traffic): N + left = W, N + right = E."""
    if turn == STRAIGHT:
        return h
    if turn == LEFT:
        return (h - 1) % 4
    return (h + 1) % 4


def phase_of(h: int) -> int:
    """Which signal phase serves heading h: N/S -> 0, E/W -> 1."""
    return h % 2


class Lane:
    """One lane of one approach, stored as a front-first list of cars."""

    __slots__ = ("id", "length", "node", "heading", "turn", "pos", "vid",
                 "departed", "n_stopped", "incident_k")

    def __init__(self, lid, length, node, heading, turn):
        self.id = lid
        self.length = length
        self.node = node        # junction at the downstream end
        self.heading = heading  # travel heading of cars in this lane
        self.turn = turn        # movement this lane serves at the junction
        self.pos = []           # cell index of each car, front (largest) first
        self.vid = []           # vehicle id of each car, same order
        self.departed = False   # did the front car cross the junction this step?
        self.n_stopped = 0      # cars that did not move this step
        self.incident_k = 0     # >0 while an incident is active: front car may cross only every k-th second


class Network:
    def __init__(self, rows: int, cols: int, scenario, seed: int = 0,
                 lane_len: int = 25, transition: int = 3):
        self.rows, self.cols = rows, cols
        self.n_nodes = rows * cols
        self.L = lane_len
        self.transition = transition
        self.scenario = scenario

        # lanes[node][h][turn] -> Lane ; flat list in self.lanes
        self.lanes: list[Lane] = []
        self.approach = [[None] * 4 for _ in range(self.n_nodes)]
        self.upstream = [[None] * 4 for _ in range(self.n_nodes)]
        self.downstream = [[None] * 4 for _ in range(self.n_nodes)]
        for node in range(self.n_nodes):
            r, c = divmod(node, cols)
            for h in range(4):
                self.approach[node][h] = [
                    self._add_lane(node, h, t) for t in (STRAIGHT, LEFT, RIGHT)
                ]
                ur, uc = r - DR[h], c - DC[h]
                self.upstream[node][h] = ur * cols + uc if (0 <= ur < rows and 0 <= uc < cols) else None
                dr, dc = r + DR[h], c + DC[h]
                self.downstream[node][h] = dr * cols + dc if (0 <= dr < rows and 0 <= dc < cols) else None

        # Entry points: one per boundary approach (node, heading)
        self.sources = []        # (node, heading)
        self.source_of = {}      # (node, heading) -> source index
        for node in range(self.n_nodes):
            for h in range(4):
                if self.upstream[node][h] is None:
                    self.source_of[(node, h)] = len(self.sources)
                    self.sources.append((node, h))
        self.source_meta = [(h, *divmod(node, cols)) for node, h in self.sources]

        self.scenario.bind(self)
        self.reset(seed)

    def _add_lane(self, node, h, t):
        ln = Lane(len(self.lanes), self.L, node, h, t)
        self.lanes.append(ln)
        return ln

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int = 0):
        self.seed = seed
        self.scenario.on_reset(seed)
        ss = np.random.SeedSequence(seed)
        a, b = ss.spawn(2)
        self.rng_arr = np.random.default_rng(a)
        self.rng_turn = np.random.default_rng(b)
        self.t = 0

        for ln in self.lanes:
            ln.pos.clear(); ln.vid.clear()
            ln.departed = False; ln.n_stopped = 0; ln.incident_k = 0
        self.entry_q = [deque() for _ in self.sources]

        self.phase = [0] * self.n_nodes
        self.next_phase = [0] * self.n_nodes
        self.trans = [0] * self.n_nodes
        self.tip = [0] * self.n_nodes      # seconds of green in the current phase
        self.switches = 0

        # vehicle storage (index = vehicle id)
        self.v_create = []
        self.v_enter = []
        self.v_wait = []
        self.v_turns = []
        self.v_tidx = []
        self.v_next = []
        cum = np.cumsum(self.scenario.turn_probs)
        self._turn_cum = cum / cum[-1]

        # metrics
        self.spawned = 0
        self.completed = 0
        self.sum_travel = 0.0
        self.sum_wait = 0.0
        self.total_stopped = 0       # vehicle-seconds spent not moving (incl. entry queues)
        self.stopped_now = 0
        self.max_backlog = 0
        self.finished_this_step = 0

    # ------------------------------------------------------------- signals
    def set_phase(self, node: int, p: int):
        """Request phase p at a node. Starts a transition if it differs."""
        if self.trans[node] == 0 and p != self.phase[node]:
            self.trans[node] = self.transition
            self.next_phase[node] = p
            self.switches += 1

    def is_green(self, node: int, h: int) -> bool:
        return self.trans[node] == 0 and self.phase[node] == phase_of(h)

    # ------------------------------------------------------------ vehicles
    def _new_vehicle(self) -> int:
        vid = len(self.v_create)
        self.v_create.append(self.t)
        self.v_enter.append(-1)
        self.v_wait.append(0)
        u = self.rng_turn.random(MAX_TURN_DRAWS)
        self.v_turns.append(np.searchsorted(self._turn_cum, u, side="right").tolist())
        self.v_tidx.append(0)
        self.v_next.append(STRAIGHT)
        self.spawned += 1
        return vid

    def _peek_turn(self, vid: int) -> int:
        i = self.v_tidx[vid]
        return self.v_turns[vid][i] if i < MAX_TURN_DRAWS else STRAIGHT

    def _target_lane(self, node: int, h: int, vid: int) -> Lane:
        """The lane a vehicle must join when it arrives at `node` heading `h`."""
        return self.approach[node][h][self._peek_turn(vid)]

    def _enter(self, ln: Lane, vid: int):
        ln.pos.append(0)
        ln.vid.append(vid)
        self.v_next[vid] = ln.turn
        self.v_tidx[vid] += 1

    def _finish(self, vid: int):
        self.completed += 1
        self.finished_this_step += 1
        self.sum_travel += self.t + 1 - self.v_create[vid]
        self.sum_wait += self.v_wait[vid] + (self.v_enter[vid] - self.v_create[vid])

    # ---------------------------------------------------------------- step
    def step(self):
        t = self.t
        lanes = self.lanes
        self.finished_this_step = 0
        self.scenario.apply_events(self, t)

        # 0. arrivals into entry queues (independent of control)
        rates = self.scenario.rates(t)
        arrive = self.rng_arr.random(len(self.sources)) < rates
        for s in np.flatnonzero(arrive):
            self.entry_q[s].append(self._new_vehicle())

        # snapshot: is cell 0 of each lane free at the start of this step?
        tail_free = [(not ln.pos) or ln.pos[-1] > 0 for ln in lanes]
        reserved = [False] * len(lanes)
        transfers = []

        # A. junction crossings (front car of each lane)
        for ln in lanes:
            ln.departed = False
            if not ln.pos or ln.pos[0] != ln.length - 1:
                continue
            node, h = ln.node, ln.heading
            if self.trans[node] or self.phase[node] != (h & 1):
                continue
            if ln.incident_k and (t % ln.incident_k):
                continue
            vid = ln.vid[0]
            h2 = turn_heading(h, ln.turn)
            d = self.downstream[node][h2]
            if d is None:  # leaves the grid
                ln.pos.pop(0); ln.vid.pop(0); ln.departed = True
                self._finish(vid)
                continue
            tgt = self._target_lane(d, h2, vid)
            if tail_free[tgt.id] and not reserved[tgt.id]:
                reserved[tgt.id] = True
                ln.pos.pop(0); ln.vid.pop(0); ln.departed = True
                transfers.append((tgt, vid))

        # B. move cars forward inside each lane (snapshot rule)
        v_wait = self.v_wait
        stopped_total = 0
        for ln in lanes:
            pos = ln.pos
            n = len(pos)
            if n == 0:
                ln.n_stopped = 0
                continue
            prev = ln.length - 1 if ln.departed else ln.length
            vids = ln.vid
            stopped = 0
            for k in range(n):
                p = pos[k]
                if p + 1 < prev:
                    pos[k] = p + 1
                else:
                    stopped += 1
                    v_wait[vids[k]] += 1
                prev = p
            ln.n_stopped = stopped
            stopped_total += stopped

        # C. crossing cars land in their next lane; entry queues feed the boundary
        for tgt, vid in transfers:
            self._enter(tgt, vid)
        backlog = 0
        for s, (node, h) in enumerate(self.sources):
            q = self.entry_q[s]
            if q:
                tgt = self._target_lane(node, h, q[0])
                if tail_free[tgt.id] and not reserved[tgt.id]:
                    reserved[tgt.id] = True
                    vid = q.popleft()
                    self.v_enter[vid] = t
                    self._enter(tgt, vid)
            backlog += len(q)
        if backlog > self.max_backlog:
            self.max_backlog = backlog

        self.stopped_now = stopped_total + backlog
        self.total_stopped += self.stopped_now

        # D. signals tick
        for i in range(self.n_nodes):
            if self.trans[i] > 0:
                self.trans[i] -= 1
                if self.trans[i] == 0:
                    self.phase[i] = self.next_phase[i]
                    self.tip[i] = 0
            else:
                self.tip[i] += 1

        self.t = t + 1

    # ------------------------------------------------------------- queries
    def approach_counts(self, node: int, h: int):
        """(cars, stopped cars) on approach (node, h), including its entry queue."""
        n = q = 0
        for ln in self.approach[node][h]:
            n += len(ln.pos)
            q += ln.n_stopped
        s = self.source_of.get((node, h))
        if s is not None:
            b = len(self.entry_q[s])
            n += b
            q += b
        return n, q

    def road_occupancy(self, node: int, h: int) -> float:
        """How full (0-1) is the road leaving `node` in heading h? 0 if it leaves the grid."""
        d = self.downstream[node][h]
        if d is None:
            return 0.0
        return sum(len(ln.pos) for ln in self.approach[d][h]) / (3 * self.L)

    def total_backlog(self) -> int:
        return sum(len(q) for q in self.entry_q)

    def vehicles_in_network(self) -> int:
        return sum(len(ln.pos) for ln in self.lanes)

    def metrics(self) -> dict:
        """Episode metrics. All times in seconds. Each name says exactly what it measures."""
        steps = max(self.t, 1)
        return {
            "steps": self.t,
            "spawned": self.spawned,
            "completed": self.completed,
            "throughput_vph": self.completed * 3600.0 / steps,
            "mean_travel_time_s": self.sum_travel / max(self.completed, 1),
            "mean_wait_per_trip_s": self.sum_wait / max(self.completed, 1),
            "total_wait_veh_h": self.total_stopped / 3600.0,
            "wait_per_vehicle_s": self.total_stopped / max(self.spawned, 1),
            "mean_stopped_vehicles": self.total_stopped / steps,
            "unfinished": self.spawned - self.completed,
            "final_backlog": self.total_backlog(),
            "max_backlog": self.max_backlog,
            "switches_per_node_h": self.switches * 3600.0 / steps / self.n_nodes,
        }
