# Learn this project from zero

This is a study guide for the person presenting the project. No prior knowledge is
assumed. Read it top to bottom once, then use the viva section to practise.

---

## 1. The problem in one minute

A city has a grid of junctions with traffic lights. Each light decides, again and
again, *which direction gets green*. Bad decisions make queues grow; queues on one
road spill back and block the junction behind it; one blocked junction can freeze
a whole area (**gridlock**).

Most real lights run on a **fixed timer**: "30 s green for north–south, then 30 s
for east–west", whether anyone is there or not. Better controllers look at the
traffic and react. This project asks:

> Can **one** small learned controller, trained on small grids, run every
> junction of a city it has never seen — and handle situations it never saw
> (rush hour, a sudden surge, a crash blocking a road) — better than the
> classical methods traffic engineers use?

---

## 2. The simulator (`gridflow/sim.py`)

We built our own because we need thousands of simulated hours and full control of
the rules. It is deliberately simple, and every rule can be explained in a sentence.

| Idea | What it means here |
|---|---|
| **Time step** | 1 second. |
| **Cell** | A road is cut into cells of 7.5 m. One car fits in one cell. |
| **Lane** | Every approach to a junction has 3 lanes: left, straight, right. A car picks its lane from where it will turn. |
| **Movement rule** | A car moves forward one cell per second **if the cell in front was empty at the start of the second**. |
| **Why that rule?** | It automatically gives a realistic queue discharge of one car every 2 s per lane (1800 cars/hour — the number traffic engineers use as "saturation flow"). |
| **Spillback** | If the lane a car needs is full, the car waits even on green. This is how jams spread backwards. |
| **Entry queue** | Cars arriving at the edge of the grid when their lane is full wait *outside*. That waiting is counted. |
| **Phases** | Two: north–south green or east–west green. Switching costs 3 s (yellow + all-red) when nobody moves. |
| **Fair comparison** | Arrivals and every car's route are fixed by a random seed *before* any controller acts, so two controllers with the same seed face identical traffic. |

### Why we did not reuse the old simulator
We audited the earlier project and found:
1. Cars entered a lane even when it was full (no spillback), so cars stacked on top of each other.
2. New cars were spawned on top of each other at the map edge.
3. The arrival rate was higher than the junction could ever serve, so queues always grew to a hard cap; "good control" mostly meant "pick long greens".
4. The reported "average waiting time" actually summed every car's *running total* every second, which counts long waits many times over (that is why it showed 15 million).
5. The random number generator was not seeded, so results could not be reproduced.

The new simulator fixes all five, and `tests/test_sim.py` checks the key physics.

---

## 3. The controllers we compare (`gridflow/controllers.py`)

| Controller | How it decides | Real-world status |
|---|---|---|
| **Fixed-time** | Switch every 15 s, always. | What most cities still run. |
| **Vehicle-actuated** | Keep green while cars keep arriving at the stop line ("gap-out"); switch after 40 s max; never under 5 s. | Very common in practice (detector loops in the road). |
| **Max-pressure** | Every 20 s, give green to the direction with the biggest *pressure* = cars waiting upstream minus cars already on the road they want to enter. | The best-known adaptive method in research; it has a mathematical stability proof. |
| **Our RL agent** | A small neural network looks at 16 numbers about its junction and picks a phase every 10 s. | Learned, not hand-designed. |

All baselines were tuned on separate "tuning" seeds so they are as strong as
possible — we are not beating straw men.

---

## 4. Reinforcement learning in plain words

- An **agent** takes an **action**, the world changes, and the agent gets a
  **reward** (a number saying how good that was).
- It repeats this millions of times and gradually prefers actions that lead to
  higher total reward. That's all "learning" means here.
- **Our agent**: every junction is an agent. Action = which phase to show for the
  next 10 s. Reward = minus the number of stopped cars at that junction
  (fewer waiting cars = higher reward).
- **Observation** (what it sees), 16 numbers between 0 and 1:
  - cars on each of its 4 incoming roads
  - stopped cars on each of its 4 incoming roads
  - **how full each of its 4 outgoing roads are** ← this is how it "sees" the
    neighbouring junctions' congestion
  - which phase is on, how long it has been on, whether it is switching
- **Policy** = the neural network that maps those 16 numbers to "show NS" or "show EW".
- **PPO** (Proximal Policy Optimisation) = the standard, stable algorithm we used
  to train the network. Key idea: improve the policy a little at a time, never
  jumping too far from what already works ("clipping").

### The key design choice: one shared policy
Every junction uses **the same network** ("parameter sharing"). Because the 16
inputs mean the same thing at every junction, the same network works on a 2×2
grid, a 3×3 grid, or a 6×6 grid. We train on 2×2 and 3×3 together and then test,
without any retraining, on 4×4 and 6×6. That is **zero-shot transfer**.

The old project used one big network that saw the whole grid at once. It could
not be applied to a different grid size at all (its input size was fixed to the
grid) and needed 4^N joint choices.

### The ablation
An **ablation** removes one ingredient to see whether it mattered. We train a
second version that gets the same inputs **except** the "how full are my
outgoing roads" numbers (set to zero). If the full version is better, the
neighbour information is doing real work.

### Things we did to keep results honest
- **Held-out scenarios**: training only uses random steady demand. Rush hour,
  surge and incident are never seen in training.
- **Held-out grid sizes**: 4×4 and 6×6 never seen.
- **Held-out seeds**: test traffic (seeds 10000+) never overlaps training traffic.
- **5 training seeds** per variant, reported together — not the luckiest one.
- **Final model**, not the best checkpoint.
- **Paired comparison + confidence intervals**: same traffic for every controller;
  we bootstrap the difference to see whether it is real or noise.

---

## 5. The metrics

| Metric | Meaning |
|---|---|
| **Wait per vehicle (s)** — main metric | Total seconds all cars spent stopped (inside the grid *and* queued outside it), divided by the number of cars that arrived. Lower is better. |
| Mean travel time (s) | Average time from arriving at the edge to leaving the grid, for finished trips. |
| Throughput (veh/h) | Finished trips per hour. |
| Max cars queued outside | Worst entry backlog — a sign of gridlock or starvation. |
| Switches per junction per hour | How often lights change. |

**Bootstrap confidence interval**: we resample the 10 paired differences thousands
of times; if the middle 95% of the averages is entirely below zero, RL is reliably
better in that setting.

---

## 6. Results

See `results/results.md` for the full tables and the README for the headline
numbers. The summary and its interpretation are filled in once the experiments
are final.

---

## 7. Viva questions and short answers

**Why build your own simulator instead of SUMO?**
SUMO is the industry standard and is a natural next step, but for a mini project we
needed something we fully understand, can test, and can run thousands of hours of
traffic on a laptop in minutes. Ours reproduces the key effects (saturation flow,
queues, spillback, gridlock) with rules short enough to explain.

**What are the simplifications?**
Single speed, no acceleration modelling, left turns treated as protected (no
conflict with oncoming cars), no pedestrians, two-phase signals, synthetic demand.

**Why is the agent's reward the queue length?**
It is local (each junction can influence it), immediate (fast learning signal), and
directly tied to waiting time. Entry queues are included so it cannot hide traffic.

**Why decide every 10 seconds?**
Switching costs 3 s of lost time. Deciding every 5 s let max-pressure switch so often
that the network gridlocked; 10 s is also standard in the literature.

**What is gamma (0.95)?**
How much the agent cares about future reward. With a decision every 10 s, 0.95 means
it effectively looks about 3 minutes ahead.

**Is this really multi-agent?**
Yes — each junction acts on its own local view and its own reward. They share the
same network weights (parameter sharing), which is a standard, well-studied setup.

**Why did max-pressure gridlock in the surge?**
Max-pressure counts how full the downstream *road* is on average, but a car needs
one specific *lane*. In a sudden flood it can keep giving green to cars that cannot
physically move, until a closed loop of full lanes forms and nothing can move.

**What would you do next?**
Port to SUMO with real map data, add pedestrian phases and protected-permissive
lefts, train with real detector data, and add explicit communication between
neighbouring agents (e.g. graph neural networks).
