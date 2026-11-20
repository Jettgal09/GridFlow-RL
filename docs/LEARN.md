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
| **Our RL agent (GridFlow-RL)** | A small neural network looks at 24 numbers about its junction and picks a phase every 5 s; a one-line safety shield can overrule it. | Learned, not hand-designed. |

All baselines were tuned on separate "tuning" seeds so they are as strong as
possible — we are not beating straw men.

---

## 4. Reinforcement learning in plain words

- An **agent** takes an **action**, the world changes, and the agent gets a
  **reward** (a number saying how good that was).
- It repeats this millions of times and gradually prefers actions that lead to
  higher total reward. That's all "learning" means here.
- **Our agent**: every junction is an agent. Action = which phase to show for the
  next 5 s. Reward = minus the number of stopped cars at that junction, including
  cars it keeps waiting outside the city (fewer waiting cars = higher reward).
- **Observation** (what it sees), 24 numbers between 0 and 1:
  - cars on each of its 4 incoming roads
  - stopped cars on each of its 4 incoming roads
  - cars in the last 60 m before each stop line (about to arrive)
  - lanes whose front car is stuck because the road it wants to enter is full
    (a green there would be wasted)
  - **how full each of its 4 outgoing roads are** ← this is how it "sees" the
    neighbouring junctions' congestion
  - which phase is on, how long it has been on, whether it is switching
- **Policy** = the neural network (two layers of 64 neurons) that maps those 24
  numbers to "show NS" or "show EW".
- **PPO** (Proximal Policy Optimisation) = the standard, stable algorithm we used
  to train the network. Key idea: improve the policy a little at a time, never
  jumping too far from what already works ("clipping").

### The key design choice: one shared policy
Every junction uses **the same network** ("parameter sharing"). Because the 24
inputs mean the same thing at every junction, the same network works on a 2×2
grid, a 3×3 grid, or a 6×6 grid. We train on 2×2 and 3×3 together and then test,
without any retraining, on 4×4 and 6×6. That is **zero-shot transfer**.

The old project used one big network that saw the whole grid at once. It could
not be applied to a different grid size at all (its input size was fixed to the
grid) and needed 4^N joint choices.

### The safety shield
A **shield** is a rule that sits between the learned policy and the real world
and blocks unsafe actions. Ours is one sentence: *if the green the agent chose
cannot move a single car, but the other direction has cars that can move,
switch.* Without it, on big unseen grids, the agent sometimes kept giving green
to lanes stuck behind full roads; queues spread backwards until thousands of cars
were stuck outside the city (**congestion collapse**). The shield needs no
retraining and we give it to max-pressure too, so the comparison stays fair.

### The ablations
An **ablation** removes one ingredient to see whether it mattered.
1. **No shield**: same trained agents, shield switched off.
2. **No neighbour view**: a second set of agents trained without the "how full
   are my outgoing roads" numbers (set to zero).

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

Full tables: `results/results.md`. What to remember:

| Controller | Mean wait per car | On average, how far from the best in each situation | Worst situation |
|---|---|---|---|
| **GridFlow-RL (ours)** | **38.8 s** | 3 % | 1.36× |
| Vehicle-actuated | 47.5 s | 31 % | 1.60× |
| Max-pressure + shield | 48.8 s | 33 % | 2.34× |
| Max-pressure | 64.5 s | 58 % | 5.19× |
| Fixed-time | 71.5 s | 119 % | 10.3× |

**How to say it in one breath:** "Averaged over four grid sizes, six scenarios
and ten test seeds, our agent cuts waiting by 18 % against the best classical
controller and by 46 % against a fixed timer. More importantly, every classical
method has a situation where it falls apart — fixed-time on a busy main road,
max-pressure in a surge — while ours stays within 3 % of the best on average."

**Where it wins**: incidents (about 25–30 % less waiting than any other method),
busy main roads, light traffic, surges on the grid sizes it trained on, and on
unseen 4×4 grids in every scenario.

**Where it loses (say this before they ask)**:
- Perfectly even traffic: a tuned 15-s fixed timer is 1–3 s better. With
  identical demand everywhere, a fast fixed rhythm is close to optimal and there
  is little to adapt to.
- The largest, most saturated main-road test (6×6 arterial): max-pressure is
  better (78 s vs 106 s). The main road there is 6 junctions long; the agent only
  ever trained on corridors of 2–3. Zero-shot transfer has limits.
- 6×6 surge: actuated is slightly better (85 s vs 94 s).

**Ablations**: removing the shield costs **+25 %** waiting on unseen 4×4 and 6×6
grids (it prevents collapse exactly where generalisation is hardest). Removing
the neighbour view costs only **~1 %** — an honest negative result: the agent's
own "blocked lane" inputs already tell it most of what the neighbour numbers
would. Saying this clearly is a strength, not a weakness.

**Learning**: the agent learns in about 300,000 samples (first few minutes)
and stays stable to 5 million; all 5 training seeds land within 36–43 s.

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

**Why does the agent decide every 5 seconds but max-pressure every 20?**
Each was tuned. Max-pressure has no notion of switching cost, so deciding every 5 s
made it flip so often that the network gridlocked; 20 s was its best setting. The
RL agent *learns* the switching cost (3 s lost each time) from its reward, so it can
safely be asked more often and use the extra reaction speed.

**What is gamma (0.97)?**
How much the agent cares about future reward. With a decision every 5 s, 0.97 means
it effectively looks about 3 minutes ahead (1 / (1 − 0.97) ≈ 33 decisions × 5 s).

**Is this really multi-agent?**
Yes — each junction acts on its own local view and its own reward. They share the
same network weights (parameter sharing), which is a standard, well-studied setup.

**Why did max-pressure gridlock in the surge?**
Max-pressure counts how full the downstream *road* is on average, but a car needs
one specific *lane*. In a sudden flood it can keep giving green to cars that cannot
physically move, until a closed loop of full lanes forms and nothing can move.

**Isn't the shield cheating — a hand-written rule?**
No: it is a standard safe-RL technique (shielding), it is tiny and fully
explainable, we give the same shield to the best baseline (max-pressure), and we
report results with and without it. Real deployments always wrap learned
controllers in safety rules.

**Why did the first version lose?**
It decided only every 10 s and could not see cars about to arrive or lanes stuck
behind full roads — exactly the information the strong baselines use. The commit
history shows this diagnosis and the fix.

**Why didn't the neighbour view help much?**
The agent already sees, locally, when its own cars are stuck because the road
ahead is full. That is most of what the neighbour numbers add. We kept the
ablation in the report because negative results are results.

**What would you do next?**
Port to SUMO with real map data, add pedestrian phases and protected-permissive
lefts, train with real detector data, and add explicit communication between
neighbouring agents (e.g. graph neural networks).
