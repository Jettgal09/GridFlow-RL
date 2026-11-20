"""
gridflow/marl.py — multi-agent training with ONE shared policy.

IDEA
  Every junction is an agent. All agents use the same small neural network
  (parameter sharing). Each agent sees only its own 24 features (its approaches
  plus how full the roads it feeds are) and gets its own reward (minus its own
  queue). Because the input and output have the same shape at every junction,
  the trained network can be dropped onto a grid of any size — including sizes
  it never saw in training.

HOW IT PLUGS INTO STABLE-BASELINES3
  SB3's PPO trains on a "vectorised environment": a batch of independent
  single-agent environments stepped together. Here each *junction* is one slot
  in that batch. Several simulations (some 2x2, some 3x3) run side by side; a
  2x2 contributes 4 slots and a 3x3 contributes 9. PPO never knows the slots
  share a road network — it just sees a batch of (observation, action, reward).
  This is the standard "independent learners with parameter sharing" setup.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from gridflow import scenarios as S
from gridflow.controllers import OBS_DIM, all_features, node_pressure, node_queue
from gridflow.sim import Network

REWARD_SCALE = 50.0  # queue of 50 stopped cars at a junction -> reward -1 per decision
PRESSURE_SCALE = 1.0


class SharedPolicyVecEnv(VecEnv):
    def __init__(self, grid_sizes=((3, 3),) * 4 + ((2, 2),) * 4, episode_len=1800,
                 decision_interval=10, downstream=True, seed=0, scenario="train", reward="queue",
                 share=0.0):
        self.sims = [Network(r, c, S.make(scenario)) for r, c in grid_sizes]
        self.offsets = np.cumsum([0] + [s.n_nodes for s in self.sims])
        self.episode_len = episode_len
        self.dt = decision_interval
        self.downstream = downstream
        self.reward = reward
        self.share = share
        self.nbrs = [[sorted({n for n in sim.upstream[i] + sim.downstream[i] if n is not None})
                      for i in range(sim.n_nodes)] for sim in self.sims]
        self._seed_counter = seed * 1_000_003
        self._actions = None
        self.ep_returns = np.zeros(self.offsets[-1])
        self.finished_episodes = []  # (sim index, metrics) for logging
        obs_space = spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32)
        super().__init__(int(self.offsets[-1]), obs_space, spaces.Discrete(2))

    # -------------------------------------------------------------- helpers
    def _next_seed(self):
        self._seed_counter += 1
        return self._seed_counter

    def _obs(self, k):
        return all_features(self.sims[k], self.downstream)

    # ------------------------------------------------------------ VecEnv API
    def reset(self):
        out = []
        for k, sim in enumerate(self.sims):
            sim.reset(self._next_seed())
            out.append(self._obs(k))
        self.ep_returns[:] = 0
        return np.concatenate(out)

    def step_async(self, actions):
        self._actions = np.asarray(actions).reshape(-1)

    def step_wait(self):
        obs_l, rew_l, done_l, info_l = [], [], [], []
        for k, sim in enumerate(self.sims):
            a = self._actions[self.offsets[k]:self.offsets[k + 1]]
            for i in range(sim.n_nodes):
                sim.set_phase(i, int(a[i]))
            q = np.zeros(sim.n_nodes)
            for _ in range(self.dt):
                sim.step()
                if self.reward == "queue":
                    for i in range(sim.n_nodes):
                        q[i] += node_queue(sim, i) / REWARD_SCALE
                else:  # "pressure" (PressLight)
                    for i in range(sim.n_nodes):
                        q[i] += node_pressure(sim, i) / PRESSURE_SCALE
            if self.share > 0:
                nb = np.array([q[js].mean() if js else 0.0 for js in self.nbrs[k]])
                q = q + self.share * nb
            r = -(q / self.dt)
            done = sim.t >= self.episode_len
            obs = self._obs(k)
            infos = [{} for _ in range(sim.n_nodes)]
            if done:
                self.finished_episodes.append((k, sim.metrics()))
                for i in range(sim.n_nodes):
                    infos[i]["terminal_observation"] = obs[i]
                    infos[i]["TimeLimit.truncated"] = True
                sim.reset(self._next_seed())
                obs = self._obs(k)
            obs_l.append(obs); rew_l.append(r)
            done_l.append(np.full(sim.n_nodes, done)); info_l.extend(infos)
        return (np.concatenate(obs_l), np.concatenate(rew_l).astype(np.float32),
                np.concatenate(done_l), info_l)

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None) for _ in self._idx(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None for _ in self._idx(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._idx(indices)]

    def seed(self, seed=None):
        if seed is not None:
            self._seed_counter = seed * 1_000_003
        return [seed] * self.num_envs

    def _idx(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices
