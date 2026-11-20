"""
Load a trained policy for inference only.

Stable-Baselines3 2.7.1's PPO.load streams tensors straight out of the zip, which
torch >= 2.11 cannot read ("PytorchStreamReader failed"). We only need the policy
network to act, so we rebuild it and load its weights from memory — no optimiser,
no environment needed.
"""

from __future__ import annotations

import io
import json
import os
import zipfile

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from gridflow.controllers import OBS_DIM


def load_policy(path: str, net_arch=(64, 64)) -> ActorCriticPolicy:
    policy = ActorCriticPolicy(
        spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32), spaces.Discrete(2),
        lr_schedule=lambda _: 0.0, net_arch=list(net_arch),
    )
    with zipfile.ZipFile(path) as z:
        state = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    policy.load_state_dict(state)
    policy.set_training_mode(False)
    return policy


def load_config(path: str) -> dict:
    cfg = os.path.join(os.path.dirname(path), "config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            return json.load(f)
    return {"variant": "neighbour", "dt": 10}


def load_controller(path: str, name: str | None = None):
    """A PolicyController set up exactly as the model was trained (decision interval, inputs)."""
    from gridflow.controllers import PolicyController
    cfg = load_config(path)
    variant = cfg.get("variant", "neighbour")
    return PolicyController(load_policy(path), name=name or f"rl_{variant}",
                            downstream=(variant == "neighbour"), interval=int(cfg.get("dt", 10)))
