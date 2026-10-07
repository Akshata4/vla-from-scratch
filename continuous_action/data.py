"""
Behavior-cloning dataset generation for the continuous-action variant.

Identical recipe to grid_world/data.py -- roll the same scripted expert
forward in the SAME GridWorld environment (imported, not reimplemented) --
with exactly one change: the recorded label is now a continuous (dx, dy)
vector (tokenizer.encode_action) instead of a discrete action-token id.
"""

import numpy as np
import torch

from grid_world.env import GridWorld

from .tokenizer import encode_action, encode_instruction

MAX_STEPS = 16  # same cap as grid_world/data.py -- the task hasn't changed, only the label has


def generate_dataset(n_episodes=3000, seed=0):
    rng = np.random.default_rng(seed)
    images, texts, actions = [], [], []
    for _ in range(n_episodes):
        env = GridWorld(rng)
        img, instr = env.reset()
        text_ids = encode_instruction(instr)
        for _ in range(MAX_STEPS):
            action = env.expert_action()
            if action is None:
                break
            images.append(img)
            texts.append(text_ids)
            actions.append(encode_action(action))  # <-- (dx, dy) tuple, not a class id
            img, success = env.step(action)
            if success:
                break

    images = torch.tensor(np.stack(images), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    texts = torch.tensor(texts, dtype=torch.long)
    actions = torch.tensor(actions, dtype=torch.float32)  # float, not long -- this is a regression target now
    return images, texts, actions
