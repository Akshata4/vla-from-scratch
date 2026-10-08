"""
Same behavior-cloning recipe as grid_world/data.py -- same GridWorld
environment, same scripted expert -- but this file's output is different:
instead of raw images, it returns PRE-EXTRACTED frozen DINOv2 features.

Because the backbone never changes (frozen), there's no reason to run it
again on every training epoch. We run every image through it exactly once,
right here, and train.py trains on these cached vectors from then on --
that's the whole practical payoff of freezing: training itself stays
exactly as fast as it's always been in this repo, and the one-time cost is
only this feature-extraction pass.
"""

import numpy as np
import torch

from grid_world.env import GridWorld
from grid_world.tokenizer import encode_action, encode_instruction

from .vision_backbone import extract_features, resize

MAX_STEPS = 16  # same cap as grid_world/data.py -- the task hasn't changed
EXTRACT_BATCH_SIZE = 256  # how many images to run through DINOv2 at once


def generate_dataset(n_episodes=3000, seed=0, device="cpu"):
    # ---- 1. roll out the scripted expert, exactly like grid_world/data.py ----
    rng = np.random.default_rng(seed)
    raw_images, texts, actions = [], [], []
    for _ in range(n_episodes):
        env = GridWorld(rng)
        img, instr = env.reset()
        text_ids = encode_instruction(instr)
        for _ in range(MAX_STEPS):
            action = env.expert_action()
            if action is None:
                break
            raw_images.append(resize(img))  # 32x32 -> 56x56 for DINOv2's patch size
            texts.append(text_ids)
            actions.append(encode_action(action))
            img, success = env.step(action)
            if success:
                break

    # ---- 2. the one new step: extract-and-cache DINOv2 features up front ----
    # (batched so we're not holding every image's activations in memory at once)
    print(f"  extracting frozen DINOv2 features for {len(raw_images)} images...")
    vision_tokens = []
    for i in range(0, len(raw_images), EXTRACT_BATCH_SIZE):
        batch = np.stack(raw_images[i:i + EXTRACT_BATCH_SIZE])
        vision_tokens.append(extract_features(batch, device=device).cpu())
    vision_tokens = torch.cat(vision_tokens, dim=0)  # (N, 16, 384) -- reused every epoch, never recomputed

    texts = torch.tensor(texts, dtype=torch.long)
    actions = torch.tensor(actions, dtype=torch.long)
    return vision_tokens, texts, actions
