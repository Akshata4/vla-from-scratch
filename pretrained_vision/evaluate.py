"""
Closed-loop evaluation for the frozen-DINOv2 VLA.

Unlike training (which runs entirely on cached features), closed-loop
rollout sees a brand new image every step that was never cached, so here
we DO call the frozen DINOv2 backbone live, once per step. Still cheap --
it's a single frozen forward pass with no gradients, same cost class as
any other inference-only model call.

The headline number is directly comparable to grid_world/evaluate.py's and
continuous_action/evaluate.py's: same task, same metric, only the source
of the visual tokens differs.
"""

import numpy as np
import torch
from grid_world.env import GridWorld
from grid_world.tokenizer import ID_TO_ACTION_NAME, encode_instruction

from .train import CHECKPOINT_PATH, build_model
from .vision_backbone import extract_features, resize

MAX_STEPS = 16


def load_model(checkpoint=CHECKPOINT_PATH, device="cpu"):
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def rollout(model, env, device="cpu"):
    img, instr = env.reset()
    text_ids = torch.tensor([encode_instruction(instr)], device=device)

    for _ in range(MAX_STEPS):
        vision_tokens = extract_features(resize(img)[None], device=device)  # (1, 16, 384), live
        action_id = model.act(vision_tokens, text_ids).item()
        img, success = env.step(ID_TO_ACTION_NAME[action_id])
        if success:
            return True, instr
    return False, instr


def evaluate(model, n_episodes=200, seed=42, device="cpu"):
    rng = np.random.default_rng(seed)
    successes = sum(rollout(model, GridWorld(rng), device=device)[0] for _ in range(n_episodes))
    return successes / n_episodes


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)
    acc = evaluate(model, device=device)
    print(f"closed-loop success rate over 200 episodes: {acc:.2%}")
