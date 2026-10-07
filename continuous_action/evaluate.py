"""
Closed-loop evaluation for the continuous-action VLA.

Same rollout structure as grid_world/evaluate.py: observe, act, step,
repeat. The only new step is tokenizer.vector_to_action() -- the model
predicts raw (dx, dy) numbers, and since our GridWorld environment only
understands 4 named moves, we snap the prediction to the nearest one
before calling env.step(). (A real continuous-control robot wouldn't need
this snapping step -- its actuators would take the raw numbers directly.)

The headline number this prints -- closed-loop success rate over 200
episodes -- is directly comparable to grid_world/evaluate.py's number,
since it's the exact same task, same metric, only the model's internal
action representation differs.
"""

import numpy as np
import torch

from grid_world.env import GridWorld

from .tokenizer import encode_instruction, vector_to_action
from .train import CHECKPOINT_PATH, build_model

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
        img_t = torch.tensor(img, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            dx, dy = model(img_t, text_ids)[0].tolist()   # raw continuous prediction
        action = vector_to_action(dx, dy)                  # snap to one of the 4 grid moves
        img, success = env.step(action)
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
