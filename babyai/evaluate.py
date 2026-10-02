import numpy as np
import torch

from .env import MAX_STEPS, Episode, make_env
from .tokenizer import ID_TO_ACTION_NAME, encode_instruction
from .train import CHECKPOINT_PATH, build_model

GYM_ACTION = {"left": 0, "right": 1, "forward": 2, "pickup": 3, "drop": 4, "toggle": 5, "done": 6}


def load_model(checkpoint=CHECKPOINT_PATH, device="cpu"):
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def rollout(model, device="cpu"):
    episode = Episode(make_env())
    img, mission = episode.reset()
    text_ids = torch.tensor([encode_instruction(mission)], device=device)

    for _ in range(MAX_STEPS):
        img_t = torch.tensor(img, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
        action_id = model.act(img_t, text_ids).item()
        gym_action = GYM_ACTION[ID_TO_ACTION_NAME[action_id]]
        img, success, done = episode.step(gym_action)
        if done:
            return success, mission
    return False, mission


def evaluate(model, n_episodes=200, device="cpu"):
    successes = sum(rollout(model, device=device)[0] for _ in range(n_episodes))
    return successes / n_episodes


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)
    acc = evaluate(model, device=device)
    print(f"closed-loop success rate over 200 episodes: {acc:.2%}")
