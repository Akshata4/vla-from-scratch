import numpy as np
import torch

from .env import MAX_STEPS, Episode, make_env
from .tokenizer import encode_action, encode_instruction


def generate_dataset(n_episodes=3000, seed=0):
    """Same recipe as data.py's generate_dataset: roll the expert (here,
    BabyAIBot instead of our own hand-written rule) forward and record
    (image, instruction, action) at every step."""
    rng = np.random.default_rng(seed)
    images, texts, actions = [], [], []
    for _ in range(n_episodes):
        episode = Episode(make_env())
        img, mission = episode.reset(seed=int(rng.integers(0, 2**31 - 1)))
        text_ids = encode_instruction(mission)
        for _ in range(MAX_STEPS):
            gym_action = episode.expert_action()
            images.append(img)
            texts.append(text_ids)
            actions.append(encode_action(gym_action))
            img, success, done = episode.step(gym_action)
            if done:
                break

    images = torch.tensor(np.stack(images), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    texts = torch.tensor(texts, dtype=torch.long)
    actions = torch.tensor(actions, dtype=torch.long)
    return images, texts, actions
