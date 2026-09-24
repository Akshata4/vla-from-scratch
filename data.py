import numpy as np
import torch

from env import GridWorld
from tokenizer import encode_instruction, encode_action

MAX_STEPS = 16  # >= worst-case Manhattan distance on an 8x8 grid (7 + 7)


def generate_dataset(n_episodes=3000, seed=0):
    """Roll out the scripted expert and record (image, instruction, action)
    at every step — this is the behavior-cloning dataset."""
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
            actions.append(encode_action(action))
            img, success = env.step(action)
            if success:
                break

    images = torch.tensor(np.stack(images), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    texts = torch.tensor(texts, dtype=torch.long)
    actions = torch.tensor(actions, dtype=torch.long)
    return images, texts, actions
