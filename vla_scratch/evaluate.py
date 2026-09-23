import numpy as np
import torch
from PIL import Image

from .data import MAX_STEPS
from .env import GridWorld
from .tokenizer import ID_TO_ACTION_NAME, encode_instruction
from .train import CHECKPOINT_PATH, build_model


def load_model(checkpoint=CHECKPOINT_PATH, device="cpu"):
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def rollout(model, env, device="cpu", record=False):
    img, instr = env.reset()
    text_ids = torch.tensor([encode_instruction(instr)], device=device)
    frames = [img] if record else None

    for _ in range(MAX_STEPS):
        img_t = torch.tensor(img, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
        action_id = model.act(img_t, text_ids).item()
        img, success = env.step(ID_TO_ACTION_NAME[action_id])
        if record:
            frames.append(img)
        if success:
            return True, instr, frames
    return False, instr, frames


def evaluate(model, n_episodes=200, seed=42, device="cpu"):
    rng = np.random.default_rng(seed)
    successes = sum(rollout(model, GridWorld(rng), device=device)[0] for _ in range(n_episodes))
    return successes / n_episodes


def save_gif(frames, path, scale=8):
    imgs = [Image.fromarray(f).resize((f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=300, loop=0)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)

    acc = evaluate(model, device=device)
    print(f"closed-loop success rate over 200 episodes: {acc:.2%}")

    rng = np.random.default_rng(123)
    env = GridWorld(rng)
    ok, instr, frames = rollout(model, env, device=device, record=True)
    save_gif(frames, "rollout_example.gif")
    print(f"example rollout | instruction: '{instr}' | success: {ok} | saved rollout_example.gif")
