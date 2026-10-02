import gymnasium as gym
import minigrid  # noqa: F401 (registers the BabyAI-* environment ids)
import numpy as np
from minigrid.utils.baby_ai_bot import BabyAIBot
from minigrid.wrappers import RGBImgPartialObsWrapper
from PIL import Image

ENV_ID = "BabyAI-GoToLocal-v0"
RENDER_SIZE = 56   # what the egocentric-partial wrapper gives us
IMG_SIZE = 32      # resized to this so model.py's VisionEncoder needs no changes
MAX_STEPS = 64     # matches the env's own episode time limit


def make_env():
    # Agent only sees what's directly in front of it, not the whole room.
    # BabyAIBot still plans from env.unwrapped's full internal grid state --
    # it's the label source (like a privileged-information demonstrator),
    # not something the learned model gets to see.
    return RGBImgPartialObsWrapper(gym.make(ENV_ID))


def resize(image):
    return np.asarray(Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))


class Episode:
    """Thin wrapper pairing a BabyAI env with its matching expert bot."""

    def __init__(self, env):
        self.env = env

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        self.bot = BabyAIBot(self.env.unwrapped)
        return resize(obs["image"]), obs["mission"]

    def expert_action(self):
        return int(self.bot.replan())

    def step(self, gym_action):
        obs, reward, terminated, truncated, _ = self.env.step(gym_action)
        success = bool(reward > 0)
        return resize(obs["image"]), success, bool(terminated or truncated)
