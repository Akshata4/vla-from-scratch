"""
VLA demo, take 2: instead of feeding the model random noise, we pull one
real frame (real camera photo + real robot joint state + real language
instruction) from an actual recording of a SO-100 robot arm doing a
pick-and-place task, and see what SmolVLA predicts.

We also print the REAL action the human demonstrator took at that same
moment, so you can eyeball how close the model's guess is - no physics
simulator needed, the "ground truth" comes from the recording itself.

Dataset: lerobot/svla_so100_pickplace (official LeRobot pick-and-place demo)
Model:   lerobot/smolvla_base

Note: smolvla_base is a *generalist* checkpoint that was never fine-tuned
on this specific dataset/robot. Its output is unnormalized using its own
pretraining statistics, so the predicted numbers will be a different scale
than this robot's real joint-degree values below - that mismatch is
expected and is exactly why real deployments fine-tune the base model on
their own robot's data before trusting its outputs.
"""

from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

DATASET_ID = "lerobot/svla_so100_pickplace"
MODEL_ID = "lerobot/smolvla_base"
FRAME_INDEX = 0
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Downloading one episode of {DATASET_ID} (just episode 0, not the full dataset)...")
    dataset = LeRobotDataset(DATASET_ID, episodes=[0])
    frame = dict(dataset[FRAME_INDEX])

    # Save the real camera image so you can actually look at what the robot saw.
    image_path = Path(__file__).parent / "input_image.png"
    to_pil_image(frame["observation.images.top"]).save(image_path)
    print(f"Saved the real input image to {image_path}")

    # smolvla_base expects generic keys "camera1"/"camera2"; this dataset
    # names them "top"/"wrist" - just relabeling, same real pixels.
    frame["observation.images.camera1"] = frame.pop("observation.images.top")
    frame["observation.images.camera2"] = frame.pop("observation.images.wrist")

    print(f"Loading {MODEL_ID} on {device} ...")
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print(f"\nInstruction: {frame['task']!r}")

    batch = preprocess(frame)
    with torch.inference_mode():
        raw_action = policy.select_action(batch)
        predicted_action = postprocess(raw_action).squeeze(0)

    real_action = frame["action"]

    print(f"\n{'joint':<15}{'predicted':>12}{'real (recorded)':>18}")
    for name, pred, real in zip(JOINT_NAMES, predicted_action.tolist(), real_action.tolist()):
        print(f"{name:<15}{pred:>12.3f}{real:>18.3f}")


if __name__ == "__main__":
    main()
