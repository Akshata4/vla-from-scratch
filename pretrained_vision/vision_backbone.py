"""
Frozen DINOv2-Small as the vision encoder, replacing model.py's
from-scratch VisionEncoder entirely.

DINOv2 was pretrained via self-supervised learning on ~142M real
photographs (LVD-142M) -- a completely different regime from our 174K-param
CNN, which only ever learned from our own 3000 toy episodes. This file's
only job is turning a raw image into the same "16 visual tokens" shape our
transformer already expects, so everything downstream (projector,
transformer, BOA, action head) is unchanged from the rest of this repo.

Because the backbone is FROZEN (requires_grad=False, no gradients ever
flow into it), we only ever need to run it in inference mode -- see
data.py, which runs every training image through it exactly ONCE up front
and caches the result, instead of recomputing DINOv2 features every epoch.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

MODEL_NAME = "facebook/dinov2-small"
IMG_SIZE = 56       # 56 / patch_size(14) = 4x4 = 16 patches -- matches our
                     # original VisionEncoder's 16 vision tokens exactly, so
                     # the transformer's sequence length doesn't change.
FEATURE_DIM = 384   # DINOv2-small's hidden size (vs. our own CNN's 32)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_backbone = None  # loaded lazily, once, the first time it's needed


def get_backbone(device="cpu"):
    global _backbone
    if _backbone is None:
        print(f"  loading frozen {MODEL_NAME} (downloads + caches on first run)...")
        _backbone = AutoModel.from_pretrained(MODEL_NAME).to(device)
        _backbone.eval()
        for p in _backbone.parameters():
            p.requires_grad = False  # frozen: this model is never trained
    return _backbone


def resize(img):
    """Our GridWorld renders at 32x32; DINOv2 needs a size divisible by its
    14x14 patch size, so we upscale to 56x56 (still just the same flat
    colored squares, bigger -- see the BabyAI section of the README for why
    we don't expect this alone to change what the model can see)."""
    return np.asarray(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))


@torch.no_grad()
def extract_features(images_uint8, device="cpu"):
    """images_uint8: (N, 56, 56, 3) uint8. Returns (N, 16, 384) patch-token
    features with the CLS token dropped -- we want DINOv2's spatial patch
    grid (where things are), not a single pooled summary, since the task
    needs relative position information to pick the right direction."""
    backbone = get_backbone(device)
    x = torch.as_tensor(np.array(images_uint8), dtype=torch.float32, device=device) / 255.0
    x = x.permute(0, 3, 1, 2)                                     # (N, 3, 56, 56)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)  # DINOv2's expected normalization
    out = backbone(x, interpolate_pos_encoding=True)              # supports non-native resolutions
    return out.last_hidden_state[:, 1:, :]                        # drop CLS -> (N, 16, 384)
