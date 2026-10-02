"""Exports the trained checkpoint's weights to JSON so a hand-written JS
forward pass can run the real model client-side (see docs/vla-explainer.html)."""
import json

import torch

from grid_world.train import CHECKPOINT_PATH, build_model

model = build_model()
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.eval()

weights = {name: param.detach().tolist() for name, param in model.state_dict().items()
           if "causal_mask" not in name and "action_ids" not in name}

with open("docs/vla_weights.json", "w") as f:
    json.dump(weights, f)

total_params = sum(torch.tensor(v).numel() for v in weights.values())
print(f"exported {len(weights)} tensors, {total_params:,} params")
for name, v in weights.items():
    shape = list(torch.tensor(v).shape)
    print(f"  {name:30s} {shape}")
