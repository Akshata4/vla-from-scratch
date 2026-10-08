"""
Training loop for the frozen-DINOv2 VLA.

Identical mechanics to grid_world/train.py -- cross-entropy over action
tokens, same optimizer, same epoch count. The only real difference is WHAT
the dataset contains: pre-extracted vision features instead of raw images.
The model itself never touches DINOv2 during training at all -- that
happened once, in data.py -- so this loop is exactly as fast as the
original grid_world recipe.
"""

import torch
import torch.nn.functional as F
from grid_world.tokenizer import ACTION_IDS, BOA_ID, MAX_TEXT_LEN, VOCAB_SIZE
from torch.utils.data import DataLoader, TensorDataset

from .data import generate_dataset
from .model import VLAPretrainedVision

CHECKPOINT_PATH = "pretrained_vision_checkpoint.pt"


def build_model():
    return VLAPretrainedVision(
        vocab_size=VOCAB_SIZE,
        n_text_tokens=MAX_TEXT_LEN,
        boa_id=BOA_ID,
        action_ids=ACTION_IDS,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("generating training set...")
    train_vision, train_texts, train_actions = generate_dataset(n_episodes=3000, seed=0, device=device)
    print("generating validation set...")
    val_vision, val_texts, val_actions = generate_dataset(n_episodes=300, seed=1, device=device)
    val_vision, val_texts, val_actions = val_vision.to(device), val_texts.to(device), val_actions.to(device)

    loader = DataLoader(TensorDataset(train_vision, train_texts, train_actions), batch_size=64, shuffle=True)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(20):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for vis, txt, act in loader:
            vis, txt, act = vis.to(device), txt.to(device), act.to(device)
            logits = model(vis, txt)
            loss = F.cross_entropy(logits, act)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * vis.size(0)
            total_correct += (logits.argmax(-1) == act).sum().item()
            total += vis.size(0)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_vision, val_texts)
            val_acc = (val_logits.argmax(-1) == val_actions).float().mean().item()

        print(f"epoch {epoch:02d} | loss {total_loss / total:.4f} "
              f"| train_acc {total_correct / total:.3f} | val_acc {val_acc:.3f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
