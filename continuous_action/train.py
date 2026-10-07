"""
Training loop for the continuous-action VLA.

Same shape as grid_world/train.py, with the one change that actually
matters for this comparison: F.mse_loss instead of F.cross_entropy. There
is no "argmax == label" accuracy metric to print anymore either -- you
can't ask "did it pick the right class" when there are no classes, only a
continuous target. MSE (mean squared error between the predicted and
target vectors) is the natural stand-in: lower is better, 0 is perfect.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import generate_dataset
from .tokenizer import MAX_TEXT_LEN, VOCAB_SIZE
from .model import VLAContinuous

CHECKPOINT_PATH = "continuous_action_checkpoint.pt"


def build_model():
    return VLAContinuous(vocab_size=VOCAB_SIZE, n_text_tokens=MAX_TEXT_LEN)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_images, train_texts, train_actions = generate_dataset(n_episodes=3000, seed=0)
    val_images, val_texts, val_actions = generate_dataset(n_episodes=300, seed=1)
    val_images, val_texts, val_actions = val_images.to(device), val_texts.to(device), val_actions.to(device)

    loader = DataLoader(TensorDataset(train_images, train_texts, train_actions), batch_size=64, shuffle=True)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(20):
        model.train()
        total_loss, total = 0.0, 0
        for img, txt, act in loader:
            img, txt, act = img.to(device), txt.to(device), act.to(device)
            pred = model(img, txt)             # (B, 2) predicted (dx, dy)
            loss = F.mse_loss(pred, act)        # regression loss, not cross-entropy

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * img.size(0)
            total += img.size(0)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_images, val_texts)
            val_mse = F.mse_loss(val_pred, val_actions).item()

        print(f"epoch {epoch:02d} | train_mse {total_loss / total:.4f} | val_mse {val_mse:.4f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
