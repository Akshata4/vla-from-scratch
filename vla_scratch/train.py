import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import generate_dataset
from .model import VLA
from .tokenizer import ACTION_IDS, BOA_ID, MAX_TEXT_LEN, VOCAB_SIZE

CHECKPOINT_PATH = "vla_scratch_checkpoint.pt"


def build_model():
    return VLA(
        vocab_size=VOCAB_SIZE,
        n_text_tokens=MAX_TEXT_LEN,
        boa_id=BOA_ID,
        action_ids=ACTION_IDS,
    )


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
        total_loss, total_correct, total = 0.0, 0, 0
        for img, txt, act in loader:
            img, txt, act = img.to(device), txt.to(device), act.to(device)
            logits = model(img, txt)
            loss = F.cross_entropy(logits, act)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * img.size(0)
            total_correct += (logits.argmax(-1) == act).sum().item()
            total += img.size(0)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_images, val_texts)
            val_acc = (val_logits.argmax(-1) == val_actions).float().mean().item()

        print(f"epoch {epoch:02d} | loss {total_loss / total:.4f} "
              f"| train_acc {total_correct / total:.3f} | val_acc {val_acc:.3f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
