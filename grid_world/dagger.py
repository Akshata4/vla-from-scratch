"""
DAgger (Dataset Aggregation) for the grid_world VLA.

THE PROBLEM THIS FIXES
-----------------------
Plain behavior cloning (train.py) only ever trains on states the SCRIPTED
EXPERT visited while walking its own optimal path (see env.py's
expert_action()). At test time the MODEL is the one in control instead, and
the first time it makes even one small mistake, it can land in a state the
expert's demonstrations never covered. There is no training example for
"what to do from here" -- so the model is guessing blind, often makes
another mistake, and drifts further off course. That's exactly the gap we
measured: ~72% single-step accuracy but only ~54% closed-loop success.

THE FIX: DAGGER (DATASET AGGREGATION)
---------------------------------------
Loop:
  1. train on whatever data we have so far (round 0 = the plain BC dataset)
  2. let the CURRENT model actually drive the agent around the grid
  3. at every state the model visits, ask the scripted expert what the
     correct action would have been FROM THAT STATE, and record that as a
     new labeled example (image, instruction, expert's answer)
  4. add those new examples to the dataset -- aggregate, never discard
  5. retrain (from scratch) on the full aggregated dataset
  6. repeat

Each round, the training set grows to cover more of the "recovery" states
the model actually wanders into -- not just the expert's own perfect path.
Run it with:  uv run python -m grid_world.dagger
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import generate_dataset
from .env import GridWorld
from .evaluate import MAX_STEPS, evaluate as closed_loop_success_rate
from .tokenizer import ID_TO_ACTION_NAME, encode_action, encode_instruction
from .train import build_model

DAGGER_ROUNDS = 4          # how many times we aggregate-and-retrain
EPISODES_PER_ROUND = 800   # new rollout episodes to collect labels from, each round
TRAIN_EPOCHS_PER_ROUND = 15
EVAL_SEED = 1000           # fixed, held-out seed for measuring progress -- never used for training data
CHECKPOINT_PATH = "grid_world_dagger_checkpoint.pt"


def collect_policy_rollout_data(model, n_episodes, device, seed):
    """
    Steps 2 + 3 of the DAgger loop.

    This looks almost identical to evaluate.py's rollout(), with one
    deliberate difference: instead of just checking whether the model
    succeeds, at every step we ALSO ask the scripted expert what it would
    have done from the model's current (possibly already-wrong) state, and
    we keep that as a label.

    The environment is then advanced using the MODEL's chosen action, not
    the expert's. That's the whole trick: we want the agent to actually
    wander into the states the model's own judgement leads it to, so we can
    teach it how to recover from exactly those states -- not just re-walk
    the expert's original path, which the model has already seen.
    """
    rng = np.random.default_rng(seed)
    images, texts, actions = [], [], []

    for _ in range(n_episodes):
        env = GridWorld(rng)
        img, instr = env.reset()
        text_ids = encode_instruction(instr)

        for _ in range(MAX_STEPS):
            # --- 3. ask the expert for the correct label AT THIS STATE ---
            # env.expert_action() is a live geometric computation from the
            # agent's *current* position -- it works from any state, not
            # just ones on the expert's own path. That's what makes DAgger
            # possible: we can query "what's correct here" on demand.
            expert_label = env.expert_action()
            if expert_label is None:  # agent is already standing on the target
                break

            images.append(img)
            texts.append(text_ids)
            actions.append(encode_action(expert_label))

            # --- 2. advance the environment using the MODEL's own choice ---
            # (not the expert's) -- this is what lets the agent actually
            # drift into "mistake" states instead of only re-tracing the
            # expert's perfect path.
            img_t = torch.tensor(img, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
            text_t = torch.tensor([text_ids], device=device)
            model_action_id = model.act(img_t, text_t).item()
            model_action_name = ID_TO_ACTION_NAME[model_action_id]

            img, success = env.step(model_action_name)
            if success:
                break

    images = torch.tensor(np.stack(images), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    texts = torch.tensor(texts, dtype=torch.long)
    actions = torch.tensor(actions, dtype=torch.long)
    return images, texts, actions


def train_on(model, images, texts, actions, device, epochs):
    """One ordinary supervised training pass over a fixed dataset -- the
    exact same mechanics as train.py's main(), just reusable so DAgger can
    call it once per round on a dataset that keeps growing underneath it."""
    loader = DataLoader(TensorDataset(images, texts, actions), batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(epochs):
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
        print(f"    epoch {epoch:02d} | loss {total_loss / total:.4f} | train_acc {total_correct / total:.3f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ================= ROUND 0: plain behavior cloning =================
    # Identical to train.py -- this is the "before DAgger" baseline. All
    # of this data comes from the EXPERT walking its own optimal path.
    print("\n=== round 0: plain behavior cloning (train.py's ordinary recipe) ===")
    images, texts, actions = generate_dataset(n_episodes=3000, seed=0)
    model = build_model().to(device)
    train_on(model, images, texts, actions, device, epochs=TRAIN_EPOCHS_PER_ROUND)

    success_rate = closed_loop_success_rate(model, n_episodes=200, seed=EVAL_SEED, device=device)
    print(f"round 0 closed-loop success rate: {success_rate:.2%}")

    # ================= ROUNDS 1..N: the DAgger loop =================
    for round_num in range(1, DAGGER_ROUNDS + 1):
        print(f"\n=== round {round_num}: let the model drive, relabel with the expert, aggregate, retrain ===")

        # 1. Let the CURRENT model drive; label the states it actually visits
        #    with what the expert would have done there.
        new_images, new_texts, new_actions = collect_policy_rollout_data(
            model, n_episodes=EPISODES_PER_ROUND, device=device, seed=EVAL_SEED + round_num)
        print(f"  collected {len(new_images)} new (state visited by the model, expert's label) pairs")

        # 2. Aggregate: append to everything collected in every previous
        #    round. Old data is never thrown away -- "Dataset Aggregation"
        #    is literally the name of the algorithm.
        images = torch.cat([images, new_images], dim=0)
        texts = torch.cat([texts, new_texts], dim=0)
        actions = torch.cat([actions, new_actions], dim=0)
        print(f"  training set is now {len(images)} examples total")

        # 3. Retrain from scratch on the full aggregated dataset. (A cheaper
        #    variant some implementations use is to keep fine-tuning the
        #    same weights instead of reinitializing -- we re-init here to
        #    match the textbook algorithm exactly, and our model is small
        #    enough that retraining from scratch each round is still fast.)
        model = build_model().to(device)
        train_on(model, images, texts, actions, device, epochs=TRAIN_EPOCHS_PER_ROUND)

        # 4. Measure progress on the SAME held-out seed every round, so the
        #    numbers across rounds are directly comparable.
        success_rate = closed_loop_success_rate(model, n_episodes=200, seed=EVAL_SEED, device=device)
        print(f"  round {round_num} closed-loop success rate: {success_rate:.2%}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nsaved final DAgger-trained checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
