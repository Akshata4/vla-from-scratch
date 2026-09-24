# vla-from-scratch

A Vision-Language-Action (VLA) model built from the ground up in plain PyTorch, to understand — end to end, no black boxes — exactly how "an LLM extended with vision and action" actually works.

This repo answers one question concretely: **what, structurally, is the difference between an LLM, a VLM, and a VLA?** The short answer is that all three share the same transformer core; a VLM splices vision tokens into the *input* sequence, and a VLA additionally splices action tokens into the *output* vocabulary and closes a control loop back out into the world. Everything here — the toy environment, the tokenizer, the model, the training loop — exists to make that one idea tangible.

## What is a VLA?

A **Vision-Language-Action model** is a robot control policy that takes an image observation and a language instruction as input and outputs a motor action instead of the next word. The cleanest mental model, if you already know how transformers work:

```
image  ──► vision encoder ──► visual tokens  ──┐
                                                 ├─► [visual tokens ; text tokens] ──► Transformer ──► hidden state ──► action head ──► action
instruction ──► tokenizer ──► text tokens  ─────┘
```

- **Vision encoder + projector** turn an image into a sequence of embeddings and map them into the LLM's token space (the same "connector" trick LLaVA uses for VLMs).
- **The transformer core is unmodified** — same self-attention, same MLP blocks, same causal mask you'd find in any GPT-style decoder.
- **The action head is the one genuinely new piece.** Real systems split into two families:
  - *Discretized action tokens* (RT-2, OpenVLA) — bin each action dimension and append the bins to the vocabulary as new tokens, then decode them exactly like next-token prediction. **This is the approach used in this repo.**
  - *Continuous action heads* (RT-1, ACT, Diffusion Policy, π0) — a separate MLP/diffusion head regresses a continuous action vector from the final hidden state.
- **Training is behavior cloning**: supervise the model to predict the action a human (or scripted expert) demonstrated, given the same image + instruction. No RL required.
- **Inference is closed-loop**: observe → act → the action changes the world → observe again → repeat.

A full visual walkthrough of this — three side-by-side pipelines for LLM/VLM/VLA, plus a zoom into exactly what happens inside one forward pass of the model in this repo — is in [`docs/vla-anatomy.html`](docs/vla-anatomy.html). It's a self-contained HTML file; download it and open it in a browser (GitHub doesn't render arbitrary HTML pages inline).

## What's actually in this repo

Two ways to see a VLA:

| | What it is | Files |
|---|---|---|
| **The from-scratch VLA** | A tiny transformer trained by behavior cloning to solve a toy "walk to the colored square" task | `env.py`, `tokenizer.py`, `data.py`, `model.py`, `train.py`, `evaluate.py` |
| **A real pretrained VLA** | Runs Hugging Face's [SmolVLA](https://huggingface.co/lerobot/smolvla_base) on one real frame from an actual robot-arm pick-and-place recording, and prints its predicted action next to what the human demonstrator actually did | `main.py` |

The point of pairing them: `main.py` shows what a production VLA's output looks like on real robot data; the rest of the repo shows you *why* it's structured that way, by building the same structure yourself at toy scale.

### The toy task

An 8×8 grid world with 2-4 colored squares and one agent. The instruction names a target color (`"go to the red"`); the agent must walk onto that square. `env.py` renders each state as a 32×32 RGB image and includes a scripted "expert" that always knows the correct next move — that's the teacher whose demonstrations get imitated.

### The model

`model.py` builds the sequence `[ v1 ... v16 | t1 ... t4 | BOA ]`:

- **16 visual tokens** — a small CNN downsamples the 32×32 image to a 4×4 grid of features, then a linear projector maps them into the transformer's embedding space.
- **4 instruction tokens** — a minimal word-level vocabulary (`go`, `to`, `the`, plus the four colors).
- **1 `<BOA>` ("beginning of action") token** — a fixed query embedding. It carries no information about the current scene by itself; causal self-attention lets it read every vision and text token that came before it, so its *final* hidden state ends up context-dependent even though its input embedding never changes.

The vocabulary is the language words **plus four action tokens** — `<UP>`, `<DOWN>`, `<LEFT>`, `<RIGHT>` — appended exactly the way RT-2/OpenVLA extend a real LLM's vocabulary with action tokens. The output head is weight-tied to the input embedding (GPT-2 style) and scores the *entire* vocabulary; training only ever supervises the 4 action-token logits at the `<BOA>` position, and inference does constrained decoding — argmax restricted to just those 4 ids.

**Where the answer actually comes from:** the model isn't told which action is correct. `data.py` rolls the scripted expert forward thousands of times, recording `(image, instruction, expert's action)` at every step as the behavior-cloning dataset. `train.py` runs plain cross-entropy between the model's prediction and that label — the action token is only ever used to compute the loss, never fed into the network as input. Nothing "knows" the answer up front; the weights get pushed toward it over thousands of gradient steps.

## Quickstart

```bash
uv sync                 # installs torch, numpy, pillow, etc. from pyproject.toml/uv.lock
uv run python train.py     # generates a fresh dataset, trains ~20 epochs, saves checkpoint.pt
uv run python evaluate.py  # closed-loop rollout success rate + saves rollout_example.gif
```

To poke at the pieces directly:

```python
from env import GridWorld
env = GridWorld()
image, instruction = env.reset()
print(instruction)          # e.g. "go to the blue"
print(env.expert_action())  # the scripted "correct" move from here
```

## Results — and the interesting part

On held-out episodes: **72% single-step action accuracy**, but only **54% closed-loop task success** over 200 full episodes.

That gap is the point. A per-step accuracy well above random doesn't guarantee the agent reaches the goal, because behavior cloning suffers from **compounding error / distribution shift**: once the agent drifts even slightly off the states the expert's demonstrations covered, it's now in a situation training never taught it about, and its predictions get worse from there — small errors snowball over a multi-step rollout. This is the same reason real systems don't stop at plain imitation learning:

- **RT-1 / RT-2 / OpenVLA** lean on enormous, diverse demonstration datasets to cover far more of state space.
- **ACT / π0** predict short *chunks* of future actions per forward pass, reducing how many independent decisions can compound.
- **DAgger**-style methods query the expert on states the *policy itself* visits (not just its own demonstrations), directly correcting the distribution mismatch.

A natural next experiment on this codebase: implement one round of DAgger — run the trained model, relabel the states it actually visits with `env.expert_action()`, retrain — and watch the closed-loop success rate move.

## LLM vs. VLM vs. VLA, side by side

| | LLM | VLM | VLA |
|---|---|---|---|
| Input tokens | text only | text + image patches (projected) | text + image patches (projected) |
| Vocabulary | language tokens | language tokens | language tokens **+ action tokens** |
| Output head | softmax over vocab | softmax over vocab | softmax over vocab, **decoding constrained to action ids** |
| Training loss | next-token cross-entropy | next-token cross-entropy (captions / VQA) | cross-entropy vs. the **expert's action token** (behavior cloning) |
| Control loop | none — batch generation | none — one question, one answer | **closed loop:** action → `env.step()` → new image → repeat |
| In this repo | — | vision encoder + projector (`model.py`) | + action ids (`tokenizer.py`), + render/step loop (`env.py`) |

## Design choices and simplifications

Everything here is sized to be understood, not to be state of the art:

- **Toy grid world instead of a physics simulator** — full control over every pixel, no simulator install, fast iteration on CPU.
- **Discretized single-token actions instead of continuous control** — matches the RT-2/OpenVLA "extend the vocabulary" mechanism directly; a real robot with continuous joint angles would need either this repo's approach applied per-dimension (multiple action tokens, decoded autoregressively) or a continuous regression/diffusion head instead.
- **A small CNN trained from scratch instead of a pretrained ViT/DINOv2/SigLIP** — the images here are simple synthetic scenes, so a few conv layers are enough; real VLAs use large pretrained vision backbones because real camera images need it.
- **A ~3-layer, ~64-dim transformer instead of a pretrained LLM** — trained entirely from scratch on synthetic data in minutes, so every weight in the model was shaped only by what's in this repo, with nothing borrowed from a pretrained checkpoint.

## References

- [RT-1](https://robotics-transformer1.github.io/) — CNN + Transformer, discretized actions.
- [RT-2](https://robotics-transformer2.github.io/) — fine-tunes a large VLM with action tokens appended to its vocabulary; the direct inspiration for this repo's action head.
- [OpenVLA](https://openvla.github.io/) — open-source RT-2-style model: Llama-2 backbone + DINOv2/SigLIP vision encoder.
- [Octo](https://octo-models.github.io/) / [π0 (Physical Intelligence)](https://www.physicalintelligence.company/blog/pi0) — transformer trunk + continuous/flow-matching action head, action chunking.
- [LeRobot](https://github.com/huggingface/lerobot) / [SmolVLA](https://huggingface.co/lerobot/smolvla_base) — the real pretrained model `main.py` runs.
