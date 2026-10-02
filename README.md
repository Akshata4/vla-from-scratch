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

A full visual walkthrough of this — three side-by-side pipelines for LLM/VLM/VLA, plus a zoom into exactly what happens inside one forward pass of the model in this repo — is in [`docs/vla-anatomy.html`](docs/vla-anatomy.html). A second diagram, [`docs/vla-pipeline.html`](docs/vla-pipeline.html), traces one real example through the trained checkpoint stage by stage, with real tensor shapes and real numbers, once for training and once for inference. Both are self-contained HTML files; download and open in a browser (GitHub doesn't render arbitrary HTML pages inline). There's also a **live, interactive version**, [`docs/vla-explainer.html`](docs/vla-explainer.html) — a hand-written JavaScript port of `model.py` running the real exported weights client-side, where you can edit the scene and watch every stage recompute in real time (see "Live explainer" below).

## What's actually in this repo

Three ways to see a VLA, plus the shared architecture they all use:

| | What it is | Where |
|---|---|---|
| **`model.py`** | The `VLA` class — vision encoder, transformer, weight-tied action head. Shared, unchanged, by both experiments below. | repo root |
| **`grid_world/`** | A tiny transformer trained by behavior cloning to solve a toy "walk to the colored square" task, entirely hand-rolled | `env.py`, `tokenizer.py`, `data.py`, `train.py`, `evaluate.py` |
| **`babyai/`** | The same model, same recipe, pointed at a real maintained benchmark ([MiniGrid](https://github.com/Farama-Foundation/Minigrid)'s `BabyAI-GoToLocal-v0`) instead of our own environment | `env.py`, `tokenizer.py`, `data.py`, `train.py`, `evaluate.py` |
| **`main.py`** | Runs Hugging Face's [SmolVLA](https://huggingface.co/lerobot/smolvla_base) on one real frame from an actual robot-arm pick-and-place recording, and prints its predicted action next to what the human demonstrator actually did | repo root |

The point of pairing them: `main.py` shows what a production VLA's output looks like on real robot data; `grid_world/` and `babyai/` show you *why* it's structured that way, by building the same structure yourself at toy scale against two different data sources, both importing the one shared `model.py`.

### The toy task (`grid_world/`)

An 8×8 grid world with 2-4 colored squares and one agent. The instruction names a target color (`"go to the red"`); the agent must walk onto that square. `grid_world/env.py` renders each state as a 32×32 RGB image and includes a scripted "expert" that always knows the correct next move — that's the teacher whose demonstrations get imitated.

### The model

`model.py` builds the sequence `[ v1 ... v16 | t1 ... t4 | BOA ]`:

- **16 visual tokens** — a small CNN downsamples the 32×32 image to a 4×4 grid of features, then a linear projector maps them into the transformer's embedding space.
- **4 instruction tokens** — a minimal word-level vocabulary (`go`, `to`, `the`, plus the four colors).
- **1 `<BOA>` ("beginning of action") token** — a fixed query embedding. It carries no information about the current scene by itself; causal self-attention lets it read every vision and text token that came before it, so its *final* hidden state ends up context-dependent even though its input embedding never changes.

The vocabulary is the language words **plus four action tokens** — `<UP>`, `<DOWN>`, `<LEFT>`, `<RIGHT>` — appended exactly the way RT-2/OpenVLA extend a real LLM's vocabulary with action tokens. The output head is weight-tied to the input embedding (GPT-2 style) and scores the *entire* vocabulary; training only ever supervises the 4 action-token logits at the `<BOA>` position, and inference does constrained decoding — argmax restricted to just those 4 ids.

**Where the answer actually comes from:** the model isn't told which action is correct. `data.py` rolls the scripted expert forward thousands of times, recording `(image, instruction, expert's action)` at every step as the behavior-cloning dataset. `train.py` runs plain cross-entropy between the model's prediction and that label — the action token is only ever used to compute the loss, never fed into the network as input. Nothing "knows" the answer up front; the weights get pushed toward it over thousands of gradient steps.

## Quickstart

```bash
uv sync                           # installs torch, numpy, pillow, minigrid, etc.
uv run python -m grid_world.train     # generates a fresh dataset, trains ~20 epochs, saves grid_world_checkpoint.pt
uv run python -m grid_world.evaluate  # closed-loop rollout success rate + saves rollout_example.gif

uv run python -m babyai.train         # same recipe, BabyAI-GoToLocal-v0 instead
uv run python -m babyai.evaluate      # closed-loop rollout success rate
```

Both are real Python packages (`grid_world/__init__.py`, `babyai/__init__.py`), so run them with `-m` from the repo root — that's what lets their internal `from .env import ...` imports and the shared root-level `from model import VLA` both resolve.

To poke at the pieces directly:

```python
from grid_world.env import GridWorld
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
| In this repo | — | vision encoder + projector (`model.py`) | + action ids (`grid_world/tokenizer.py`), + render/step loop (`grid_world/env.py`) |

## Design choices and simplifications

Everything here is sized to be understood, not to be state of the art:

- **Toy grid world instead of a physics simulator** — full control over every pixel, no simulator install, fast iteration on CPU.
- **Discretized single-token actions instead of continuous control** — matches the RT-2/OpenVLA "extend the vocabulary" mechanism directly; a real robot with continuous joint angles would need either this repo's approach applied per-dimension (multiple action tokens, decoded autoregressively) or a continuous regression/diffusion head instead.
- **A small CNN trained from scratch instead of a pretrained ViT/DINOv2/SigLIP** — the images here are simple synthetic scenes, so a few conv layers are enough; real VLAs use large pretrained vision backbones because real camera images need it.
- **A ~3-layer, ~64-dim transformer instead of a pretrained LLM** — trained entirely from scratch on synthetic data in minutes, so every weight in the model was shaped only by what's in this repo, with nothing borrowed from a pretrained checkpoint.

## Extension: swapping in a real library (BabyAI / MiniGrid)

The toy grid world above is entirely hand-rolled — our own renderer, our own scripted expert, our own single instruction template. `babyai/` swaps the data source for [MiniGrid](https://github.com/Farama-Foundation/Minigrid)'s `BabyAI-GoToLocal-v0`, a maintained benchmark, while reusing the same root-level `model.py`'s `VLA` class completely unchanged — only the package name differs between `from grid_world.train import ...` and `from babyai.train import ...`:

- `babyai/tokenizer.py` — a real mission vocabulary (`go to a red key`, `go to the purple box`, ...: 13 words, 36 unique missions) plus all 7 of BabyAI's action tokens
- `babyai/env.py` — wraps the env for RGB image observations and pairs it with `BabyAIBot`, MiniGrid's own built-in expert (solved 300/300 sampled episodes with zero failures — a much stronger label source than a hand-written rule)
- `babyai/data.py` / `babyai/train.py` / `babyai/evaluate.py` — same behavior-cloning recipe as before, new data source

**Camera view matters more than anything else we tried.** The first version used `RGBImgObsWrapper`, a full top-down view of the whole room: 69% single-step accuracy but only ~18-20% closed-loop success — a much bigger drop than the original toy's 72% → 54%. Feeding the native 64×64 render instead of downsizing to 32×32 made no measurable difference (69.1%/17.5% vs 69.5%/18%), so it wasn't image fidelity. The real suspect was BabyAI's actions being **egocentric and relative** (`left`/`right`/`forward` depend on which way the agent is currently facing) versus our toy's **absolute** compass directions — from a top-down view, picking the right action means jointly reading the target's position *and* decoding the agent's tiny heading-indicator triangle, then mentally rotating one against the other.

Switching to `RGBImgPartialObsWrapper` (an egocentric view — the agent only sees what's directly in front of it, not the whole room) fixed almost all of it: **75.8% single-step accuracy, 53% closed-loop success** — back in line with the original toy, despite the task now being genuinely partially observable (the target is often off-screen entirely, and the model has no memory across frames — one image in, one action out). The expected failure mode of partial observability — identical "empty wall ahead" frames needing different actions depending on an off-screen target's unseen location — turned out not to dominate in practice. The lesson: an egocentric view turns "which way do I turn" into something close to directly readable from the image (object visible and off-center → turn toward it; object centered → go forward), whereas the top-down view demanded an extra, harder, rotation-dependent inference step. Matching the camera convention to the action convention mattered more than resolution, dataset size, or anything else adjusted here.

(`BabyAIBot`, the label source, always plans from the environment's full internal grid state regardless of which wrapper the learner's image comes from — it's a privileged-information demonstrator, same role our own `env.expert_action()` played, not something the trained model has access to.)

## Live explainer

`docs/vla-explainer.html` is an interactive, in-browser version of the `grid_world` model, inspired by [Transformer Explainer](https://poloclub.github.io/transformer-explainer/). `export_weights.py` dumps every parameter of the trained `grid_world` checkpoint to `docs/vla_weights.json` (174,592 numbers), and the page's own hand-written JavaScript forward pass — conv layers, causal self-attention, layer norm, the weight-tied head, all reimplemented from scratch — loads that JSON and recomputes the real model on every click. No video, no precomputed frames: click a grid cell to move the agent or an object, pick the instruction, and the vision tokens, attention weights, and logits all update live. Toggle Training mode to see the scripted expert's label and the real cross-entropy loss; toggle Inference mode to see constrained decoding and step the agent using the model's own choice.

```bash
uv run python export_weights.py   # regenerate docs/vla_weights.json after retraining grid_world
```

The JS forward pass was checked against the real PyTorch output on a fixed example before any UI was built (matched to within ~0.005, rounding-level precision) — see the git history for the verification script.

## References

- [RT-1](https://robotics-transformer1.github.io/) — CNN + Transformer, discretized actions.
- [RT-2](https://robotics-transformer2.github.io/) — fine-tunes a large VLM with action tokens appended to its vocabulary; the direct inspiration for this repo's action head.
- [OpenVLA](https://openvla.github.io/) — open-source RT-2-style model: Llama-2 backbone + DINOv2/SigLIP vision encoder.
- [Octo](https://octo-models.github.io/) / [π0 (Physical Intelligence)](https://www.physicalintelligence.company/blog/pi0) — transformer trunk + continuous/flow-matching action head, action chunking.
- [LeRobot](https://github.com/huggingface/lerobot) / [SmolVLA](https://huggingface.co/lerobot/smolvla_base) — the real pretrained model `main.py` runs.
