"""
The frozen-pretrained-vision VLA.

Compare to model.py's VLA and continuous_action/model.py's VLAContinuous:
both of those build their own VisionEncoder from scratch and train it end
to end alongside everything else. This variant has NO vision encoder
module at all here -- vision feature extraction already happened once,
offline, in data.py (frozen DINOv2), so forward() starts from
already-computed (N, 16, 384) vision tokens, never from raw pixels.

Everything downstream is otherwise identical to model.py's VLA: a
projector -> concat with text + BOA -> causal transformer (literally
imported, not rewritten) -> weight-tied action-token head. This also
reuses grid_world's tokenizer.py unchanged, so the action representation
(discrete tokens) is held constant too -- the ONLY variable this
experiment changes is where the visual tokens come from.
"""

import torch
import torch.nn as nn

from model import TransformerBlock

from .vision_backbone import FEATURE_DIM


class VLAPretrainedVision(nn.Module):
    def __init__(self, vocab_size, n_text_tokens, boa_id, action_ids,
                 d_model=64, n_heads=4, n_layers=3, n_vision_tokens=16):
        super().__init__()
        self.boa_id = boa_id
        self.register_buffer("action_ids", torch.tensor(action_ids), persistent=False)

        seq_len = n_vision_tokens + n_text_tokens + 1

        # the LLaVA-style connector -- same role as model.py's
        # VisionEncoder.projector, just mapping from DINOv2's 384-dim
        # pretrained features instead of our own CNN's 32-dim ones.
        self.projector = nn.Linear(FEATURE_DIM, d_model)
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.text_embed.weight  # weight tying, same as model.py's VLA

        mask = torch.full((seq_len, seq_len), float("-inf")).triu(1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, vision_tokens, text_ids):
        b = text_ids.shape[0]
        v = self.projector(vision_tokens)                             # (B, 16, D) <- (B, 16, 384) pre-extracted
        t = self.text_embed(text_ids)                                  # (B, Nt, D)
        boa = self.text_embed(text_ids.new_full((b, 1), self.boa_id))
        x = torch.cat([v, t, boa], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x, self.causal_mask)
        x = self.ln_f(x)
        return self.lm_head(x[:, -1])

    @torch.no_grad()
    def act(self, vision_tokens, text_ids):
        logits = self(vision_tokens, text_ids)
        action_logits = logits[:, self.action_ids]
        return self.action_ids[action_logits.argmax(dim=-1)]
