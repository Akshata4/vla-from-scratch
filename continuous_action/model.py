"""
The continuous-action VLA.

Compare this file to the root model.py's VLA class side by side -- that's
the whole point of this package. Everything through the transformer is
IDENTICAL and literally imported, not rewritten:

    - VisionEncoder  (image -> 16 visual tokens)      <- same object
    - TransformerBlock (causal self-attn + MLP)        <- same object
    - the [ v_1 ... v_16 | t_1 ... t_Nt | BOA ] sequence layout  <- same idea

The ONLY thing that changes is what happens to BOA's final hidden state:

    discrete (model.py's VLA):   Linear(64 -> vocab_size), weight-tied to
                                  the text embedding, softmax + cross-entropy
                                  against an action-TOKEN id (one of 13-22
                                  classes depending on the vocabulary).

    continuous (this file):      Linear(64 -> 2), MSE against a continuous
                                  (dx, dy) TARGET VECTOR. No vocabulary
                                  entry for actions exists at all -- the
                                  model is doing regression, not
                                  classification.

Real systems split along exactly this line too: RT-2/OpenVLA are the
"discrete token" family, RT-1/ACT/Diffusion Policy/pi-0 are the
"continuous head" family (pi-0 goes further and uses flow matching instead
of plain MSE, but the core idea -- regress a vector instead of classify a
token -- is the same one implemented here in its simplest form).
"""

import torch
import torch.nn as nn

from model import N_VISION_TOKENS, TransformerBlock, VisionEncoder


class VLAContinuous(nn.Module):
    def __init__(self, vocab_size, n_text_tokens,
                 d_model=64, n_heads=4, n_layers=3, n_vision_tokens=N_VISION_TOKENS):
        super().__init__()
        seq_len = n_vision_tokens + n_text_tokens + 1  # +1 for the BOA query position

        self.vision_encoder = VisionEncoder(d_model)
        self.text_embed = nn.Embedding(vocab_size, d_model)

        # BOA is a plain learned parameter here, NOT a lookup into
        # text_embed like the discrete VLA uses. There, tying BOA to a real
        # vocabulary entry mattered because the output head is weight-tied
        # to that same embedding table. Here there's no vocabulary for
        # actions to tie anything to, so BOA is just its own free parameter
        # -- still a fixed, content-free "decide now" signal either way.
        self.boa_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

        # THE key difference from model.py's VLA: a 2-number regression
        # head instead of a vocab_size-way classification head.
        self.action_head = nn.Linear(d_model, 2)

        mask = torch.full((seq_len, seq_len), float("-inf")).triu(1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, images, text_ids):
        b = images.shape[0]
        v = self.vision_encoder(images)                     # (B, 16, D)      -- unchanged from model.py
        t = self.text_embed(text_ids)                        # (B, Nt, D)      -- unchanged from model.py
        boa = self.boa_query.expand(b, -1, -1)                # (B, 1, D)
        x = torch.cat([v, t, boa], dim=1) + self.pos_embed    # (B, 16+Nt+1, D)
        for block in self.blocks:                             # causal self-attn + MLP x3 -- unchanged
            x = block(x, self.causal_mask)
        x = self.ln_f(x)
        return self.action_head(x[:, -1])                     # (B, 2) -- raw (dx, dy), NOT logits
