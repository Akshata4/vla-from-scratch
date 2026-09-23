import torch
import torch.nn as nn
import torch.nn.functional as F

N_VISION_TOKENS = 16  # the CNN below turns a 32x32 image into a 4x4 grid of tokens


class VisionEncoder(nn.Module):
    """Image -> sequence of visual tokens, the VLA analogue of patchifying in a ViT.

    Three stride-2 convs shrink 32x32 -> 4x4, giving 16 spatial "patches" with
    `conv_channels` features each. `projector` is the LLaVA-style connector:
    a linear layer that maps vision features into the LLM's token space so
    they can sit in the same sequence as text embeddings.
    """

    def __init__(self, d_model, conv_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.projector = nn.Linear(conv_channels, d_model)

    def forward(self, images):
        feats = self.conv(images)                       # (B, C, 4, 4)
        tokens = feats.flatten(2).transpose(1, 2)        # (B, 16, C)
        return self.projector(tokens)                    # (B, 16, D)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask):
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).reshape(b, n, d)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class VLA(nn.Module):
    """Sequence layout: [ v_1 ... v_Nv | t_1 ... t_Nt | BOA ]

    A single causal mask over the whole multimodal sequence means the final
    BOA position attends to every vision and text token that came before it
    — this is exactly how RT-2/OpenVLA feed images+text into an otherwise
    unmodified causal LM. The output head is tied to the input embedding
    (GPT-2 style weight tying) and produces logits over the *entire*
    vocabulary; only action tokens ever appear as training targets at the
    BOA position, and at inference we do constrained decoding by restricting
    the argmax to the action-token subset (`act()` below).
    """

    def __init__(self, vocab_size, n_text_tokens, boa_id, action_ids,
                 d_model=64, n_heads=4, n_layers=3, n_vision_tokens=N_VISION_TOKENS):
        super().__init__()
        self.boa_id = boa_id
        self.register_buffer("action_ids", torch.tensor(action_ids), persistent=False)

        seq_len = n_vision_tokens + n_text_tokens + 1

        self.vision_encoder = VisionEncoder(d_model)
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.text_embed.weight  # weight tying

        mask = torch.full((seq_len, seq_len), float("-inf")).triu(1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, images, text_ids):
        b = images.shape[0]
        v = self.vision_encoder(images)                             # (B, Nv, D)
        t = self.text_embed(text_ids)                                # (B, Nt, D)
        boa = self.text_embed(text_ids.new_full((b, 1), self.boa_id))
        x = torch.cat([v, t, boa], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x, self.causal_mask)
        x = self.ln_f(x)
        return self.lm_head(x[:, -1])                                # (B, vocab_size)

    @torch.no_grad()
    def act(self, images, text_ids):
        logits = self(images, text_ids)
        action_logits = logits[:, self.action_ids]
        return self.action_ids[action_logits.argmax(dim=-1)]
