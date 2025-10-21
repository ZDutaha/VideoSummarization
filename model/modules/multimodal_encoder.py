import torch, torch.nn as nn
from ..layers.multi_head_attn import CrossModalAttention


class MultiModalEncoder(nn.Module):
    def __init__(self, vid_dim: int, txt_dim: int, hidden: int = 256, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.vid_proj = nn.Linear(vid_dim, hidden);
        self.hidden = hidden
        self.txt_proj = None  # lazy init with runtime text dim
        self.v2t = CrossModalAttention(hidden, hidden, hidden, heads, dropout)  # t <- v
        self.t2v = CrossModalAttention(hidden, hidden, hidden, heads, dropout)  # v <- t
        self.ln_v = nn.LayerNorm(hidden);
        self.ln_t = nn.LayerNorm(hidden)

    def forward(self, v, t, t_mask=None):
        v = self.vid_proj(v);
        # lazy init: build txt_proj with actual input dim
        if self.txt_proj is None or (hasattr(self.txt_proj, 'in_features') and self.txt_proj.in_features != t.size(-1)):
            self.txt_proj = nn.Linear(t.size(-1), self.hidden).to(t.device).type(t.dtype)
        t = self.txt_proj(t)
        v2 = self.t2v(v, t, t_mask)
        t2 = self.v2t(t, v)
        return self.ln_v(v2), self.ln_t(t2)