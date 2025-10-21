import torch, torch.nn as nn


class CrossModalAttention(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, dim_out: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(dim_q, dim_out)
        self.k = nn.Linear(dim_kv, dim_out)
        self.v = nn.Linear(dim_kv, dim_out)
        self.mha = nn.MultiheadAttention(dim_out, heads, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(dim_out);
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv, kv_allow_mask=None):
        Q, K, V = self.q(q), self.k(kv), self.v(kv)
        attn_mask = None
        if kv_allow_mask is not None:
            attn_mask = (~kv_allow_mask).float() * -1e9
        y, _ = self.mha(Q, K, V, attn_mask=attn_mask)
        return self.ln(q + self.drop(y))
