# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

def _to_mha_mask(adj_mask, B, T, device):
    """
    将邻接矩阵转成 MultiheadAttention 的 attn_mask (2D, T x T, bool)。
    约定: 邻接 True/1 = 允许注意；MHA 的 bool mask 中 True=禁止注意，故这里取反。
    支持 (T,T) 或 (B,T,T)；若是 (B,T,T) 则取所有样本的“交集”形成共享 (T,T)。
    """
    if adj_mask is None:
        return None
    m = adj_mask
    if not isinstance(m, torch.Tensor):
        m = torch.as_tensor(m)
    m = m.to(device)
    if m.dtype != torch.bool:
        m = m != 0
    if m.dim() == 3:
        if m.size(0) == B:
            m = torch.all(m, dim=0)  # 更保守：仅保留所有样本都允许的边
        else:
            m = m[0]
    # 现在 m: (T,T) True=允许；MHA 需要 True=禁止
    return (~m).to(torch.bool)

class MaskedGraphAttention(nn.Module):
    """
    一个通用的“图注意力”块（用 MHA 实现 + 残差 + FFN + LN）
    期望输入: x (B,T,D) 或 (T,D)
    邻接掩码: adj_mask (T,T) 或 (B,T,T), True/1=允许
    """
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout
        )
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor = None):
        if x.dim() == 2:
            x = x.unsqueeze(0)   # -> (1,T,D)
        B, T, D = x.shape
        attn_mask = _to_mha_mask(adj_mask, B, T, x.device)

        y, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + self.drop(y))
        y = self.ff(x)
        x = self.norm2(x + self.drop(y))
        return x  # (B,T,D)

class GATBlock(MaskedGraphAttention):
    """
    兼容旧接口的别名/子类：多数地方用 hidden 表示维度。
    """
    def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.1):
        super().__init__(dim=hidden, heads=heads, dropout=dropout)
