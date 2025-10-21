# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from model.layers.gat import GATBlock
from utils.graph_utils import build_window_adj_mask

class VideoEncoder(nn.Module):
    """
    输入: x (B,T,D) 或 (T,D)，D=video_dim (如 ResNet50->2048)
    输出: (B,T,H)，H=hidden
    """
    def __init__(self, vid_dim: int, hidden: int = 256, heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, win: int = 15):
        super().__init__()
        self.hidden = hidden
        self.win = win
        # 正确的首层投影: D(=vid_dim) -> H
        self.proj = nn.Linear(vid_dim, hidden)
        self.blocks = nn.ModuleList([GATBlock(hidden, heads, dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor = None):
        # 统一到 (B,T,D)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, T, D = x.shape

        # 如果没传邻接掩码，按窗口构造 (T,T) 的邻接布尔矩阵
        if adj_mask is None:
            adj_mask = build_window_adj_mask(T, self.win, device=x.device)  # (T,T) bool/0-1

        # 首层降维到 H
        if getattr(self.proj, "in_features", None) != D:
            # 保险: 若配置与特征维不一致，动态重建线性层（不会影响本次从头训练）
            new_proj = nn.Linear(D, self.hidden).to(x.device).type(x.dtype)
            self.proj = new_proj

        x = self.proj(x)  # (B,T,H)

        # 堆叠 GAT block（内部已处理多头注意力与 mask）
        for blk in self.blocks:
            x = blk(x, adj_mask)

        return self.ln(x)  # (B,T,H)
