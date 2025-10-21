import torch, torch.nn as nn
from .gat import MaskedGraphAttention


class MultiScaleEncoder(nn.Module):
    def __init__(self, dim: int, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.frame_blocks = nn.ModuleList([MaskedGraphAttention(dim, heads, dropout) for _ in range(layers)])
        self.seg_blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True, dropout=dropout) for _ in
             range(layers)])
        self.fuse = nn.Linear(dim * 2, dim)

    def forward(self, x_frames, mask_frames, x_segs, mask_segs, frame_adj_mask):
        xf = x_frames
        for blk in self.frame_blocks: xf = blk(xf, frame_adj_mask)
        xs = x_segs
        for blk in self.seg_blocks: xs = blk(xs, src_key_padding_mask=(~mask_segs))
        B, T, D = xf.shape;
        Ts = xs.shape[1]
        idx = torch.linspace(0, Ts - 1, steps=T, device=xf.device).round().long().clamp(0, Ts - 1)
        xs_to_f = xs[:, idx, :]
        x_fused = self.fuse(torch.cat([xf, xs_to_f], dim=-1))
        return x_fused, xs
