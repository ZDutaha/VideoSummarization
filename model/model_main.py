import torch, torch.nn as nn
from .modules.encoder import VideoEncoder
from .modules.multimodal_encoder import MultiModalEncoder
from .modules.contrastive_head import ContrastiveHead
from utils.graph_utils import build_window_adj_mask, make_segment_indices, segment_pool
from .layers.fusion import MultiScaleEncoder


class MultiModalSummarizer(nn.Module):
    def __init__(self, video_dim: int, text_dim: int, hidden: int = 256, heads: int = 4, gat_layers: int = 2,
                 dropout: float = 0.1, use_text: bool = True, use_contrastive: bool = True):
        super().__init__()
        self.use_text = use_text;
        self.use_contrastive = use_contrastive
        self.video_enc = VideoEncoder(video_dim, hidden, heads, gat_layers, dropout)
        self.mscale = MultiScaleEncoder(hidden, heads, layers=2, dropout=dropout)
        self.mm_enc = MultiModalEncoder(hidden, hidden, hidden, heads, dropout) if use_text else None
        self.cls = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.contrast = ContrastiveHead(hidden, proj=128, tau=0.07)

    def forward(self, feats, texts=None, budget_ratio: float = 0.15):
        device = feats.device;
        T = feats.size(0)
        adj = build_window_adj_mask(T, win=8, device=device).unsqueeze(0)
        v = self.video_enc(feats.unsqueeze(0), adj)  # (1,T,H)
        segs = make_segment_indices(T, seg_len=max(16, T // 10))
        vs = segment_pool(v.squeeze(0), segs).unsqueeze(0)  # (1,S,H)
        mask_frames = torch.ones((1, T), dtype=torch.bool, device=device)
        mask_segs = torch.ones((1, vs.size(1)), dtype=torch.bool, device=device)
        v_fused, v_segs = self.mscale(v, mask_frames, vs, mask_segs, adj)
        t_global = None
        if self.use_text and texts is not None and (isinstance(texts, torch.Tensor) and texts.numel() > 0):
            t = texts.unsqueeze(0)  # (1,S,H) pre-embedded
            v_fused, t2 = self.mm_enc(v_fused, t)
            t_global = t2.mean(dim=1)
        v_global = v_fused.mean(dim=1)
        scores = self.cls(v_fused).squeeze(-1).squeeze(0)  # (T,)
        out = {"scores": scores}
        if self.training and self.use_contrastive and t_global is not None:
            k = max(1, int(T * budget_ratio))
            topk = torch.topk(scores, k).indices
            summary_emb = v_fused[:, topk, :].mean(dim=1)
            loss_inter = self.contrast.inter_modal_loss(v_global, t_global)
            loss_intra = self.contrast.intra_video_loss(summary_emb, v_global)
            out["loss"] = loss_inter + loss_intra
            out["loss_inter"] = loss_inter.detach();
            out["loss_intra"] = loss_intra.detach()
        return out
