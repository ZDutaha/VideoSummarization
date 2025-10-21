import torch, torch.nn as nn, torch.nn.functional as F


def _normalize(x, eps=1e-6): return x / (x.norm(dim=-1, keepdim=True) + eps)


class ContrastiveHead(nn.Module):
    def __init__(self, dim: int, proj: int = 128, tau: float = 0.07):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(dim, proj), nn.ReLU(), nn.Linear(proj, proj))
        self.t = nn.Sequential(nn.Linear(dim, proj), nn.ReLU(), nn.Linear(proj, proj))
        self.tau = tau

    def inter_modal_loss(self, vg, tg):
        v = _normalize(self.v(vg));
        t = _normalize(self.t(tg))
        logits = (v @ t.t()) / self.tau
        labels = torch.arange(v.size(0), device=v.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def intra_video_loss(self, s_emb, v_emb, neg_emb=None):
        s = _normalize(self.v(s_emb));
        v = _normalize(self.v(v_emb))
        pos = (s * v).sum(dim=-1, keepdim=True) / self.tau
        logits = pos
        if neg_emb is not None:
            n = _normalize(self.v(neg_emb));
            logits = torch.cat([pos, s @ n.t() / self.tau], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)
