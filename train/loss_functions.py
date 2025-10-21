# path: train/loss_functions.py
import torch
import torch.nn.functional as F

"""
Losses for video summarization.

Shapes:
- pred_scores: (..., T) or (T,)
- labels     : same shape, float tensor in {0,1}

All losses return a scalar (mean reduction) by default.
"""


def bce_with_logits_balanced(pred_scores: torch.Tensor,
                             labels: torch.Tensor,
                             eps: float = 1e-6,
                             reduction: str = "mean") -> torch.Tensor:
    """
    Class-imbalance aware BCEWithLogits:
      pos_weight = (#neg / #pos).
    """
    logits = pred_scores.float()
    target = labels.float()

    # count positives/negatives on the whole batch
    pos = target.sum()
    neg = target.numel() - pos
    # avoid div-by-zero; clamp so it's at least 1 (不会把正样本权重压到<1)
    pos_weight = (neg / (pos + eps)).clamp(min=1.0).to(logits.device)

    loss = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction=reduction
    )
    return loss


def focal_bce_with_logits(pred_scores: torch.Tensor,
                          labels: torch.Tensor,
                          gamma: float = 1.5,
                          alpha: float | None = None,
                          reduction: str = "mean") -> torch.Tensor:
    """
    Focal BCE with logits.
    - gamma: focusing parameter
    - alpha: positive-class weighting in [0,1]; if None -> no alpha balancing
    Note: 这里的 alpha 与 BCE 的 pos_weight 概念不同，
          alpha 更像是“正类比例系数”，pos_weight 是“正类相对权重”。
    """
    logits = pred_scores.float()
    target = labels.float()

    # per-element BCE
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    # pt = P(correct)
    p = torch.sigmoid(logits).detach()
    pt = p * target + (1.0 - p) * (1.0 - target)
    mod = (1.0 - pt).pow(gamma)

    if alpha is not None:
        # 正类用 alpha，负类用 (1-alpha)
        w_pos = torch.full_like(target, fill_value=alpha)
        w_neg = torch.full_like(target, fill_value=(1.0 - alpha))
        w = torch.where(target > 0.5, w_pos, w_neg)
        mod = mod * w

    loss = mod * bce
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


# -----------------------------
# Backward-compatibility shim
# -----------------------------
def bce_with_logits(pred_scores: torch.Tensor,
                    labels: torch.Tensor,
                    **kwargs) -> torch.Tensor:
    """
    Legacy name kept for compatibility with existing train.py.
    Internally calls the balanced version.
    """
    return bce_with_logits_balanced(pred_scores, labels, **kwargs)
