import numpy as np
from scipy.stats import kendalltau, spearmanr


def _resize_binary(pred, L):
    import numpy as np
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    if pred.shape[0] == L:
        return (pred > 0.5).astype(np.int32)
    x_old = np.linspace(0.0, 1.0, pred.shape[0])
    x_new = np.linspace(0.0, 1.0, L)
    out = np.interp(x_new, x_old, pred)
    return (out > 0.5).astype(np.int32)

def f1_from_binary(pred, gt):
    import numpy as np
    gt = np.asarray(gt, dtype=np.int32).reshape(-1)
    pred = np.asarray(pred).reshape(-1)

    # 若 pred 不是严格的 0/1，则视为分数先阈值化
    if not np.array_equal(np.unique(pred), np.array([0,1])):
        pred = (pred > 0.5).astype(np.int32)

    # 长度不等则把 pred 重采样到 gt 的长度
    if pred.shape[0] != gt.shape[0]:
        pred = _resize_binary(pred, gt.shape[0])

    tp = int((pred * gt).sum())
    fp = int((pred * (1 - gt)).sum())
    fn = int(((1 - pred) * gt).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


def choose_topk_binary(scores, budget_ratio=0.15):
    T = len(scores);
    k = max(1, int(T * budget_ratio))
    idx = np.argsort(-scores)[:k];
    mask = np.zeros(T, dtype=int);
    mask[idx] = 1;
    return mask


def spearman_corr(pred_scores, gt_scores):
    import numpy as np
    from scipy.stats import spearmanr
    gt = np.asarray(gt_scores, dtype=np.float32).reshape(-1)
    pred = _resize_scores(pred_scores, gt.shape[0])
    res = spearmanr(pred, gt)
    corr = getattr(res, "correlation", res[0] if isinstance(res, tuple) else np.nan)
    if corr is None or np.isnan(corr):
        return 0.0
    return float(corr)


def kendall_corr(pred_scores, gt_scores):
    import numpy as np
    from scipy.stats import kendalltau
    gt = np.asarray(gt_scores, dtype=np.float32).reshape(-1)
    pred = _resize_scores(pred_scores, gt.shape[0])
    res = kendalltau(pred, gt)
    corr = getattr(res, "correlation", res[0] if isinstance(res, tuple) else np.nan)
    if corr is None or np.isnan(corr):
        return 0.0
    return float(corr)


def _resize_scores(pred, L):
    import numpy as np
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    if pred.shape[0] == L:
        return pred
    x_old = np.linspace(0.0, 1.0, pred.shape[0])
    x_new = np.linspace(0.0, 1.0, L)
    return np.interp(x_new, x_old, pred)
