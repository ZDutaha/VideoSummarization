import numpy as np
from scipy.special import expit as sigmoid
from scipy.stats import spearmanr, kendalltau

def moving_average(x, k=9):
    k = max(1, int(k))
    if k <= 1: return x
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k, dtype=np.float32) / k
    return np.convolve(xpad, ker, mode="valid")

def resample_to_len(arr, T):
    if len(arr) == T: return arr.astype(np.float32)
    xs = np.linspace(0, 1, num=len(arr), endpoint=True)
    xi = np.linspace(0, 1, num=T, endpoint=True)
    return np.interp(xi, xs, arr).astype(np.float32)

def sample_at_picks(frame_level, picks):
    picks = np.asarray(picks).astype(int)
    picks = np.clip(picks, 0, len(frame_level) - 1)
    return frame_level[picks]

def expand_from_picks(sample_level, n_frames, picks):
    out = np.zeros(int(n_frames), dtype=np.float32)
    picks = np.asarray(picks).astype(int)
    for i in range(len(picks)):
        a = picks[i]
        b = picks[i + 1] if i + 1 < len(picks) else n_frames
        out[a:b] = sample_level[i]
    return out

def knapsack(values, weights, capacity):
    n = len(values)
    if n == 0 or capacity <= 0: return []
    capacity = int(max(1, capacity))
    dp   = np.zeros((n + 1, capacity + 1), dtype=np.float32)
    keep = np.zeros((n + 1, capacity + 1), dtype=np.uint8)
    for i in range(1, n + 1):
        v, w = float(values[i - 1]), int(max(1, weights[i - 1]))
        for c in range(capacity + 1):
            if w <= c and dp[i - 1, c - w] + v > dp[i - 1, c]:
                dp[i, c] = dp[i - 1, c - w] + v
                keep[i, c] = 1
            else:
                dp[i, c] = dp[i - 1, c]
    c = capacity
    chosen = []
    for i in range(n, 0, -1):
        if keep[i, c] == 1:
            chosen.append(i - 1)
            c -= int(max(1, weights[i - 1]))
    return chosen[::-1]

def select_keyshots_by_knapsack(probs_on_picks, change_points, n_frames, picks, ratio=0.15):
    cps = np.asarray(change_points).astype(int)
    picks = np.asarray(picks).astype(int)
    n_frames = int(n_frames)
    budget = int(round(ratio * n_frames))
    budget = max(1, min(budget, n_frames))

    seg_scores, seg_lens = [], []
    for a, b in cps:
        seg_len = int(b - a + 1)
        seg_len = max(1, seg_len)
        mask = (picks >= a) & (picks <= b)
        seg_score = float(probs_on_picks[mask].mean()) if mask.any() else 0.0
        seg_scores.append(seg_score); seg_lens.append(seg_len)

    chosen = knapsack(seg_scores, seg_lens, budget)
    summary = np.zeros(n_frames, dtype=np.uint8)
    for idx in chosen:
        a, b = cps[idx]
        summary[a:b + 1] = 1
    return summary

def f1_binary(pred, gt):
    pred = pred.astype(np.uint8)
    gt   = gt.astype(np.uint8)
    tp = int((pred & gt).sum())
    fp = int((pred == 1).sum()) - tp
    fn = int((gt   == 1).sum()) - tp
    if tp == 0 and (fp > 0 or fn > 0): return 0.0
    if tp == 0 and fp == 0 and fn == 0: return 1.0
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))

def correlations(pred_on_picks, gt_on_picks):
    r = spearmanr(pred_on_picks, gt_on_picks).correlation
    t = kendalltau(pred_on_picks, gt_on_picks).correlation
    r = 0.0 if r is None or np.isnan(r) else float(r)
    t = 0.0 if t is None or np.isnan(t) else float(t)
    return r, t

def tvsum_metrics(logits_on_picks, data_npz, ratio=0.15, smooth_k=9):
    probs = sigmoid(np.asarray(logits_on_picks).astype(np.float32))
    probs = moving_average(probs, k=smooth_k)

    n_frames = int(data_npz.get('n_frames', len(probs)))
    picks    = data_npz.get('picks', None)
    cps      = data_npz.get('change_points', None)

    if cps is not None and picks is not None:
        machine = select_keyshots_by_knapsack(
            probs_on_picks=probs, change_points=cps, n_frames=n_frames, picks=picks, ratio=ratio
        )
    else:
        k = max(1, int(round(ratio * n_frames)))
        frame_probs = expand_from_picks(probs, n_frames, picks) if picks is not None else resample_to_len(probs, n_frames)
        idx = np.argpartition(frame_probs, -k)[-k:]
        machine = np.zeros(n_frames, dtype=np.uint8); machine[idx] = 1

    if 'user_summary' in data_npz:
        user_sum = np.asarray(data_npz['user_summary']).astype(np.uint8)  # [M, n_frames]
        f1 = float(np.mean([f1_binary(machine, us) for us in user_sum]))
        if picks is not None:
            gt_on_picks = sample_at_picks(user_sum.mean(axis=0).astype(np.float32), picks)
        else:
            gt_on_picks = resample_to_len(user_sum.mean(axis=0).astype(np.float32), len(probs))
    else:
        labels = data_npz.get('labels', None)
        if labels is not None:
            labels = np.asarray(labels)
            if len(labels) == n_frames:
                gt_frame = (labels > 0.5).astype(np.uint8)
            elif picks is not None and len(labels) == len(picks):
                gt_frame = (expand_from_picks(labels.astype(np.float32), n_frames, picks) > 0.5).astype(np.uint8)
            else:
                gt_frame = np.zeros(n_frames, dtype=np.uint8)
            f1 = f1_binary(machine, gt_frame)
            gt_on_picks = sample_at_picks(gt_frame.astype(np.float32), picks) if picks is not None else resample_to_len(gt_frame.astype(np.float32), len(probs))
        else:
            f1, gt_on_picks = 0.0, np.zeros_like(probs)

    rho, tau = correlations(probs, gt_on_picks.astype(np.float32))
    return f1, rho, tau
