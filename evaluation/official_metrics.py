import numpy as np
from .eval_metrics import f1_from_binary
from .generate_summary import summarize_with_budget
from selection.segment_kts import kts_change_points, segments_from_cps


def gt_binary_from_user_scores(user_scores, budget_ratio=0.15, use_kts=True, feats_for_kts=None, max_ncp=8):
    T, U = user_scores.shape
    if use_kts:
        if feats_for_kts is None:
            # fall back to KTS on user mean score
            feats = user_scores.mean(axis=1).reshape(-1, 1)
        else:
            feats = feats_for_kts
        cps = kts_change_points(feats.astype(np.float64), max_ncp=max_ncp, kernel="linear")
        segs = segments_from_cps(T, cps)
    else:
        step = max(5, T // 10);
        segs = [(i, min(T, i + step)) for i in range(0, T, step)]
    # For each user, select segments within budget using user's score as value
    budget = max(1, int(T * budget_ratio))
    gt_bins = []
    for u in range(U):
        scores_u = user_scores[:, u]
        values = np.array([scores_u[s:e].sum() for (s, e) in segs], dtype=np.float32)
        lengths = np.array([e - s for (s, e) in segs], dtype=np.int32)
        # knapsack selection
        from selection.knapsack import knapsack_select
        sel = knapsack_select(values, lengths, budget)
        bin_mask = np.zeros(T, dtype=int)
        for on, (s, e) in zip(sel, segs):
            if on: bin_mask[s:e] = 1
        gt_bins.append(bin_mask)
    return np.stack(gt_bins, axis=1)  # (T, U)


def f1_max_avg(pred_binary, gt_binaries):
    # pred_binary: (T,), gt_binaries: (T, U)
    U = gt_binaries.shape[1]
    f1s = []
    for u in range(U):
        f1s.append(f1_from_binary(pred_binary, gt_binaries[:, u]))
    return float(np.max(f1s)), float(np.mean(f1s))
