import numpy as np
from selection.knapsack import knapsack_select
from selection.segment_kts import kts_change_points, segments_from_cps


def summarize_with_budget(scores, budget_ratio=0.15, use_kts=True, max_ncp=8):
    T = len(scores);
    budget = int(T * budget_ratio);
    budget = max(1, budget)
    if use_kts:
        feats = scores.reshape(-1, 1).astype(np.float64)
        cps = kts_change_points(feats, max_ncp=max_ncp, kernel="linear")
        segs = segments_from_cps(T, cps)
    else:
        step = max(5, T // 10);
        segs = [(i, min(T, i + step)) for i in range(0, T, step)]
    values = np.array([scores[s:e].sum() for (s, e) in segs], dtype=np.float32)
    lengths = np.array([e - s for (s, e) in segs], dtype=np.int32)
    sel = knapsack_select(values, lengths, budget)
    bin_mask = np.zeros(T, dtype=int)
    for on, (s, e) in zip(sel, segs):
        if on: bin_mask[s:e] = 1
    return bin_mask
