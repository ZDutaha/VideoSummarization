import numpy as np
def knapsack_select(values, lengths, budget):
    N = len(values); W = int(budget)
    dp = np.zeros((N+1, W+1), dtype=np.float32)
    keep = np.zeros((N+1, W+1), dtype=np.int8)
    for i in range(1, N+1):
        v = float(values[i-1]); w = int(lengths[i-1])
        for b in range(W+1):
            if w <= b and dp[i-1, b-w] + v > dp[i-1, b]:
                dp[i, b] = dp[i-1, b-w] + v; keep[i, b] = 1
            else:
                dp[i, b] = dp[i-1, b]
    sel = np.zeros(N, dtype=np.int32); b = W
    for i in range(N, 0, -1):
        if keep[i, b] == 1: sel[i-1] = 1; b -= int(lengths[i-1])
    return sel
