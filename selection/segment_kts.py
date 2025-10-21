import numpy as np
def _cumsum_triu(K): return np.cumsum(np.cumsum(K, axis=0), axis=1)

def kts_change_points(feats, max_ncp=5, kernel="linear", sigma=1.0):
    X = feats.astype(np.float64); T = X.shape[0]
    if kernel == "linear":
        K = X @ X.T
    else:
        X2 = np.sum(X*X, axis=1, keepdims=True)
        d2 = X2 + X2.T - 2*(X@X.T)
        K = np.exp(-0.5 * d2 / (sigma**2 + 1e-8))
    K2 = _cumsum_triu(K)
    J = np.zeros((T, T))
    for s in range(T):
        for e in range(s+1, T+1):
            tot = K2[e-1, e-1]
            if s>0: tot = tot - K2[s-1, e-1] - K2[e-1, s-1] + K2[s-1, s-1]
            J[s, e-1] = tot / (e - s)
    m = max_ncp
    DP = np.full((m+1, T), -1e18); P = -np.ones((m+1, T), dtype=int)
    for t in range(T): DP[0, t] = J[0, t]
    for k in range(1, m+1):
        for t in range(k, T):
            best, arg = -1e18, -1
            for s in range(k, t+1):
                val = DP[k-1, s-1] + J[s, t]
                if val > best: best, arg = val, s
            DP[k, t] = best; P[k, t] = arg
    kstar = int(np.argmax(DP[:, T-1]))
    cps = []; t = T-1; k = kstar
    while k>0:
        s = P[k, t]; cps.append(int(s)); t = s-1; k -= 1
    return sorted(cps)

def segments_from_cps(T, cps):
    cps = [0] + list(cps) + [T]
    return [(cps[i], cps[i+1]) for i in range(len(cps)-1)]
