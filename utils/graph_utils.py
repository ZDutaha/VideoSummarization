import torch

def build_window_adj_mask(T:int, win:int=8, device=None):
    m = torch.zeros((T, T), dtype=torch.bool, device=device)
    for i in range(T):
        lo, hi = max(0, i-win), min(T, i+win+1)
        m[i, lo:hi] = True
    return m

def make_segment_indices(T:int, seg_len:int):
    segs = []
    s=0
    while s<T:
        e = min(T, s+seg_len)
        segs.append((s,e)); s=e
    return segs

def segment_pool(x:torch.Tensor, segs):
    pools = [x[s:e].mean(dim=0, keepdim=True) for (s,e) in segs]
    return torch.cat(pools, dim=0)
