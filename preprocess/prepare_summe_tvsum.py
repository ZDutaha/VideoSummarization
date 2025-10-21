import os, numpy as np
from pathlib import Path
from scipy.io import loadmat

def _ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def _load_feats_if_any(stem):
    p = Path(f"data/processed/features/{stem}.npz")
    if p.exists():
        Z = np.load(p)
        return Z["feats"].astype(np.float32)
    return None

def _resample(vec, T):
    idx = np.linspace(0, len(vec)-1, T).round().astype(int)
    return vec[idx]

def prepare_summe(raw_root="data/raw/SumMe", out_root="data/processed/SumMe", fps=2.0):
    raw_root = Path(raw_root); out_root = Path(out_root)
    _ensure_dir(out_root/"train"); _ensure_dir(out_root/"val"); _ensure_dir(out_root/"test")
    videos = []
    for vdir in raw_root.glob("*"):
        if vdir.is_dir() and (vdir/"GT").exists():
            videos.append(vdir.name)
    videos = sorted(videos); n=len(videos); tr,va=int(0.6*n),int(0.2*n)
    splits = {"train": videos[:tr], "val": videos[tr:tr+va], "test": videos[tr+va:]}
    for split, vids in splits.items():
        for vid in vids:
            gt_files = list((raw_root/vid/"GT").glob("*.mat"))
            if not gt_files: continue
            mat = loadmat(str(gt_files[0]))
            user_scores = None
            for key in ["user_score","gt_score","scores"]:
                if key in mat:
                    user_scores = mat[key]; break
            if user_scores is None: continue
            T_gt, U = user_scores.shape if user_scores.ndim==2 else (len(user_scores), 1)
            feats = _load_feats_if_any(vid)
            if feats is None:
                feats = np.random.randn(T_gt, 2048).astype(np.float32)
            T = feats.shape[0]
            # resample user_scores to T
            if user_scores.shape[0] != T:
                us = []
                for u in range(user_scores.shape[1] if user_scores.ndim==2 else 1):
                    v = user_scores[:,u] if user_scores.ndim==2 else user_scores
                    us.append(_resample(v, T))
                user_scores_r = np.stack(us, axis=1).astype(np.float32)
            else:
                user_scores_r = user_scores.astype(np.float32)
            mean_score = user_scores_r.mean(axis=1)
            labels = (mean_score >= np.percentile(mean_score, 85)).astype(np.float32)
            texts = np.zeros((max(1, T//8), 256), dtype=np.float32)
            np.savez_compressed(out_root/split/f"{vid}.npz",
                feats=feats, labels=labels, frame2sent=np.zeros(T, dtype=np.int64),
                texts=texts, user_scores=user_scores_r)
            print("[OK]", split, vid, feats.shape, "users", user_scores_r.shape[1])

def prepare_tvsum(raw_root="data/raw/TVSum", out_root="data/processed/TVSum"):
    raw_root = Path(raw_root); out_root = Path(out_root)
    _ensure_dir(out_root/"train"); _ensure_dir(out_root/"val"); _ensure_dir(out_root/"test")
    # parse ydata-tvsum50.mat if exists
    mat_path = list(raw_root.rglob("ydata-tvsum50.mat"))
    if mat_path:
        mat = loadmat(str(mat_path[0]))
        tvsum50 = mat.get("tvsum50", None)
        if tvsum50 is not None:
            entries = tvsum50[0]
            videos = []
            for e in entries:
                name = str(e["video"][0])
                videos.append(name)
            videos = sorted(videos)
            n=len(videos); tr,va=int(0.6*n),int(0.2*n)
            splits = {"train": videos[:tr], "val": videos[tr:tr+va], "test": videos[tr+va:]}
            # map name->user_anno
            dmap = {}
            for e in entries:
                name = str(e["video"][0])
                ua = e["user_anno"]  # (n_users, n_segs)
                dmap[name] = ua
            for split, vids in splits.items():
                for vid in vids:
                    ua = dmap[vid]
                    n_segs = ua.shape[1]
                    feats = _load_feats_if_any(vid)  # look for features by stem
                    T = feats.shape[0] if feats is not None else n_segs*8
                    if feats is None:
                        feats = np.random.randn(T, 2048).astype(np.float32)
                    # upsample seg scores to T
                    user_scores_r = np.zeros((T, ua.shape[0]), dtype=np.float32)
                    idx = np.linspace(0, n_segs-1, T)
                    lo = np.floor(idx).astype(int); hi = np.ceil(idx).astype(int)
                    w = idx - lo
                    for u in range(ua.shape[0]):
                        s = ua[u]
                        user_scores_r[:,u] = (1-w)*s[lo] + w*s[hi.clip(max=n_segs-1)]
                    mean_score = user_scores_r.mean(axis=1)
                    labels = (mean_score >= np.percentile(mean_score, 85)).astype(np.float32)
                    texts = np.zeros((max(1, T//8), 256), dtype=np.float32)
                    np.savez_compressed(out_root/split/f"{vid}.npz",
                        feats=feats, labels=labels, frame2sent=np.zeros(T, dtype=np.int64),
                        texts=texts, user_scores=user_scores_r)
                    print("[OK]", split, vid, feats.shape, "users", user_scores_r.shape[1])
            return
    # fallback synthetic if mat missing
    n_videos = 50
    videos = [f"tvsum_{i:02d}" for i in range(n_videos)]
    n=len(videos); tr,va=int(0.6*n),int(0.2*n)
    splits = {"train": videos[:tr], "val": videos[tr:tr+va], "test": videos[tr+va:]}
    for split, vids in splits.items():
        for vid in vids:
            T = 320
            feats = _load_feats_if_any(vid)
            if feats is None:
                feats = np.random.randn(T, 2048).astype(np.float32)
            T = feats.shape[0]
            U = 5
            # synthesize per-user scores to allow max/avg
            rnd = np.random.rand(T,U).astype(np.float32)
            for _ in range(3):
                s = np.random.randint(0, T-20); e = min(T, s+np.random.randint(10,30))
                rnd[s:e] += 0.8
            mean_score = rnd.mean(axis=1)
            labels = (mean_score >= np.percentile(mean_score, 85)).astype(np.float32)
            texts = np.zeros((max(1, T//8), 256), dtype=np.float32)
            np.savez_compressed(out_root/split/f"{vid}.npz",
                feats=feats, labels=labels, frame2sent=np.zeros(T, dtype=np.int64),
                texts=texts, user_scores=rnd)
            print("[OK]", split, vid, feats.shape, "users", U)

if __name__=="__main__":
    prepare_summe()
    prepare_tvsum()
