import os, json, random
from pathlib import Path

def build_folds(dataset_root="data/processed/SumMe", out_dir="data/processed/SumMe/folds",
                n_folds=5, seed=2025):
    d = Path(dataset_root)
    videos = sorted([p.stem for p in (d/"train").glob("*.npz")] + [p.stem for p in (d/"val").glob("*.npz")] + [p.stem for p in (d/"test").glob("*.npz")])
    videos = sorted(list(set(videos)))
    rnd = random.Random(seed)
    rnd.shuffle(videos)
    folds = []
    for k in range(n_folds):
        test = videos[k::n_folds]
        train = [v for v in videos if v not in test]
        folds.append({"train":train, "test":test})
    # materialize as folders with symlinks (or copy fallback)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for k,f in enumerate(folds):
        base = out/f"fold_{k+1}"
        for sp in ["train","val","test"]:
            (base/sp).mkdir(parents=True, exist_ok=True)
        # simple split: use last 20% of train as val
        ntr = len(f["train"]); nval = max(1, int(0.2*ntr))
        tr_ids = f["train"][:-nval]; va_ids = f["train"][-nval:]; te_ids = f["test"]
        for split, ids in [("train",tr_ids),("val",va_ids),("test",te_ids)]:
            for vid in ids:
                src = None
                # find source npz in original splits
                for sp in ["train","val","test"]:
                    p = (Path(dataset_root)/sp/f"{vid}.npz")
                    if p.exists(): src = p; break
                if src is None: continue
                dst = base/split/f"{vid}.npz"
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    # fallback to copy
                    from shutil import copy2
                    copy2(src, dst)
    # Save manifest
    (out/"folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    return folds
