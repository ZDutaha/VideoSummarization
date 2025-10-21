import numpy as np, torch, random
from pathlib import Path

def make_toy(root="data/processed/toy", T=160, Dv=256, St=20, seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    for split in ["train","val","test"]:
        d = Path(root)/split; d.mkdir(parents=True, exist_ok=True)
        for i in range(6 if split=="train" else (2 if split=="val" else 2)):
            feats = np.random.randn(T,Dv).astype("float32")
            labels = np.zeros(T, dtype=np.float32)
            for _ in range(3):
                s = np.random.randint(0, T-20); e = min(T, s+np.random.randint(10,25))
                labels[s:e] = 1.0; feats[s:e] += 0.8 + 0.5*np.random.randn(1,Dv).astype("float32")
            # generate per-segment sentences, then SBERT脚本会编码
            texts = np.array([f"segment {j} with salient content level {np.random.randint(0,5)}" for j in range(St)], dtype=object)
            frame2sent = np.linspace(0, St-1, T).round().astype("int64")
            np.savez_compressed(d/f"vid_{i:03d}.npz",
                feats=feats, labels=labels, frame2sent=frame2sent, texts=texts)
    print("[Toy] dataset generated. Note: texts are strings; run SBERT encoding if you enable cross-modal.")

def train():
    from train.train import load_config, train_one
    cfg = load_config("experiments/config.yaml"); train_one(cfg)

def test():
    from evaluation.test import load_config, run_eval
    cfg = load_config("experiments/config.yaml")
    f1, rho, tau = run_eval(cfg)
    print(f"[Toy/Test] F1={f1:.4f} Spearman={rho:.4f} Kendall={tau:.4f}")

if __name__ == "__main__":
    make_toy(); train(); test()
