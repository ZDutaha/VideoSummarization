import numpy as np
import os
import torch
import yaml
from evaluation.eval_metrics import f1_from_binary, spearman_corr, kendall_corr
from evaluation.generate_summary import summarize_with_budget
from model.model_main import MultiModalSummarizer
from tqdm import tqdm
from utils.data_loader import VideoSummaryDataset

from scipy.special import expit as sigmoid


def _is_binary(arr):
    u = np.unique(arr)
    return set(u.tolist()).issubset({0, 1})


def _topk_binarize(scores, k):
    T = len(scores)
    k = int(max(1, min(T, k)))
    # 先用 argpartition 选出 top-k，再用 argsort 保证恰好 k 个
    idx_topk = np.argpartition(scores, -k)[-k:]
    # 处理分数并列导致的 >k 个的问题
    order = idx_topk[np.argsort(-scores[idx_topk], kind="mergesort")]
    binmask = np.zeros(T, dtype=np.uint8)
    binmask[order[:k]] = 1
    return binmask


def load_flexible(model, state):
    msd = model.state_dict()
    keep, dropped = {}, []
    for k, v in state.items():
        if k in msd and msd[k].shape == v.shape:
            keep[k] = v
        else:
            dropped.append(k)
    missing, unexpected = model.load_state_dict(keep, strict=False)
    print(f"[load_flexible] loaded={len(keep)} dropped={len(dropped)} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if dropped:
        print("[load_flexible] dropped (shape mismatch) example:", dropped[:8])


def load_config(p="experiments/config.yaml"):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def _resize_to_len(arr, L):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] == L:
        return arr
    x_old = np.linspace(0.0, 1.0, arr.shape[0])
    x_new = np.linspace(0.0, 1.0, L)
    return np.interp(x_new, x_old, arr).astype(np.float32)


def run_eval(cfg):
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    ds = VideoSummaryDataset(cfg["data_root"], split="test", dataset_name=cfg.get("dataset", "toy"))
    model = MultiModalSummarizer(cfg["model"]["video_dim"], cfg["model"]["text_dim"], hidden=cfg["model"]["hidden"],
                                 heads=cfg["model"]["heads"], gat_layers=cfg["model"]["gat_layers"],
                                 dropout=cfg["model"]["dropout"], use_text=cfg["model"]["use_text"],
                                 use_contrastive=False).to(device)
    ckpt_path = os.path.join(cfg["save_dir"], "best.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        load_flexible(model, state)
    model.eval()
    f1s, rhos, taus = [], [], []
    for i in tqdm(range(len(ds))):
        item = ds[i]
        feats = item["feats"].to(device)
        labels = item["labels"].cpu().numpy().astype(int)
        texts_arr = item["texts"]
        texts = None
        if isinstance(texts_arr, list):
            # if list[str], you should run encode_texts_sbert first; here fallback to zeros
            pass
        else:
            import torch as th, numpy as np
            texts = th.from_numpy(np.asarray(texts_arr)).float().to(device)
        with torch.no_grad():
            scores = model(feats, texts)["scores"].detach().cpu().numpy()
        pred_bin = summarize_with_budget(scores, cfg["eval"]["budget_ratio"], use_kts=True, max_ncp=8)
        f1s.append(f1_from_binary(pred_bin, labels))
        rhos.append(spearman_corr(scores, labels))
        taus.append(kendall_corr(scores, labels))
    return float(np.mean(f1s)), float(np.mean(rhos)), float(np.mean(taus))


if __name__ == "__main__":
    cfg = load_config()
    f1, rho, tau = run_eval(cfg)
    print(f"[TEST] F1={f1:.4f}  Spearman={rho:.4f}  Kendall={tau:.4f}")
