import os, json, time, random
import numpy as np
import torch

from train.train import train_one
from evaluation.test import run_eval


def seed_all(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _fill_defaults(cfg: dict, video_dim: int = 2048):
    """
    补齐 train_one / run_eval 可能访问到的字段，并提供常见 key 的别名以防 KeyError。
    """
    # ---------------- training ----------------
    tr = cfg.setdefault("training", {})
    tr.setdefault("epochs", 60)
    tr.setdefault("batch_size", 1)
    tr.setdefault("device", "cuda")
    tr.setdefault("val_interval", 1)
    tr.setdefault("eval_interval", tr["val_interval"])     # 别名
    tr.setdefault("lr", 2e-4)
    tr.setdefault("weight_decay", 1e-4)
    tr.setdefault("optimizer", "adamw")
    tr.setdefault("scheduler", "cosine")      # or 'none'
    tr.setdefault("min_lr", 1e-6)
    tr.setdefault("warmup_epochs", 5)
    tr.setdefault("grad_accum_steps", 1)
    tr.setdefault("amp", True)                # 混合精度开关
    tr.setdefault("label_smoothing", 0.0)
    tr.setdefault("early_stop_patience", 0)   # 0 表示不启用

    # 梯度裁剪命名不一致：两种都给
    if "grad_clip" not in tr and "clip_grad" in tr:
        tr["grad_clip"] = tr["clip_grad"]
    if "clip_grad" not in tr and "grad_clip" in tr:
        tr["clip_grad"] = tr["grad_clip"]
    tr.setdefault("grad_clip", 1.0)
    tr.setdefault("clip_grad", tr["grad_clip"])

    # dataloader
    tr.setdefault("num_workers", 2)
    tr.setdefault("pin_memory", True)

    # ---------------- model ----------------
    m = cfg.setdefault("model", {})
    m.setdefault("video_dim", video_dim)
    m.setdefault("text_dim", 768)
    m.setdefault("hidden", 256)
    m.setdefault("heads", 4)
    m.setdefault("gat_layers", 2)
    m.setdefault("dropout", 0.1)
    m.setdefault("bidirectional", True)
    m.setdefault("pool", "mean")              # 'mean' | 'max' | 'attn'
    m.setdefault("proj_infer", False)         # 仅推理强制线性投影
    m.setdefault("lock_txtproj", False)       # 冻结文本投影层
    m.setdefault("use_text", True)
    m.setdefault("txt_backbone", "bert-base-uncased")
    m.setdefault("txt_max_len", 64)
    m.setdefault("use_pos_enc", True)
    m.setdefault("graph_type", "gat")         # 'gat' | 'none'
    m.setdefault("fusion", "concat")          # 'concat' | 'gated' | 'sum'

    # 多任务/对比学习
    m.setdefault("use_contrastive", False)
    m.setdefault("contrastive_type", "infoNCE")     # or 'margin'
    m.setdefault("contrastive_temp", 0.07)
    m.setdefault("contrastive_margin", 0.2)
    m.setdefault("contrastive_weight", 0.1)
    m.setdefault("use_alignment", False)
    m.setdefault("alignment_weight", 0.0)
    m.setdefault("aux_cls_weight", 0.0)

    # 输出/阈值
    m.setdefault("score_activation", "sigmoid")     # 'sigmoid' | 'tanh' | 'none'
    m.setdefault("bin_threshold", 0.5)

    # ---------------- eval ----------------
    ev = cfg.setdefault("eval", {})
    ev.setdefault("budget_ratio", 0.15)
    ev.setdefault("bin_threshold", 0.5)
    ev.setdefault("smooth_sigma", 0.0)
    ev.setdefault("eval_batch_size", 1)
    ev.setdefault("ensure_len_match", True)
    ev.setdefault("align_method", "interp")   # 'interp' | 'crop'
    ev.setdefault("ranking", "none")
    ev.setdefault("aggregation", "mean")


def run_fivefold(dataset: str = "TVSum",
                 seed: int = 2025,
                 video_dim: int = 2048,
                 data_root: str | None = None):
    seed_all(seed)
    dataset = dataset.strip()
    data_root = data_root or os.getenv("DATA_ROOT", "data/processed")

    logs_dir = "./experiments/logs"
    os.makedirs(logs_dir, exist_ok=True)

    folds_metrics = []

    for k in range(1, 6):
        save_dir = f"./experiments/saved_models/{dataset}_fold_{k}"
        os.makedirs(save_dir, exist_ok=True)

        # ---- Train cfg ----
        cfg = {
            "seed": seed,
            "fold": k,
            "dataset_name": dataset,
            "dataset": dataset,
            "data_root": data_root,
            "save_dir": save_dir,
            "logs_dir": logs_dir,
            # 常见别名，避免其他模块硬编码不同字段名
            "work_dir": save_dir,
            "output_dir": save_dir,
            "exp_dir": save_dir,

            "model": {"video_dim": video_dim},
            "eval": {},
        }

        print(f"\n=== Train {dataset} Fold {k} ===", flush=True)
        print(f"[paths] data_root={data_root}  save_dir={save_dir}  logs_dir={logs_dir}", flush=True)
        _fill_defaults(cfg, video_dim=video_dim)
        train_one(cfg)

        ckpt = os.path.join(save_dir, "best.pt")

        # ---- Eval cfg ----
        cfg_eval = {
            "seed": seed,
            "fold": k,
            "dataset_name": dataset,
            "dataset": dataset,
            "data_root": data_root,
            "split": "test",

            # 评测阶段也放 save_dir，兼容 evaluation/test.py 的写法
            "save_dir": save_dir,
            "logs_dir": logs_dir,
            "work_dir": save_dir,
            "output_dir": save_dir,
            "exp_dir": save_dir,

            "model": {"video_dim": video_dim},

            # 兼容 evaluation/test.py / 其他脚本的不同字段名
            "load_from": ckpt,
            "model_path": ckpt,
            "resume": ckpt,
            "checkpoint": ckpt,
            "ckpt": ckpt,
            "weights": ckpt,

            "eval": {}
        }
        _fill_defaults(cfg_eval, video_dim=video_dim)

        print(f"\n=== Eval  {dataset} Fold {k} ===", flush=True)
        f1, rho, tau = run_eval(cfg_eval)
        folds_metrics.append({
            "fold": k,
            "ok": True,
            "metrics": {"F1": float(f1), "Spearman": float(rho), "Kendall": float(tau)}
        })

    # ---- aggregate & save ----
    f1_avg = float(sum(m["metrics"]["F1"] for m in folds_metrics) / len(folds_metrics))
    f1_max = float(max(m["metrics"]["F1"] for m in folds_metrics))
    result = {
        "dataset": dataset,
        "folds": folds_metrics,
        "F1_max": f1_max,
        "F1_avg": f1_avg,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = os.path.join(logs_dir, f"fivefold_{dataset.lower()}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {out_path}", flush=True)


if __name__ == "__main__":
    dataset = os.getenv("DATASET", "TVSum")
    seed = int(os.getenv("SEED", "2025"))
    video_dim = int(os.getenv("VIDEO_DIM", "2048"))
    data_root = os.getenv("DATA_ROOT", "data/processed")

    run_fivefold(dataset=dataset, seed=seed, video_dim=video_dim, data_root=data_root)
