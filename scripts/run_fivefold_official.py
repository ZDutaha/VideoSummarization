# scripts/run_fivefold_official.py
import os, json, time, yaml
from copy import deepcopy

from train.train import train_one
from evaluation.test import run_eval

CFG_PATH = "experiments/config.yaml"

def _load_cfg():
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f) or {}

def _ensure(d, *keys):
    """Create nested dicts if not exist and return leaf dict."""
    for k in keys:
        d = d.setdefault(k, {})
    return d

def run(datasets):
    logs_dir = "./experiments/logs"
    os.makedirs(logs_dir, exist_ok=True)

    for dataset_name in datasets:
        print(f"Running {dataset_name} five-fold (official splits)...")
        folds_metrics = []

        for k in range(1, 6):
            # ------- 构建当前 fold 的配置 -------
            cfg = _load_cfg()
            cfg["data_root"] = "./data/processed"
            cfg["dataset"]   = f"{dataset_name}/folds/fold_{k}"
            cfg["save_dir"]  = f"./experiments/saved_models/{dataset_name}_fold_{k}"
            cfg["log_dir"]   = "./experiments/logs"

            # 模型强制与我们提取的 ResNet50(2048) + SBERT(384) 对齐
            _ensure(cfg, "model")
            cfg["model"].update({
                "video_dim": 2048,
                "use_text":  True,
                "text_dim":  384,
            })
            # 训练超参：如 config.yaml 没写 epochs，这里兜底到 60
            _ensure(cfg, "training")
            cfg["training"].setdefault("epochs", 60)

            # ------- 训练 -------
            print(f"\n=== Train {dataset_name} Fold {k} ===")
            train_one(cfg)

            # ------- 评测 -------
            print(f"\n=== Eval  {dataset_name} Fold {k} ===")
            cfg_eval = deepcopy(cfg)              # 复用同一份配置（包含 save_dir）
            cfg_eval["model"]["video_dim"] = 2048 # 再次明确，防止外部改动
            f1, rho, tau = run_eval(cfg_eval)

            folds_metrics.append({
                "fold": k,
                "ok": True,
                "metrics": {
                    "F1": float(f1),
                    "Spearman": float(rho),
                    "Kendall": float(tau)
                }
            })

        # ------- 汇总并落盘 -------
        f1s = [m["metrics"]["F1"] for m in folds_metrics]
        out = {
            "dataset": dataset_name,
            "folds": folds_metrics,
            "F1_max": float(max(f1s)) if f1s else None,
            "F1_avg": float(sum(f1s)/len(f1s)) if f1s else None,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        out_path = os.path.join(logs_dir, f"fivefold_{dataset_name.lower()}_results.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    # 当前先只跑 TVSum；SumMe 我们等官方包问题处理好再启用
    run(["TVSum"])
