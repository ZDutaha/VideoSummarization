import numpy as np
import os
import torch
import torch.nn.functional as F
import yaml
from model.model_main import MultiModalSummarizer
from tqdm import tqdm
from utils.data_loader import VideoSummaryDataset

from .loss_functions import bce_with_logits
from .optimizer_scheduler import build_optimizer, build_scheduler


def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)


def load_config(p="experiments/config.yaml"):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def train_one(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    ds_tr = VideoSummaryDataset(cfg["data_root"], split="train",
                                dataset_name=cfg.get("dataset_name", cfg.get("dataset", "TVSum")))
    ds_va = VideoSummaryDataset(cfg["data_root"], split="val",
                                dataset_name=cfg.get("dataset_name", cfg.get("dataset", "TVSum")))

    model = MultiModalSummarizer(cfg["model"]["video_dim"], cfg["model"]["text_dim"], hidden=cfg["model"]["hidden"],
                                 heads=cfg["model"]["heads"], gat_layers=cfg["model"]["gat_layers"],
                                 dropout=cfg["model"]["dropout"], use_text=cfg["model"]["use_text"],
                                 use_contrastive=cfg["model"]["use_contrastive"]).to(device)
    opt = build_optimizer(model, lr=cfg["training"]["lr"], wd=cfg["training"]["weight_decay"])
    sched = build_scheduler(opt)

    best_f1, best_path = -1.0, os.path.join(cfg["save_dir"], "best.pt")
    os.makedirs(cfg["save_dir"], exist_ok=True)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train();
        losses = []
        for i in tqdm(range(len(ds_tr)), desc=f"Epoch {epoch}"):
            item = ds_tr[i]
            feats = item["feats"].to(device)
            labels = item["labels"].float().to(device)
            texts_arr = item["texts"]
            texts = None
            if not isinstance(texts_arr, list):
                import torch as th, numpy as np
                texts = th.from_numpy(np.asarray(texts_arr)).float().to(device)

            out = model(feats, texts, budget_ratio=cfg["eval"]["budget_ratio"])
            pred = out["scores"].view(-1)  # (T_pred,)
            lab = labels.view(-1).float()  # (T_lab,)

            if pred.numel() != lab.numel():
                # 线性插值把标签长度对齐到预测长度
                lab = F.interpolate(
                    lab[None, None, :],  # (1,1,T_lab)
                    size=pred.numel(),
                    mode="linear",
                    align_corners=False
                ).view(-1)

            ce = bce_with_logits(pred, lab)
            loss = ce + (out.get("loss", 0.0))
            opt.zero_grad();
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            opt.step();
            losses.append(float(loss.detach().cpu().item()))
        # validation
        f1s = []
        for j in range(len(ds_va)):
            item = ds_va[j]
            feats = item["feats"].to(device);
            labels = item["labels"]
            texts_arr = item["texts"]
            texts = None
            if not isinstance(texts_arr, list):
                import torch as th, numpy as np
                texts = th.from_numpy(np.asarray(texts_arr)).float().to(device)
            model.eval()
            with torch.no_grad():
                scores = model(feats, texts)["scores"].detach().cpu().numpy()
            from evaluation.eval_metrics import choose_topk_binary, f1_from_binary
            pred_bin = choose_topk_binary(scores, cfg["eval"]["budget_ratio"])
            f1s.append(f1_from_binary(pred_bin, labels.cpu().numpy().astype(int)))
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        print(f"[VAL] epoch={epoch} meanF1={mean_f1:.4f}  trainLoss={np.mean(losses):.4f}")
        sched.step(mean_f1)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            torch.save(model.state_dict(), best_path)
    print(f"[TRAIN] Done. Best F1={best_f1:.4f}. Saved: {best_path}")


if __name__ == "__main__":
    cfg = load_config()
    train_one(cfg)
