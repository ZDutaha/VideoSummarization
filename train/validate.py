import torch, numpy as np
from evaluation.eval_metrics import f1_from_binary, choose_topk_binary

def validate_step(model, feats, labels, texts=None, budget_ratio=0.15):
    model.eval()
    with torch.no_grad(): scores = model(feats, texts)["scores"].detach().cpu().numpy()
    pred_bin = choose_topk_binary(scores, budget_ratio)
    return f1_from_binary(pred_bin, labels.cpu().numpy().astype(int))
