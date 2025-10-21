# -*- coding: utf-8 -*-
# file: preprocess/extract_features_torch.py
"""
用 torchvision ResNet 为 TVSum / SumMe 提取视觉特征，并把结果写回
data/processed/<dataset>/**/<video>.npz 的 feats 字段（保留原有其它键）。
若 processed 下尚无对应 npz，则写到 data/processed/<dataset>/features/<video>.npz。
"""

import os
import math
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T


# ------------------------------
# helpers
# ------------------------------
def _get_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，自动切到 CPU")
        return torch.device("cpu")
    return torch.device(name)


def _build_resnet(backbone: str = "resnet50", device: torch.device = torch.device("cuda")) -> nn.Module:
    """
    返回去掉分类头、输出 (N, D) 的特征网络。
    D = 512(resnet18/34) 或 2048(resnet50/101/152)。
    """
    name = backbone.lower()
    if not hasattr(tv.models, name):
        raise ValueError(f"未知 backbone: {backbone}")

    # 兼容不同 torchvision 版本的权重写法
    weights = None
    try:
        mapping = {
            "resnet18": tv.models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": tv.models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": tv.models.ResNet50_Weights.IMAGENET1K_V1,
            "resnet101": tv.models.ResNet101_Weights.IMAGENET1K_V1,
            "resnet152": tv.models.ResNet152_Weights.IMAGENET1K_V1,
        }
        weights = mapping.get(name, None)
        model = getattr(tv.models, name)(weights=weights)
    except Exception:
        # 退化到旧接口
        model = getattr(tv.models, name)(pretrained=True)

    # 去掉分类头 fc，保留 avgpool，最后 Flatten -> (N, D)
    body = nn.Sequential(*list(model.children())[:-1])
    feat = nn.Sequential(body, nn.Flatten(1))
    feat.eval().to(device)
    for p in feat.parameters():
        p.requires_grad_(False)
    return feat


def _build_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def _iter_sampled_frames(cap: cv2.VideoCapture, target_fps: float) -> Tuple[List[np.ndarray], int, float]:
    """按 target_fps 采样帧，返回采样帧列表、原始总帧数、原始 FPS。"""
    ori_fps = cap.get(cv2.CAP_PROP_FPS)
    if not ori_fps or ori_fps <= 1e-6 or math.isnan(ori_fps):
        ori_fps = 30.0
    step = max(int(round(ori_fps / max(1e-6, target_fps))), 1)

    frames = []
    idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    return frames, (total if total > 0 else idx), ori_fps


def _extract_video_feature(
    video_path: Path,
    model: nn.Module,
    transform: T.Compose,
    device: torch.device,
    fps: float,
    batch_size: int,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    frames, _, _ = _iter_sampled_frames(cap, fps)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"未采到任何帧: {video_path}")

    feats = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            tensor = torch.stack([transform(x) for x in batch], dim=0).to(device, non_blocking=True)
            out = model(tensor)  # (B, D)
            feats.append(out.detach().cpu().numpy())
    feats = np.concatenate(feats, axis=0)  # (T, D)
    return feats


def _find_raw_video(dataset_raw_dir: Path, vid_stem: str) -> Optional[Path]:
    """在 <raw>/<dataset>/videos 下寻找与 stem 匹配的视频文件。"""
    video_dir = dataset_raw_dir / "videos"
    # 常见后缀
    for ext in ("mp4", "MP4", "mkv", "avi", "webm", "mov", "MOV"):
        p = video_dir / f"{vid_stem}.{ext}"
        if p.exists():
            return p
    # 放宽：同名不同后缀取第一个
    for p in video_dir.glob(f"{vid_stem}.*"):
        return p
    return None


def _update_npz(npz_path: Path, feats: np.ndarray, n_frames_total: int):
    """
    写回/创建 npz：更新 feats(float32)、n_frames(int)，保留其它键（若存在）。
    """
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
    data["feats"] = feats.astype(np.float32)
    data["n_frames"] = int(n_frames_total)
    np.savez_compressed(npz_path, **data)


def _collect_targets(processed_dir: Path) -> List[Path]:
    """
    优先匹配 processed/<dataset>/**/<vid>.npz（train/val/test 或 fold_*），
    返回这些 npz 的路径以进行“就地更新”。
    """
    return list(processed_dir.rglob("*.npz"))


# ------------------------------
# pipeline
# ------------------------------
def run(dataset: str,
        raw_root: Path,
        processed_root: Path,
        backbone: str,
        fps: float,
        batch_size: int,
        device_str: str,
        img_size: int):
    device = _get_device(device_str)
    model = _build_resnet(backbone, device=device)
    transform = _build_transform(img_size)

    dataset_raw = raw_root / dataset
    dataset_proc = processed_root / dataset

    if not (dataset_raw / "videos").exists():
        raise SystemExit(f"[ERROR] 未找到原始视频目录: {dataset_raw/'videos'}")

    # 1) 优先更新已存在的 npz（保留标注相关键）
    targets = _collect_targets(dataset_proc)

    # 2) 若不存在任何 npz，则退化到写入 processed/<dataset>/features/<stem>.npz
    write_to_features_dir = False
    if len(targets) == 0:
        write_to_features_dir = True
        (dataset_proc / "features").mkdir(parents=True, exist_ok=True)
        vids = []
        for ext in ("*.mp4", "*.MP4", "*.mkv", "*.avi", "*.webm", "*.mov", "*.MOV"):
            vids.extend((dataset_raw / "videos").glob(ext))
        targets = [dataset_proc / "features" / (v.stem + ".npz") for v in vids]

    print(f"[INFO] 数据集: {dataset}")
    print(f"[INFO] 目标 NPZ 数量: {len(targets)}")
    print(f"[INFO] 输出写入: {'features/ 目录(新建)' if write_to_features_dir else '覆盖/更新现有 NPZ'}")
    print(f"[INFO] Backbone={backbone}  FPS={fps}  Batch={batch_size}  Device={device}  Img={img_size}")

    for npz_path in targets:
        vid_stem = npz_path.stem
        video_path = _find_raw_video(dataset_raw, vid_stem)
        if video_path is None:
            print(f"[WARN] 找不到原始视频，跳过: {vid_stem}")
            continue

        print(f"[EXTRACT] {video_path.name} -> {npz_path.relative_to(processed_root)}")

        feats = _extract_video_feature(
            video_path, model, transform, device,
            fps=fps, batch_size=batch_size
        )

        # 记录原始总帧数
        cap = cv2.VideoCapture(str(video_path))
        n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
        if n_frames_total <= 0:
            n_frames_total = feats.shape[0]  # 兜底

        _update_npz(npz_path, feats, n_frames_total)
        print(f"[OK] feats={feats.shape}, n_frames={n_frames_total}")

    print("[DONE] 全部完成。")


def main():
    parser = argparse.ArgumentParser(description="用 ResNet 提取视频特征并写回 processed/<dataset> 的 NPZ。")
    parser.add_argument("--dataset", type=str, required=True, choices=["TVSum", "SumMe"],
                        help="要处理的数据集名称")
    parser.add_argument("--raw_root", type=str, default="data/raw",
                        help="原始数据根目录（需包含 <dataset>/videos/*.mp4）")
    parser.add_argument("--processed_root", type=str, default="data/processed",
                        help="处理后数据根目录（需已有 npz，或将写入 features/）")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        help="ResNet 骨干网络")
    parser.add_argument("--fps", type=float, default=2.0, help="采样 FPS（每秒抽几帧）")
    parser.add_argument("--batch_size", type=int, default=64, help="前向批大小")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 / cpu")
    parser.add_argument("--img_size", type=int, default=224, help="输入分辨率")
    args = parser.parse_args()

    run(dataset=args.dataset,
        raw_root=Path(args.raw_root),
        processed_root=Path(args.processed_root),
        backbone=args.backbone,
        fps=args.fps,
        batch_size=args.batch_size,
        device_str=args.device,
        img_size=args.img_size)


if __name__ == "__main__":
    main()
