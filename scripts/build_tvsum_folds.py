import argparse
import glob
import numpy as np
import os
import pathlib
import random
import re
import shutil


def natural_key(p):
    # tvsum_12.npz -> 12
    m = re.search(r'(\d+)', os.path.basename(p))
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/processed/TVSum', help='processed 根目录')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    assert root.exists(), f'{root} not found'

    # 收集所有 npz（train/val/test 都纳入，确保 50 个视频）
    all_npz = []
    for sub in ['train', 'val', 'test']:
        d = root / sub
        all_npz += glob.glob(str(d / '*.npz'))
    all_npz = sorted(set(all_npz), key=natural_key)
    print(f'[INFO] found npz = {len(all_npz)}')
    assert len(all_npz) == 50, f'期望 50 个 TVSum 视频，当前 {len(all_npz)}'

    # 固定划分：按编号排序后均分 5 份（每折 10 个 test）
    folds = [all_npz[i::5] for i in range(5)]  # 间隔采样，避免同类聚在一起
    folds_dir = root / 'folds'
    if folds_dir.exists():
        shutil.rmtree(folds_dir)
    folds_dir.mkdir(parents=True, exist_ok=True)

    for k in range(5):
        fold_k = folds[k]
        test_set = set(fold_k)
        train_set = [p for p in all_npz if p not in test_set]

        out_train = folds_dir / f'fold_{k + 1}' / 'train'
        out_test = folds_dir / f'fold_{k + 1}' / 'test'
        out_train.mkdir(parents=True, exist_ok=True)
        out_test.mkdir(parents=True, exist_ok=True)

        for src in train_set:
            dst = out_train / os.path.basename(src)
            try:
                os.link(src, dst)  # 硬链接（更快更省空间）
            except OSError:
                shutil.copy2(src, dst)
        for src in test_set:
            dst = out_test / os.path.basename(src)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        print(f'[FOLD {k + 1}] train={len(train_set)} test={len(test_set)} -> {out_train} / {out_test}')


if __name__ == '__main__':
    main()
