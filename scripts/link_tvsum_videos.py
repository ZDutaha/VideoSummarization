# -*- coding: utf-8 -*-
# file: scripts/link_tvsum_videos.py
"""
将 data/raw/TVSum/videos/ 下的 YouTube-ID 命名视频
映射/链接为 tvsum_00.mp4 ... tvsum_49.mp4 。

优先顺序：
1) 使用 data/raw/TVSum/tvsum_index_to_ytid.json （若存在）
2) 解析 data/raw/TVSum/ 解压后的官方包（*.tsv/*.txt 中的 watch?v=ID）
3) 兜底：按文件名中 11 位 YouTube-ID 的字母序排序取前 50 个
"""

import re, json, os, sys, shutil
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path("data/raw/TVSum")
VIDEOS = ROOT / "videos"

YID_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def find_official_dir() -> Optional[Path]:
    # 在 data/raw/TVSum 下找一个包含“tvsum”子串的目录（解压目录）
    for p in sorted(ROOT.iterdir()):
        if p.is_dir() and "tvsum" in p.name.lower():
            return p
    return None


def parse_ids_from_texts(base: Path) -> List[str]:
    ids = []
    for p in list(base.rglob("*.tsv")) + list(base.rglob("*.txt")):
        try:
            s = p.read_text(errors="ignore")
        except Exception:
            continue
        # 1) watch?v=ID
        for m in re.finditer(r"watch\?v=([A-Za-z0-9_-]{11})", s):
            ids.append(m.group(1))
        # 2) 裸 ID（按列或逐行）
        for token in re.split(r"[\s\t\n\r]+", s):
            if YID_RE.fullmatch(token):
                ids.append(token)
    # 去重保序
    seen, uniq = set(), []
    for x in ids:
        if x not in seen:
            uniq.append(x);
            seen.add(x)
    return uniq


def collect_ids_from_videos_dir() -> List[str]:
    cands = []
    for p in VIDEOS.glob("*.*"):
        stem = p.stem
        if YID_RE.fullmatch(stem):
            cands.append(stem)
    return sorted(set(cands))


def load_mapping() -> Dict[str, str]:
    # 1) 显式映射 JSON
    mp = ROOT / "tvsum_index_to_ytid.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        # 只保留 tvsum_00..49
        out = {k: v for k, v in m.items() if re.fullmatch(r"tvsum_\d{2}", k)}
        if len(out) >= 50:
            return {f"tvsum_{i:02d}": out[f"tvsum_{i:02d}"] for i in range(50)}
        # 不足 50 也先返回，后面用其它来源补齐
        base = out
    else:
        base = {}

    # 2) 解析官方包
    off = find_official_dir()
    ids = []
    if off is not None:
        ids = parse_ids_from_texts(off)

    # 3) 兜底：从 videos 文件名收集 ID
    if len(ids) < 50:
        ids2 = collect_ids_from_videos_dir()
        # 合并
        seen = set(ids)
        ids.extend([x for x in ids2 if x not in seen])

    # 用 base 覆盖（如果已有）
    final = {}
    for i in range(50):
        k = f"tvsum_{i:02d}"
        if k in base:
            final[k] = base[k]
        else:
            if i < len(ids):
                final[k] = ids[i]
            else:
                raise SystemExit(f"[ERROR] 无法为 {k} 找到对应的 YouTube ID（总计ID不足50）。")

    return final


def main():
    if not VIDEOS.exists():
        raise SystemExit(f"[ERROR] 未找到目录：{VIDEOS}")

    mapping = load_mapping()

    created, copied, miss = 0, 0, 0
    for k, ytid in mapping.items():
        src = None
        for ext in ("mp4", "mkv", "webm", "mov", "MP4", "MOV", "m4v"):
            cand = VIDEOS / f"{ytid}.{ext}"
            if cand.exists():
                src = cand
                break
        if src is None:
            print("[MISS]", k, "->", ytid, "(未找到对应视频文件)")
            miss += 1
            continue

        dst = VIDEOS / f"{k}.mp4"
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            # 尝试软链接（相对路径），失败则复制
            os.symlink(src.name, dst)
            print("[LINK]", dst.name, "=>", src.name)
            created += 1
        except OSError:
            shutil.copy2(src, dst)
            print("[COPY]", dst.name, "<-", src.name)
            copied += 1

    print(f"[DONE] 链接: {created}，复制: {copied}，缺失: {miss}")


if __name__ == "__main__":
    main()
