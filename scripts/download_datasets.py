#!/usr/bin/env python3
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

SUMME_CANDIDATES = [
    "https://people.ee.ethz.ch/~gyglim/SHOT/SumMe.zip",
    "https://people.ee.ethz.ch/~gyglim/downloads/SumMe/SumMe.zip",
]
TVSUM_CANDIDATES = [
    "http://yalesong.github.io/tvsum/_data/ydata-tvsum50-v1_1.tgz",
    "https://github.com/yalesong/tvsum/releases/download/v1.1/ydata-tvsum50-v1_1.tgz",
]


def _download(url, out_path):
    try:
        print(f"[DL] {url}")
        with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
            shutil.copyfileobj(r, f)
        print(f"[OK] saved -> {out_path}")
        return True
    except Exception as e:
        print(f"[WARN] fail {url}: {e}")
        return False


def ensure_summe(root="data/raw/SumMe"):
    root = Path(root);
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "SumMe.zip"
    if not zip_path.exists():
        for url in SUMME_CANDIDATES:
            if _download(url, zip_path): break
    if not zip_path.exists():
        print("[ERR] SumMe download failed. Please download manually to", zip_path);
        return False
    assert zipfile.is_zipfile(zip_path), f"{zip_path} is not a valid zip";
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(root)
    print("[OK] SumMe extracted ->", root);
    return True


def ensure_tvsum(root="data/raw/TVSum"):
    root = Path(root);
    root.mkdir(parents=True, exist_ok=True)
    tgz = root / "ydata-tvsum50-v1_1.tgz"
    if not tgz.exists():
        for url in TVSUM_CANDIDATES:
            if _download(url, tgz): break
    if not tgz.exists():
        print("[ERR] TVSum download failed. Please download manually to", tgz);
        return False
    with tarfile.open(tgz, "r:gz") as t:
        t.extractall(root)
    print("[OK] TVSum extracted ->", root);
    return True


if __name__ == "__main__":
    ok1 = ensure_summe();
    ok2 = ensure_tvsum();
    sys.exit(0 if (ok1 and ok2) else 1)
