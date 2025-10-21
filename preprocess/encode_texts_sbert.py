import os, numpy as np
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from preprocess.srt_utils import parse_srt, align_frames_to_subs
from selection.segment_kts import kts_change_points, segments_from_cps


def read_lines_if_exists(p: Path) -> List[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] if p.exists() else []


def auto_generate_captions(feats: np.ndarray, max_ncp: int = 10):
    T, D = feats.shape
    cps = kts_change_points(feats.astype(np.float64), max_ncp=max_ncp, kernel="linear")
    segs = segments_from_cps(T, cps)
    texts = []
    frame2sent = np.zeros(T, dtype=np.int64)
    dif = np.vstack([np.zeros((1, D)), np.abs(feats[1:] - feats[:-1])]).sum(1)
    norms = np.linalg.norm(feats, axis=1)
    for i, (s, e) in enumerate(segs):
        seg_len = e - s
        motion = float(dif[s:e].mean())
        energy = float(norms[s:e].mean())
        keywords = []
        if motion > np.percentile(dif, 70):
            keywords.append("fast motion")
        else:
            keywords.append("steady scene")
        if energy > np.percentile(norms, 50):
            keywords.append("salient content")
        else:
            keywords.append("background")
        texts.append(f"Segment {i}: {', '.join(keywords)} length {seg_len}")
        frame2sent[s:e] = i
    return texts, frame2sent


def process_dataset(data_root="data/processed", dataset="SumMe", splits=("train", "val", "test"),
                    texts_root="data/texts", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    for split in splits:
        d = Path(data_root) / dataset / split
        if not d.exists(): print("[WARN] missing", d); continue
        for npz in sorted(d.glob("*.npz")):
            Z = np.load(npz, allow_pickle=True)
            feats = Z["feats"].astype("float32")
            labels = Z["labels"].astype("float32")
            frame2sent = Z["frame2sent"].astype("int64")
            stem = npz.stem
            tdir = Path(texts_root) / dataset / split
            srtp = (tdir / f"{stem}.srt");
            txtp = (tdir / f"{stem}.txt")
            sentences = []
            new_f2s = None
            if srtp.exists():
                try:
                    subs = parse_srt(srtp.read_text(encoding="utf-8"))
                    fps = 2.0
                    new_f2s = np.array(align_frames_to_subs(len(feats), fps, subs), dtype=np.int64)
                    sentences = [s.text for s in subs]
                except Exception as e:
                    print("[WARN] srt parse failed", srtp, e)
            if not sentences and txtp.exists():
                sentences = read_lines_if_exists(txtp)
                if len(sentences) > 0:
                    new_f2s = np.linspace(0, len(sentences) - 1, len(feats)).round().astype("int64")
            if not sentences:
                sentences, new_f2s = auto_generate_captions(feats, max_ncp=10)
            emb = model.encode(sentences, convert_to_numpy=True, batch_size=64, normalize_embeddings=True).astype(
                "float32")
            if emb.ndim == 1: emb = emb[None, :]
            np.savez_compressed(npz, feats=feats, labels=labels,
                                frame2sent=new_f2s if new_f2s is not None else frame2sent,
                                texts=emb)
            print("[OK]", dataset, split, npz.name, emb.shape)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/processed")
    ap.add_argument("--dataset", type=str, default="SumMe")
    ap.add_argument("--texts_root", type=str, default="data/texts")
    ap.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    process_dataset(args.data_root, args.dataset, texts_root=args.texts_root, model_name=args.model_name)
