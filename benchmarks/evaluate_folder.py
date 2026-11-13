"""Evaluate Basira on YOUR data (e.g. CASIA v2, Columbia, COVERAGE, IMD2020, ASVspoof, FakeAVCeleb).

    python benchmarks/evaluate_folder.py --clean path/to/authentic --forged path/to/tampered \
           [--masks path/to/masks] [--score manipulation|ai_generation|overall] [--limit 500]

Reports ROC-AUC, the true/false-positive rate at the built-in 0.5 operating point and, when
ground-truth masks are given (same file stem, white = edited), pixel-level F1 and IoU of
the fused localisation map. No dataset is bundled or downloaded: bring your own.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import basira  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".wav", ".flac", ".mp3",
       ".ogg", ".m4a", ".mp4", ".mov", ".avi", ".mkv", ".webm"}


def files(d, limit):
    out = sorted(os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs
                 if os.path.splitext(f)[1].lower() in EXT)
    return out[:limit] if limit else out


def auc(neg, pos):
    """Mann-Whitney ROC-AUC (ties count half)."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if not len(neg) or not len(pos):
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    v = np.concatenate([neg, pos])[order]
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def find_mask(mask_dir, path):
    stem = os.path.splitext(os.path.basename(path))[0]
    for suffix in ("", "_gt", "_mask", "_forged"):
        for ext in (".png", ".jpg", ".bmp", ".tif"):
            p = os.path.join(mask_dir, stem + suffix + ext)
            if os.path.exists(p):
                return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--forged", required=True)
    ap.add_argument("--masks")
    ap.add_argument("--score", default="overall", choices=["manipulation", "ai_generation", "overall"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    res = {}
    f1s, ious = [], []
    for label, d in (("clean", a.clean), ("forged", a.forged)):
        res[label] = []
        for i, p in enumerate(files(d, a.limit)):
            try:
                r = basira.analyze(p)
            except Exception as e:
                print(f"skip {p}: {e}", file=sys.stderr)
                continue
            res[label].append(r.scores[a.score])
            if label == "forged" and a.masks and r.media_type == "image":
                mp = find_mask(a.masks, p)
                if mp:
                    import cv2

                    gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127
                    heat = r.fused_heatmap if r.fused_heatmap is not None else np.zeros(gt.shape, np.float32)
                    if heat.shape != gt.shape:
                        heat = cv2.resize(heat, (gt.shape[1], gt.shape[0]))
                    pred = heat >= 0.5
                    tp = (pred & gt).sum()
                    f1s.append(2 * tp / max(pred.sum() + gt.sum(), 1))
                    ious.append(tp / max((pred | gt).sum(), 1))
            if (i + 1) % 25 == 0:
                print(f"  {label}: {i + 1} files", file=sys.stderr)
    neg, pos = np.array(res["clean"]), np.array(res["forged"])
    print(f"files: {len(neg)} clean, {len(pos)} forged   score: {a.score}")
    print(f"ROC-AUC                    {auc(neg, pos):.3f}")
    print(f"detection rate  (>=0.5)    {np.mean(pos >= 0.5) if len(pos) else float('nan'):.3f}")
    print(f"false positives (>=0.5)    {np.mean(neg >= 0.5) if len(neg) else float('nan'):.3f}")
    if f1s:
        print(f"pixel F1 / IoU of fused map (n={len(f1s)})  {np.mean(f1s):.3f} / {np.mean(ious):.3f}")


if __name__ == "__main__":
    main()
