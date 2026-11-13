"""Evaluate Basira on public forgery datasets (download them yourself; none is bundled).

    python benchmarks/public_datasets.py comofod  --root DIR [--limit N] [--variants JC]
    python benchmarks/public_datasets.py columbia --root DIR [--limit N]

CoMoFoD (small, 512x512 copy-move, Tralic et al. 2013)  https://www.vcl.fer.hr/comofod/
    NNN_O.png original, NNN_F.png forged, NNN_B.png binary mask, plus post-processed
    versions NNN_{O,F}_{JC1..9,IB1..3,NA1..3,BC1..3,CR1..3,CA1..3}.
Columbia Uncompressed Image Splicing (Hsu & Chang 2006)  https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/
    4cam_auth/*.tif authentic, 4cam_splc/*.tif spliced, edgemask/ colour-coded regions.

Reported: ROC-AUC of the manipulation score, flag rate at the default operating point
("Suspicious" or worse), and pixel-level F1 / IoU of the fused localisation map for
flagged forgeries and over all forgeries (unflagged = empty prediction).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_folder import auc  # noqa: E402

import basira  # noqa: E402


def pixel_scores(report, gt):
    heat = report.fused_heatmap
    if heat is None or report.verdict_level < 2:
        return 0.0, 0.0
    if heat.shape != gt.shape:
        heat = cv2.resize(heat, (gt.shape[1], gt.shape[0]))
    pred = heat >= 0.5
    tp = float((pred & gt).sum())
    return 2 * tp / max(pred.sum() + gt.sum(), 1), tp / max((pred | gt).sum(), 1)


def summarise(name, clean, forged, f1, iou, fired, t0):
    c, f = np.array(clean), np.array(forged)
    flagged_f1 = [x for x in f1 if x > 0]
    print(f"\n== {name}: {len(c)} authentic, {len(f)} forged  ({time.time() - t0:.0f}s)")
    print(f"ROC-AUC (manipulation score)        {auc(c[:, 1], f[:, 1]):.3f}")
    print(f"forged flagged                      {int((f[:, 0] >= 2).sum())} / {len(f)}"
          f"  ({(f[:, 0] >= 2).mean() * 100:.0f}%)")
    print(f"authentic flagged (false positives) {int((c[:, 0] >= 2).sum())} / {len(c)}"
          f"  ({(c[:, 0] >= 2).mean() * 100:.1f}%)")
    if f1:
        print(f"pixel F1 / IoU, all forged          {np.mean(f1):.3f} / {np.mean(iou):.3f}")
        if flagged_f1:
            print(f"pixel F1, flagged forged only       {np.mean(flagged_f1):.3f}")
    for split in ("authentic", "forged"):
        if fired[split]:
            print(f"detectors fired on {split}: "
                  + ", ".join(f"{k.split('.')[1]}={v}" for k, v in sorted(fired[split].items())))


def run(pairs, name):
    """pairs: list of (path, is_forged, mask or None)."""
    t0 = time.time()
    clean, forged, f1, iou = [], [], [], []
    fired = {"authentic": defaultdict(int), "forged": defaultdict(int)}
    for i, (p, is_forged, mask) in enumerate(pairs):
        try:
            r = basira.analyze(p)
        except Exception as e:
            print(f"skip {os.path.basename(p)}: {e}", file=sys.stderr)
            continue
        (forged if is_forged else clean).append((r.verdict_level, r.scores["manipulation"]))
        for e in r.evidences:
            if e.score >= 0.5 and e.confidence > 0.2 and e.category.value == "manipulation":
                fired["forged" if is_forged else "authentic"][e.detector] += 1
        if is_forged and mask is not None:
            a, b = pixel_scores(r, mask)
            f1.append(a)
            iou.append(b)
        if (i + 1) % 50 == 0:
            print(f"  {name}: {i + 1}/{len(pairs)}", file=sys.stderr)
    if clean and forged:
        summarise(name, clean, forged, f1, iou, fired, t0)
    return clean, forged


def comofod(root, limit, variants):
    base = sorted(glob.glob(os.path.join(root, "**", "*_F.png"), recursive=True))
    if limit:
        base = base[:limit]
    if not base:
        sys.exit(f"no *_F.png under {root}")

    def pairs_for(suffix, ext):
        out = []
        for f in base:
            stem = f[:-len("_F.png")]
            mask = cv2.imread(stem + "_B.png", cv2.IMREAD_GRAYSCALE)
            fp, op = f"{stem}_F{suffix}.{ext}", f"{stem}_O{suffix}.{ext}"
            if os.path.exists(fp) and os.path.exists(op) and mask is not None:
                out.append((op, False, None))
                out.append((fp, True, mask > 127))
        return out

    run(pairs_for("", "png"), "CoMoFoD (no post-processing)")
    for v in variants:
        for k in range(1, 10):
            ps = pairs_for(f"_{v}{k}", "jpg") or pairs_for(f"_{v}{k}", "png")
            if ps:
                run(ps, f"CoMoFoD {v}{k}")


def columbia(root, limit):
    auth = sorted(glob.glob(os.path.join(root, "**", "4cam_auth", "*.tif"), recursive=True))
    splc = sorted(glob.glob(os.path.join(root, "**", "4cam_splc", "*.tif"), recursive=True))
    if limit:
        auth, splc = auth[:limit], splc[:limit]
    if not auth or not splc:
        sys.exit(f"no 4cam_auth / 4cam_splc under {root}")
    pairs = [(p, False, None) for p in auth]
    for p in splc:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = glob.glob(os.path.join(os.path.dirname(p), "edgemask", stem + "_edgemask.*"))
        mask = None
        if m:
            bgr = cv2.imread(m[0])
            if bgr is not None:
                # Red and green mark the two source regions (two cameras). Which of them is
                # "the forgery" is not defined, so the smaller one is used, as most papers do.
                red = bgr[..., 2] > 150
                green = (bgr[..., 1] > 150) & ~red
                mask = red if 0 < red.sum() <= green.sum() or green.sum() == 0 else green
        pairs.append((p, True, mask))
    run(pairs, "Columbia Uncompressed Splicing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dataset", choices=["comofod", "columbia"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=[], help="CoMoFoD post-processing codes, e.g. JC IB NA")
    a = ap.parse_args()
    if a.dataset == "comofod":
        comofod(a.root, a.limit, a.variants)
    else:
        columbia(a.root, a.limit)
