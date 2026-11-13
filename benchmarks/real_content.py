"""Benchmark on REAL image content that you supply.

    python benchmarks/real_content.py --sources DIR [--videos V1.mov ...] [--n 30] [--robust]

Takes real photographs (and/or frames of real videos), cuts them into 1024x768
"photos", simulates a camera save, and builds forgeries with ground truth:

    clean      one JPEG save                  resaved    two JPEG saves
    splice     region from another photo      copymove   region cloned inside the photo
    inpaint    object removal (OpenCV Telea in-painting)

With --robust, every forgery is additionally post-processed the way platforms do
(re-compression at lower quality, downscaling) to show how fast the traces fade.

Nothing is downloaded and nothing is written outside the output directory.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import zlib
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import basira  # noqa: E402

W, H = 1024, 768


def textured(c, min_activity=3.0):
    """Skip featureless crops (blank sky, fog, water): an edit there leaves nothing to
    measure and nothing to see, so it would only dilute the numbers."""
    g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean()) >= min_activity


def load_crops(sources, videos, n, seed=0):
    rng = np.random.RandomState(seed)
    crops = []
    files = []
    if sources:
        files = sorted(os.path.join(sources, f) for f in os.listdir(sources)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")))
    for f in files:
        img = np.asarray(Image.open(f).convert("RGB"))
        h, w = img.shape[:2]
        for y in range(0, h - H + 1, H):
            for x in range(0, w - W + 1, W):
                c = img[y:y + H, x:x + W]
                if textured(c):
                    crops.append(c.copy())
    for v in videos or []:
        cap = cv2.VideoCapture(v)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for pos in np.linspace(0.05, 0.9, 8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pos))
            ok, bgr = cap.read()
            if not ok:
                continue
            fr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = fr.shape[:2]
            for y in range(0, h - H + 1, H):
                for x in range(0, w - W + 1, W):
                    if textured(fr[y:y + H, x:x + W]):
                        crops.append(fr[y:y + H, x:x + W].copy())
        cap.release()
    idx = rng.permutation(len(crops))[:n]
    return [crops[i] for i in idx]


def jpeg(rgb, q):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=int(q))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def save(rgb, path, q):
    Image.fromarray(rgb).save(path, "JPEG", quality=int(q))
    return path


def box(rng, lo=140, hi=260):
    s = int(rng.randint(lo, hi))
    return int(rng.randint(16, H - s - 16)), int(rng.randint(16, W - s - 16)), s


def make(kind, i, crops, out, rng):
    host = crops[i]
    donor = crops[(i + 1 + rng.randint(len(crops) - 1)) % len(crops)]
    mask = np.zeros((H, W), bool)
    q1 = int(rng.choice([72, 80, 86]))
    q2 = int(rng.choice([88, 92, 95]))
    p = os.path.join(out, f"{kind}_{i}.jpg")
    if kind == "clean":
        return save(host, p, int(rng.choice([75, 82, 90, 95]))), mask
    first = jpeg(host, q1).copy()                      # the "camera original"
    if kind == "resaved":
        return save(first, p, q2), mask
    y, x, s = box(rng)
    if kind == "splice":
        d = jpeg(donor, int(rng.choice([85, 95])))
        dy, dx = int(rng.randint(0, H - s)), int(rng.randint(0, W - s))
        first[y:y + s, x:x + s] = d[dy:dy + s, dx:dx + s]
        mask[y:y + s, x:x + s] = True
    elif kind == "copymove":
        for _ in range(200):
            ty, tx = int(rng.randint(16, H - s - 16)), int(rng.randint(16, W - s - 16))
            if abs(ty - y) > s or abs(tx - x) > s:
                break
        first[ty:ty + s, tx:tx + s] = first[y:y + s, x:x + s]
        mask[ty:ty + s, tx:tx + s] = True
        mask[y:y + s, x:x + s] = True
    elif kind == "inpaint":
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (x + s // 2, y + s // 2), (s // 2, s // 3), 0, 0, 360, 255, -1)
        first = cv2.inpaint(first, m, 5, cv2.INPAINT_TELEA)
        mask = m > 0
    return save(first, p, q2), mask


def contrast(heat, mask):
    if heat is None or not mask.any() or mask.all():
        return float("nan")
    if heat.shape != mask.shape:
        heat = cv2.resize(heat, (mask.shape[1], mask.shape[0]))
    return float(heat[mask].mean() / (heat[~mask].mean() + 1e-3))


def postprocess(path, mode, out):
    img = np.asarray(Image.open(path).convert("RGB"))
    name = os.path.join(out, os.path.basename(path).replace(".jpg", f"_{mode}.jpg"))
    if mode.startswith("q"):
        return save(img, name, int(mode[1:]))
    f = float(mode[1:]) / 100
    small = cv2.resize(img, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
    return save(small, name, 90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sources", help="folder of real photographs")
    ap.add_argument("--videos", nargs="*", help="real videos to sample frames from")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out")
    ap.add_argument("--robust", action="store_true")
    a = ap.parse_args()
    out = a.out or tempfile.mkdtemp(prefix="basira_real_")
    os.makedirs(out, exist_ok=True)
    crops = load_crops(a.sources, a.videos, a.n)
    if len(crops) < 4:
        sys.exit("need at least 4 usable 1024x768 crops; give --sources and/or --videos")
    print(f"{len(crops)} real crops")
    kinds = ("clean", "resaved", "splice", "copymove", "inpaint")
    flagged = defaultdict(list)
    loc = defaultdict(list)
    fired = defaultdict(lambda: defaultdict(int))
    paths = defaultdict(list)
    for kind in kinds:
        for i in range(len(crops)):
            rng = np.random.RandomState(zlib.crc32(f"{kind}-{i}".encode()))
            p, mask = make(kind, i, crops, out, rng)
            r = basira.analyze(p)
            flagged[kind].append(r.verdict_level >= 2)
            loc[kind].append(contrast(r.fused_heatmap, mask))
            paths[kind].append((p, mask))
            for e in r.evidences:
                if e.score >= 0.5 and e.confidence > 0.2 and e.category.value == "manipulation":
                    fired[kind][e.detector] += 1
    print(f"\n{'family':10s} {'flagged':>10s} {'located':>10s}   detectors that fired")
    for kind in kinds:
        n = len(flagged[kind])
        c = np.array(loc[kind])
        located = int(np.sum((np.array(flagged[kind])) & (c > 3))) if kind not in ("clean", "resaved") else 0
        print(f"{kind:10s} {sum(flagged[kind]):4d} / {n:<3d} "
              + (f"{located:4d} / {n:<3d}" if kind not in ("clean", "resaved") else f"{'-':>10s}")
              + "   " + ", ".join(f"{k.split('.')[1]}={v}" for k, v in sorted(fired[kind].items())))
    if a.robust:
        modes = ("q85", "q70", "q50", "r75", "r50")
        print(f"\nflagged after post-processing   {'':6s}" + "".join(f"{m:>8s}" for m in modes))
        for kind in ("clean", "splice", "copymove", "inpaint"):
            row = []
            for m in modes:
                hits = 0
                for p, _ in paths[kind]:
                    hits += basira.analyze(postprocess(p, m, out)).verdict_level >= 2
                row.append(f"{hits}/{len(paths[kind])}")
            print(f"{kind:32s}" + "".join(f"{x:>8s}" for x in row))
        print("(qNN = re-saved as JPEG quality NN; rNN = downscaled to NN% then saved at Q90)")


if __name__ == "__main__":
    main()
