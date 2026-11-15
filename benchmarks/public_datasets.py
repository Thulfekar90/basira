"""Evaluate Basira on public forgery datasets (download them yourself; none is bundled).

    python benchmarks/public_datasets.py comofod  --root DIR [--limit N] [--variants JC]
    python benchmarks/public_datasets.py columbia --root DIR [--limit N]
    python benchmarks/public_datasets.py casia1   --root DIR [--limit N] [--csv OUT]
    python benchmarks/public_datasets.py casia2   --root DIR [--half dev|test] [--csv OUT]
    python benchmarks/public_datasets.py coverage --root DIR [--half dev|test] [--csv OUT]

CoMoFoD (small, 512x512 copy-move, Tralic et al. 2013)  https://www.vcl.fer.hr/comofod/
    NNN_O.png original, NNN_F.png forged, NNN_B.png binary mask, plus post-processed
    versions NNN_{O,F}_{JC1..9,IB1..3,NA1..3,BC1..3,CR1..3,CA1..3}.
Columbia Uncompressed Image Splicing (Hsu & Chang 2006)  https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/
    4cam_auth/*.tif authentic, 4cam_splc/*.tif spliced, edgemask/ colour-coded regions.

Datasets first evaluated out-of-sample (see README for later tuning/exposure):
CASIA v1.0 (Dong et al. 2013), masks by Pham et al.  https://github.com/namtpham/casia1groundtruth
    Au/*.jpg authentic, Tp/Sp/*.jpg spliced, Tp/CM/*.jpg copy-move; <stem>_gt.png masks.
CASIA v2.0, corrected masks  https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth
    Au/ authentic, Tp/ tampered (Tp_D_* spliced, Tp_S_* copy-move), Gt/<stem>_gt.png.
COVERAGE (Wen et al. 2016)  https://github.com/wenbihan/coverage
    image/NN.tif original with similar-but-genuine objects, image/NNt.tif copy-move forgery,
    mask/NNforged.tif pasted region.
    --half dev selects odd target IDs; test selects even IDs. Both were previously seen
    in aggregate, so the even split is internal validation, not pristine out-of-sample.

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


def summarise(name, clean, forged, f1, iou, fired, t0, groups=None):
    c, f = np.array(clean), np.array(forged)
    if len(f1) != len(forged) or len(iou) != len(forged):
        raise ValueError("Localisation scores must align with every analysed forgery")
    flagged_f1 = [x for x, (level, _) in zip(f1, forged) if level >= 2]
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
    for g, levels in sorted((groups or {}).items()):
        lv = np.array(levels)
        print(f"  forged flagged, {g:<22} {int((lv >= 2).sum())} / {len(lv)}  ({(lv >= 2).mean() * 100:.0f}%)")
    for split in ("authentic", "forged"):
        if fired[split]:
            print(f"detectors fired on {split}: "
                  + ", ".join(f"{k.split('.')[1]}={v}" for k, v in sorted(fired[split].items())))


def run(pairs, name, csv_path=None):
    """pairs: list of (path, is_forged, mask or None[, group]).

    mask may be a callable returning the mask, so large sets are not held in memory.
    """
    t0 = time.time()
    clean, forged, f1, iou = [], [], [], []
    groups = defaultdict(list)
    fired = {"authentic": defaultdict(int), "forged": defaultdict(int)}
    out = open(csv_path, "w", encoding="utf-8") if csv_path else None
    if out:
        out.write("file,forged,group,level,manipulation,ai,fired,pixel_f1\n")
    skipped = missing_masks = 0
    for i, item in enumerate(pairs):
        p, is_forged, mask = item[:3]
        group = item[3] if len(item) > 3 else ""
        try:
            r = basira.analyze(p)
        except Exception as e:
            skipped += 1
            print(f"skip {os.path.basename(p)}: {e}", file=sys.stderr)
            continue
        (forged if is_forged else clean).append((r.verdict_level, r.scores["manipulation"]))
        if is_forged and group:
            groups[group].append(r.verdict_level)
        hits = sorted(e.detector.split(".")[-1] for e in r.evidences
                      if e.score >= 0.5 and e.confidence > 0.2)
        for e in r.evidences:
            if e.score >= 0.5 and e.confidence > 0.2 and e.category.value == "manipulation":
                fired["forged" if is_forged else "authentic"][e.detector] += 1
        if callable(mask):
            mask = mask()
        a = None
        if is_forged:
            # Keep scores aligned with the forged verdicts, including complete misses.
            # No mask is not evidence of accurate localisation: count it conservatively
            # as zero and disclose the count rather than silently shrinking the sample.
            if mask is None:
                a, b = 0.0, 0.0
                missing_masks += 1
            else:
                a, b = pixel_scores(r, mask)
            f1.append(a)
            iou.append(b)
        if out:
            out.write(f"{os.path.basename(p)},{int(is_forged)},{group},{r.verdict_level},"
                      f"{r.scores['manipulation']:.4f},{r.scores.get('ai_generation', 0):.4f},"
                      f"{'+'.join(hits)},{'' if a is None else f'{a:.4f}'}\n")
        if (i + 1) % 100 == 0:
            print(f"  {name}: {i + 1}/{len(pairs)}", file=sys.stderr)
            if out:
                out.flush()
    if out:
        out.close()
    if skipped:
        print(f"\n{skipped} file(s) could not be analysed (listed on stderr)")
    if missing_masks:
        print(f"{missing_masks} forged file(s) have no usable mask; localisation scored as zero")
    if clean and forged:
        summarise(name, clean, forged, f1, iou, fired, t0, groups)
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


IMAGE_EXT = (".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".png")


def _images(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "*"))
                  if p.lower().endswith(IMAGE_EXT))


def _find_dir(root, *names):
    for name in names:
        hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
        hits = [h for h in hits if os.path.isdir(h)]
        if hits:
            return hits[0]
    return None


def _mask_loader(path):
    def load():
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if path else None
        return None if m is None else m > 127
    return load


def _spread(items, limit):
    """An evenly spaced subset, so a limit does not keep only the first category."""
    if not limit or len(items) <= limit:
        return items
    idx = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[i] for i in idx]


def casia2_half(path):
    """'dev' or 'test' for a CASIA v2 file. The key is the target photo (Au_ani_00001 ->
    ani00001; Tp_D_CND_M_N_ani00018_sec00096_00138 -> sec00096), so a photo and every
    forgery built on it land in the same half. Tune on dev only."""
    import hashlib
    import re
    name = os.path.basename(path)
    m = re.match(r"Au_([a-z]+)_(\d+)", name)
    if m:
        key = m.group(1) + m.group(2)
    else:
        parts = os.path.splitext(name)[0].split("_")
        key = parts[6] if len(parts) >= 8 else name
    return "dev" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 2 == 0 else "test"


def casia(root, limit, version, csv_path, half=None):
    au = _find_dir(root, "Au")
    if not au:
        sys.exit(f"no Au/ folder under {root}")
    auth = _images(au)
    gt_index = {}
    for g in glob.glob(os.path.join(root, "**", "*_gt.png"), recursive=True):
        gt_index[os.path.basename(g)[:-len("_gt.png")]] = g
    if version == 1:
        tp = []
        for sub, group in (("Sp", "splicing"), ("CM", "copy-move")):
            d = _find_dir(os.path.join(root, "raw"), os.path.join("Tp", sub)) or _find_dir(root, os.path.join("Tp", sub))
            tp += [(p, group) for p in _images(d)] if d else []
    else:
        d = _find_dir(root, "Tp")
        tp = [(p, "copy-move" if os.path.basename(p).startswith("Tp_S_") else "splicing")
              for p in (_images(d) if d else [])]
    if not tp:
        sys.exit(f"no tampered images under {root}")
    if half:
        auth = [p for p in auth if casia2_half(p) == half]
        tp = [(p, g) for p, g in tp if casia2_half(p) == half]
    tp, auth = _spread(tp, limit), _spread(auth, limit)
    pairs = [(p, False, None) for p in auth]
    missing = 0
    for p, group in tp:
        g = gt_index.get(os.path.splitext(os.path.basename(p))[0])
        missing += g is None
        pairs.append((p, True, _mask_loader(g), group))
    if missing:
        print(f"{missing} tampered image(s) have no mask; they count as forged, F1 = 0")
    run(pairs, f"CASIA v{version}.0" + (f" ({half} half)" if half else ""), csv_path)


def coverage(root, limit, csv_path, half=None):
    """Odd target IDs are development; even IDs are previously seen internal validation.

    Original, forgery and mask share the numeric target ID. ``limit`` counts target
    IDs so selecting a small diagnostic subset cannot separate an original/forgery pair.
    """
    if half not in (None, "dev", "test"):
        raise ValueError("COVERAGE half must be dev or test")
    img = _find_dir(root, "image")
    mask = _find_dir(root, "mask")
    if not img or not mask:
        sys.exit(f"no image/ and mask/ folders under {root}")
    targets = defaultdict(list)
    for p in _images(img):
        stem = os.path.splitext(os.path.basename(p))[0]
        target = stem[:-1] if stem.endswith("t") else stem
        if not target.isascii() or not target.isdigit():
            raise ValueError(f"Invalid COVERAGE target id: {stem}")
        target_id = int(target)
        if half and ("dev" if target_id % 2 else "test") != half:
            continue
        if stem.endswith("t"):
            m = os.path.join(mask, stem[:-1] + "forged.tif")
            targets[target_id].append((p, True, _mask_loader(m if os.path.exists(m) else None), "copy-move"))
        else:
            targets[target_id].append((p, False, None))
    if not targets:
        sys.exit("no COVERAGE images in requested split")
    pairs = [p for target_id in _spread(sorted(targets), limit) for p in targets[target_id]]
    label = "COVERAGE" + (f" ({half}: {'odd' if half == 'dev' else 'even'} target IDs; previously seen)" if half else "")
    run(pairs, label, csv_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dataset", choices=["comofod", "columbia", "casia1", "casia2", "coverage"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="limit samples (COVERAGE: target IDs, preserving original/forgery pairs)")
    ap.add_argument("--variants", nargs="*", default=[], help="CoMoFoD post-processing codes, e.g. JC IB NA")
    ap.add_argument("--csv", help="write one row per image (casia1, casia2, coverage)")
    ap.add_argument("--half", choices=["dev", "test"],
                    help="casia2: target hash split; coverage: odd dev / even test (previously seen)")
    a = ap.parse_args(argv)
    if a.half and a.dataset not in ("casia2", "coverage"):
        ap.error("--half applies to casia2 or coverage only")
    if a.variants and a.dataset != "comofod":
        ap.error("--variants applies to comofod only")
    if a.csv and a.dataset not in ("casia1", "casia2", "coverage"):
        ap.error("--csv applies to casia1, casia2 or coverage only")
    if a.limit < 0:
        ap.error("--limit must be nonnegative")
    print(f"basira {basira.__version__}")
    if a.dataset == "comofod":
        comofod(a.root, a.limit, a.variants)
    elif a.dataset == "columbia":
        columbia(a.root, a.limit)
    elif a.dataset == "coverage":
        coverage(a.root, a.limit, a.csv, a.half)
    else:
        casia(a.root, a.limit, int(a.dataset[-1]), a.csv, a.half)
    return 0


if __name__ == "__main__":
    sys.exit(main())
