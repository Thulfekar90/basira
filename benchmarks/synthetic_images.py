"""Synthetic image benchmark: per-detector statistics on clean vs forged images.

    python benchmarks/synthetic_images.py [--n 12] [--out DIR]

Prints, for every detector, the score distribution on clean images (false-positive
rate at the 0.5 threshold) and on each forgery family (detection rate, localisation).
The data is synthetic (see tests/synth_image.py): it validates the implementation
and calibrates thresholds; it is NOT a substitute for evaluation on real datasets.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import synth_image as S  # noqa: E402

import basira  # noqa: E402

FAMILIES = {
    "authentic": S.make_authentic, "resaved": S.make_resaved, "splice": S.make_splice,
    "copymove": S.make_copymove, "inpaint": S.make_inpaint, "gridshift": S.make_grid_shift,
    "generated": S.make_generated,
}
CLEAN = ("authentic", "resaved")


def run(n: int, out: str, verbose: bool = False):
    rows = defaultdict(lambda: defaultdict(list))   # family -> detector -> [(score, stat, contrast)]
    verdicts = defaultdict(list)
    for fam, fn in FAMILIES.items():
        for seed in range(n):
            path, mask = fn(out, seed)
            r = basira.analyze(path)
            verdicts[fam].append((r.verdict_level, r.scores["manipulation"], r.scores["ai_generation"]))
            loc = S.hit_ratio(r.fused_heatmap, mask) if (mask.any() and not mask.all()) else float("nan")
            rows[fam]["FUSED"].append((r.scores["overall"], r.scores["overall"], loc))
            for e in r.evidences:
                c = S.hit_ratio(e.heatmap, mask) if (e.heatmap is not None and mask.any()
                                                     and not mask.all()) else float("nan")
                st = e.details.get("statistic", e.details.get("lattice_excess_db", float("nan")))
                rows[fam][e.detector].append((e.score, st, c))
                if verbose and e.error:
                    print("ERR", fam, seed, e.detector, e.error)
    dets = sorted({d for fam in rows.values() for d in fam})
    print(f"\n{'detector':26s}" + "".join(f"{f:>12s}" for f in FAMILIES))
    print("fraction of images with score >= 0.5  (clean families = false-positive rate)")
    for d in dets:
        line = f"{d:26s}"
        for fam in FAMILIES:
            v = [s for s, _, _ in rows[fam].get(d, [])]
            line += f"{(np.mean(np.array(v) >= 0.5) if v else float('nan')):12.2f}"
        print(line)
    print("\nmedian raw statistic")
    for d in dets:
        line = f"{d:26s}"
        for fam in FAMILIES:
            v = [st for _, st, _ in rows[fam].get(d, []) if st == st]
            line += f"{(np.median(v) if v else float('nan')):12.3f}"
        print(line)
    print("\nmax raw statistic on clean / min on forged (for threshold setting)")
    for d in dets:
        cl = [st for f in CLEAN for _, st, _ in rows[f].get(d, []) if st == st]
        if cl:
            print(f"{d:26s} clean max={np.max(cl):.3f} p90={np.percentile(cl, 90):.3f}")
    print("\nmedian localisation contrast (heat inside true mask / outside; >3 is useful)")
    for d in dets:
        line = f"{d:26s}"
        for fam in FAMILIES:
            v = [c for _, _, c in rows[fam].get(d, []) if c == c]
            line += f"{(np.median(v) if v else float('nan')):12.1f}"
        print(line)
    print("\nverdict level counts (0 none, 1 inconclusive, 2 suspicious, 3 likely, 4 decisive)")
    for fam in FAMILIES:
        lv = [v[0] for v in verdicts[fam]]
        print(f"{fam:12s}", {k: lv.count(k) for k in range(5) if lv.count(k)})
    return rows, verdicts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    out = a.out or tempfile.mkdtemp(prefix="basira_bench_")
    os.makedirs(out, exist_ok=True)
    run(a.n, out, a.v)
