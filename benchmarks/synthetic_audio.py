"""Synthetic audio benchmark: clean recordings vs three edit types at random positions.

    python benchmarks/synthetic_audio.py [--n 10]

Each recording is 30 s of speech-like signal + room noise + weak 50 Hz mains hum
(tests/synth_audio.py). An edit counts as located when a reported segment lies within
0.5 s of the true edit time. Synthetic data: validates the implementation only.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import synth_audio as A  # noqa: E402

import basira  # noqa: E402


def near(segs, t, tol=0.5):
    return any(s.start - tol <= t <= s.end + tol for s in segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    out = tempfile.mkdtemp(prefix="basira_audio_")
    sr = A.SR
    tally = {k: [0, 0] for k in ("clean flagged", "cut located", "insert located", "zeros located")}
    for seed in range(a.n):
        rng = np.random.default_rng(100 + seed)
        x = A.recording(30.0, sr, seed=seed)
        cases = {"clean": (x, None)}
        at = float(rng.uniform(6, 24))
        y, t = A.forge_cut(x, sr, at=at, remove=float(rng.uniform(0.4, 1.2)))
        cases["cut"] = (y, t)
        at = float(rng.uniform(6, 22))
        y, span = A.forge_insert(x, sr, at=at, seg=A.foreign_segment(3.0, sr, seed=50 + seed))
        cases["insert"] = (y, span[0])
        at = float(rng.uniform(6, 24))
        y, t = A.forge_zeros(x, sr, at=at, dur=0.3)
        cases["zeros"] = (y, t)
        for name, (sig, truth) in cases.items():
            p = os.path.join(out, f"{name}_{seed}.wav")
            sf.write(p, A.quantize16(sig), sr, subtype="PCM_16")
            r = basira.analyze(p)
            if name == "clean":
                tally["clean flagged"][0] += int(r.verdict_level >= 2)
                tally["clean flagged"][1] += 1
            else:
                key = f"{name} located"
                tally[key][0] += int(r.verdict_level >= 2 and near(r.fused_segments, truth))
                tally[key][1] += 1
    for k, (hit, n) in tally.items():
        print(f"{k:16s} {hit} / {n}")


if __name__ == "__main__":
    main()
