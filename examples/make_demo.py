"""Generate synthetic forgeries with ground truth, analyse them, and write figures + reports.

    python examples/make_demo.py [out_dir]

Everything is generated locally (no downloads). Figures land in docs/img/.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import synth_image as S  # noqa: E402

import basira  # noqa: E402
from basira.core.report_html import save_html  # noqa: E402
from basira.core.utils import overlay  # noqa: E402


def panel(rgb, title):
    out = rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def figure(path, mask, name, img_dir):
    r = basira.analyze(path)
    rgb = np.asarray(Image.open(path).convert("RGB"))
    truth = rgb.copy()
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(truth, cs, -1, (255, 40, 40), 3)
    heat = r.fused_heatmap if r.fused_heatmap is not None else np.zeros(rgb.shape[:2], np.float32)
    fig = np.hstack([panel(rgb, "input"), panel(truth, "ground truth (edited area)"),
                     panel(overlay(rgb, heat), f"Basira: {r.verdict}")])
    fig = cv2.resize(fig, (fig.shape[1] // 2, fig.shape[0] // 2), interpolation=cv2.INTER_AREA)
    Image.fromarray(fig).save(os.path.join(img_dir, f"demo_{name}.jpg"), quality=88)
    return r


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "examples", "out")
    img_dir = os.path.join(ROOT, "docs", "img")
    os.makedirs(out, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    for name, fn, seed in (("splice", S.make_splice, 2), ("copymove", S.make_copymove, 3),
                           ("inpaint", S.make_inpaint, 2)):
        path, mask = fn(out, seed)
        r = figure(path, mask, name, img_dir)
        save_html(r, path + ".basira.html")
        print(f"{name:9s} {r.verdict:28s} manipulation={r.scores['manipulation']:.2f}  {path}")
    clean, _ = S.make_authentic(out, 3)
    r = basira.analyze(clean)
    print(f"{'clean':9s} {r.verdict:28s} manipulation={r.scores['manipulation']:.2f}  {clean}")


if __name__ == "__main__":
    main()
