"""Traces of an earlier JPEG save, block by block.

``image.jpeg_trace`` — most images were a JPEG at some point. Decoding a JPEG leaves
every 8x8 DCT coefficient of the luma plane close to a multiple of that save's
quantisation step, and that stays measurable after the picture is stored losslessly
(TIFF, PNG, BMP). Content pasted in afterwards — scaled,
rotated, shifted off the 8x8 grid, or taken from a JPEG of another quality — does not
sit on those multiples.

For each low frequency the step q is found from the whole image (the largest q whose
phase ``cos(2 pi X / q)`` is clearly positive on average, mean resultant length R).
Each coefficient then votes with a von Mises log-likelihood "on the comb vs uniform",
weighted by the comb's concentration, and a compact cluster of blocks that are off the
comb is reported. It is the pixel-domain relative of ``image.jpeg_dq`` (Fan & de
Queiroz 2003 for the comb estimate; Bianchi & Piva 2012 for the local map) and needs no
knowledge of the encoder.

Preconditions: a JPEG history on an 8x8 grid that survived (no resize, no global
filtering after the save); textured content (smooth blocks carry no information).

JPEG files are left to ``image.jpeg_dq``: on the CASIA v2 development half this map
found no JPEG forgery that jpeg_dq had missed, and on re-compressed CoMoFoD originals
it came closer to its threshold than on any lossless file.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..core.types import Category, Evidence
from ..core.utils import calibrate
from .common import cohesive_anomaly, heat_from_z, regions_from_mask
from .jpeg import block_dct
from .load import ZIGZAG, ImageData

N_FREQ = 15
Q_MAX = 40
R_MIN = 0.5               # mean cos needed to accept a comb at one frequency
MIN_FREQS = 4             # frequencies with a comb needed before mapping anything
MIN_SPAN = 4.0            # expected LLR mass in a 3x3 neighbourhood to score a block
CLIP_MAX = 0.02           # blocks with more clipped colour pixels than this are skipped
SAMPLE_BLOCKS = 40000     # comb estimation uses at most this many blocks
STAT_THRESHOLD = 0.35     # set on the CASIA v2 development half; see docs/METHODS.md
STAT_SOFTNESS = 0.05


def _y_plane(d: ImageData) -> np.ndarray:
    rgb = d.rgb.astype(np.float64)
    if rgb.ndim == 2:
        return rgb
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _grid(y: np.ndarray) -> Tuple[int, int, float]:
    """8x8 grid offset of an earlier JPEG save from the blocking at block boundaries."""
    def peak(diff: np.ndarray) -> Tuple[int, float]:
        m = np.array([diff[r::8].mean() for r in range(8)])
        r = int(np.argmax(m))
        rest = np.delete(m, r)
        return (r + 1) % 8, float((m[r] - rest.mean()) / (rest.std() + 1e-9))
    ox, sx = peak(np.abs(np.diff(y, axis=1)).mean(axis=0))
    oy, sy = peak(np.abs(np.diff(y, axis=0)).mean(axis=1))
    return oy, ox, min(sx, sy)


def _comb(x: np.ndarray, q: int) -> Tuple[float, int]:
    sel = np.abs(x) >= q / 2
    n = int(sel.sum())
    if n < 30:
        return 0.0, n
    return float(np.cos(2 * np.pi * x[sel] / q).mean()), n


def _kappa(r: float) -> float:
    """Von Mises concentration for mean resultant length r (Fisher's approximation)."""
    r = min(r, 0.98)
    if r < 0.53:
        return 2 * r + r ** 3 + 5 * r ** 5 / 6
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1 - r)
    return 1 / (r ** 3 - 4 * r ** 2 + 3 * r)


def analyze_jpeg_trace(d: ImageData) -> Evidence:
    det, title = "image.jpeg_trace", "Earlier-JPEG trace map"
    if d.is_jpeg:
        return Evidence.not_applicable(det, title, Category.MANIPULATION,
                                       "JPEG file (covered by the double-quantisation map)")
    if d.format not in ("PNG", "BMP", "TIFF", "WEBP") or not d.lossless:
        return Evidence.not_applicable(det, title, Category.MANIPULATION,
                                       "requires a known lossless PNG, BMP, TIFF or WEBP encoding")
    y = _y_plane(d)
    if min(y.shape) < 64:
        return Evidence.not_applicable(det, title, Category.MANIPULATION, "image too small")
    oy, ox, grid_strength = _grid(y)
    y = y[oy:, ox:]
    C = block_dct(y)
    hb, wb = C.shape[:2]
    raw = y[: hb * 8, : wb * 8].reshape(hb, 8, wb, 8)
    sat = ((raw <= 1) | (raw >= 254)).mean(axis=(1, 3)) > 0.1
    if d.rgb.ndim == 3:
        # A saturated colour is clipped in R, G or B by the decoder, so the luma recomputed
        # from RGB is off the comb there although nothing was edited (measured on the
        # red, orange and green objects that produced every false alarm on CASIA v2).
        rgb = d.rgb[oy:oy + hb * 8, ox:ox + wb * 8, :3]
        clip = ((rgb <= 0) | (rgb >= 255)).any(axis=2)
        sat |= clip.reshape(hb, 8, wb, 8).mean(axis=(1, 3)) > CLIP_MAX
    rng = np.random.default_rng(0)
    usable = np.flatnonzero(~sat.ravel())
    if usable.size > SAMPLE_BLOCKS:
        usable = rng.choice(usable, SAMPLE_BLOCKS, replace=False)
    num = np.zeros((hb, wb))
    span = np.zeros((hb, wb))
    combs: List[Dict] = []
    for zz in range(1, N_FREQ + 1):
        u, v = divmod(int(ZIGZAG[zz]), 8)
        x = C[:, :, u, v]
        xs = x.ravel()[usable]
        found = None
        for q in range(Q_MAX, 1, -1):
            r, n = _comb(xs, q)
            if r >= R_MIN and n * r * r >= 25:
                found = (q, r, n)
                break
        if found is None:
            continue
        q, r, n = found
        combs.append({"freq": [u, v], "step": q, "R": round(r, 3), "n": n})
        k = _kappa(r)
        inf = (~sat) & (np.abs(x) >= q / 2)
        c = np.cos(2 * np.pi * x / q)
        num += np.where(inf, k * (r - c), 0.0)
        span += np.where(inf, k * r, 0.0)
    info = {"combs": combs[:10], "n_combs": len(combs), "grid": [oy, ox],
            "grid_strength": round(grid_strength, 2)}
    if len(combs) < MIN_FREQS:
        return Evidence(detector=det, title=title, category=Category.MANIPULATION, score=0.0,
                        confidence=0.1, summary="Not applicable here: no trace of an earlier "
                        "JPEG save (camera original, resized, or never a JPEG).", details=info)
    k3 = (3, 3)
    a = cv2.blur(num.astype(np.float32), k3)
    b = cv2.blur(span.astype(np.float32), k3)
    enough = b >= MIN_SPAN
    if enough.sum() < 50:
        return Evidence(detector=det, title=title, category=Category.MANIPULATION, score=0.0,
                        confidence=0.1, summary="Too little texture to map the earlier JPEG "
                        "save.", details=info)
    off = np.where(enough, np.clip(a / np.maximum(b, 1e-6), -0.5, 1.5), 0.0)
    z = off * 5.0                            # cluster where more than half is off the comb
    stat, comp, cinfo = cohesive_anomaly(z, thr=2.5)
    score = calibrate(stat, STAT_THRESHOLD, STAT_SOFTNESS)
    conf = 0.8 if len(combs) >= 6 else 0.6
    H, W = d.rgb.shape[:2]
    heat = np.zeros((H, W), np.float32)
    heat[oy:oy + hb * 8, ox:ox + wb * 8] = heat_from_z(z, (hb * 8, wb * 8), lo=2.0, hi=5.0)
    src = "an earlier JPEG save"
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION, score=score, confidence=conf,
        summary=(f"Most of the image still carries the quantisation of {src} "
                 f"({len(combs)} DCT frequencies), but a compact region "
                 f"({cinfo['frac'] * 100:.1f}% of blocks) does not — consistent with content "
                 f"pasted in after that save (scaled, shifted or from another JPEG)."
                 if score >= 0.5 else
                 f"The image carries the quantisation of {src} ({len(combs)} DCT "
                 f"frequencies), consistently across the picture."),
        details={**info, "statistic": stat, "threshold": STAT_THRESHOLD, **cinfo},
        heatmap=heat if score >= 0.3 else heat * 0.5,
        regions=[r.__class__(r.x + ox, r.y + oy, r.w, r.h, r.score, r.label)
                 for r in regions_from_mask(comp, 8, 8, score, "block cluster off the JPEG comb")]
        if score >= 0.5 else [])
