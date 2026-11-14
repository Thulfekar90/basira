"""Texture-normalised Error Level Analysis.

Classic ELA (re-save, take the difference) is notorious for false positives:
textured areas *always* light up. Basira regresses the per-block ELA energy on
the block's own texture energy and only flags blocks whose compression error is
abnormal *for their texture*, and only when they form a compact cluster.
"""
from __future__ import annotations

import io
import warnings

import cv2
import numpy as np
from PIL import Image

from ..core.types import Category, Evidence
from ..core.utils import calibrate, robust_z
from .common import cohesive_anomaly, heat_from_z, regions_from_mask
from .load import ImageData

BLOCK = 16
Z_THR = 3.0
STAT_THRESHOLD = 0.45   # clean images measured <= 0.40
STAT_SOFTNESS = 0.06


def _resave(rgb: np.ndarray, q: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=int(q), subsampling=0)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def analyze_ela(d: ImageData) -> Evidence:
    rgb = d.rgb
    h, w = rgb.shape[:2]
    if min(h, w) < 4 * BLOCK:
        return Evidence.not_applicable("image.ela", "Error level analysis",
                                       Category.MANIPULATION, "image too small")
    if max(h, w) > 65500:                                  # the JPEG format's own size limit
        return Evidence.not_applicable("image.ela", "Error level analysis",
                                       Category.MANIPULATION,
                                       "image too large to re-save as JPEG")
    q = d.jpeg_quality if (d.is_jpeg and d.jpeg_quality) else 90
    q = int(np.clip(q, 70, 98))
    re = _resave(rgb, q).astype(np.float32)
    err = np.abs(rgb.astype(np.float32) - re).max(axis=2)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    hb, wb = h // BLOCK, w // BLOCK
    e_blk = err[: hb * BLOCK, : wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).mean(axis=(1, 3))
    t_blk = lap[: hb * BLOCK, : wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).mean(axis=(1, 3))

    le, lt = np.log(e_blk + 0.05), np.log(t_blk + 0.5)
    flat = t_blk < 0.8          # saturated / perfectly flat blocks carry no information
    valid = ~flat
    if valid.sum() < 16:
        return Evidence.not_applicable("image.ela", "Error level analysis",
                                       Category.MANIPULATION, "image has almost no texture")
    # Robust quadratic fit of log-error vs log-texture (two Huber-like passes).
    x, y = lt[valid], le[valid]
    wts = np.ones_like(x)
    for _ in range(3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coef = np.polyfit(x, y, 2, w=wts)
        r = y - np.polyval(coef, x)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        wts = 1.0 / np.maximum(1.0, np.abs(r) / (2.0 * s))
    resid = le - np.polyval(coef, lt)
    z = robust_z(np.where(valid, resid, np.nan))
    z = np.where(valid, z, 0.0)

    stat, comp, info = cohesive_anomaly(z, thr=Z_THR)
    score = calibrate(stat, STAT_THRESHOLD, STAT_SOFTNESS)
    # ELA is kept mainly as a familiar visual: on our benchmark it localises poorly
    # compared with the DQ and noise maps, so it gets a deliberately small vote.
    conf = 0.25 if d.is_jpeg else 0.15
    heat = heat_from_z(z, (h, w), lo=2.0, hi=6.0)
    ela_vis = np.clip(err * (255.0 / max(np.percentile(err, 99.5), 1.0)), 0, 255).astype(np.uint8)
    return Evidence(
        detector="image.ela", title="Error level analysis (texture-normalised)",
        category=Category.MANIPULATION, score=score, confidence=conf,
        summary=(f"A compact region ({info['frac'] * 100:.1f}% of the image) re-compresses "
                 f"abnormally for its texture (mean z={info['mean_z']:.1f}) — it likely has a "
                 f"different compression history from the rest." if score >= 0.5 else
                 f"Compression error is consistent with texture everywhere (statistic "
                 f"{stat:.3f} < {STAT_THRESHOLD})."),
        details={"resave_quality": q, "statistic": stat, "threshold": STAT_THRESHOLD, **info},
        heatmap=heat if score >= 0.3 else heat * 0.5,
        regions=regions_from_mask(comp, BLOCK, BLOCK, score, "ELA anomaly") if score >= 0.5 else [],
        visuals={"ela": np.stack([ela_vis] * 3, -1)})
