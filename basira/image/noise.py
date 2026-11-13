"""Abnormally smooth regions (AI in-painting / generative fill / heavy retouching).

Every capture pipeline leaves a characteristic noise level that depends on the
sensor, ISO and processing, and on pixel intensity and local texture. An
AI-in-painted or heavily retouched area usually carries much less noise than its
surroundings. We estimate noise per block with Immerkaer's fast
Laplacian-difference estimator on non-edge pixels, remove the expected dependence
on brightness and texture with a robust fit, and look for a compact cluster of
blocks that are abnormally smooth. (The opposite direction, "noisier than
expected", is not used: it is dominated by fine texture and produced false alarms
on camera originals.)
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.types import Category, Evidence
from ..core.utils import calibrate, robust_z
from .common import cohesive_anomaly, heat_from_z, regions_from_mask, working_gray
from .load import ImageData

KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
Z_THR = 3.0
STAT_THRESHOLD = 0.29   # clean images measured <= 0.25
STAT_SOFTNESS = 0.03


def noise_map(gray: np.ndarray, block: int):
    """Per-block noise sigma (NaN where too few smooth pixels) and block mean intensity."""
    res = np.abs(cv2.filter2D(gray, cv2.CV_32F, KERNEL, borderType=cv2.BORDER_REFLECT))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    edge_thr = max(np.percentile(grad, 85), 8.0)
    ok = (grad < edge_thr) & (gray > 4) & (gray < 251)   # avoid edges and clipping
    h, w = gray.shape
    hb, wb = h // block, w // block
    sl = (slice(0, hb * block), slice(0, wb * block))
    r = res[sl].reshape(hb, block, wb, block)
    k = ok[sl].reshape(hb, block, wb, block)
    cnt = k.sum(axis=(1, 3))
    s = (r * k).sum(axis=(1, 3)) / np.maximum(cnt, 1)
    sigma = np.sqrt(np.pi / 2) / 6.0 * s
    sigma[cnt < 0.25 * block * block] = np.nan
    inten = gray[sl].reshape(hb, block, wb, block).mean(axis=(1, 3))
    return sigma, inten


def analyze_noise(d: ImageData) -> Evidence:
    gray, s = working_gray(d.rgb, 2048)
    h, w = gray.shape
    block = int(np.clip(min(h, w) // 24, 16, 48))
    if min(h, w) < 6 * block:
        return Evidence.not_applicable("image.noise", "Noise consistency",
                                       Category.MANIPULATION, "image too small")
    sigma, inten = noise_map(gray, block)
    valid = np.isfinite(sigma) & (sigma > 0)
    if valid.sum() < 20:
        return Evidence.not_applicable("image.noise", "Noise consistency",
                                       Category.MANIPULATION, "not enough smooth area")
    ls = np.log(np.where(valid, sigma, 1.0) + 0.05)
    # Expected noise depends on brightness (shot noise) and, after JPEG, on local
    # texture (quantisation wipes noise out of smooth blocks). Fit both, robustly.
    lowpass = cv2.GaussianBlur(gray, (0, 0), 2.0)
    hb, wb = sigma.shape
    lp = lowpass[: hb * block, : wb * block].reshape(hb, block, wb, block)
    tex = np.log(lp.std(axis=(1, 3)) + 0.5)
    I = inten / 255.0
    A = np.stack([np.ones_like(I), I, I * I, tex, tex * tex, I * tex], axis=-1)
    Av, y = A[valid], ls[valid]
    wts = np.ones(len(y))
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(Av * wts[:, None], y * wts, rcond=None)
        r = y - Av @ coef
        sd = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        wts = np.sqrt(1.0 / np.maximum(1.0, np.abs(r) / (2.0 * sd)))
    resid = np.where(valid, ls - A @ coef, np.nan)
    z = robust_z(resid)
    z = np.where(valid, z, 0.0)

    # Only *abnormally smooth* regions count. Regions that look noisier than expected are
    # usually fine texture (fabric, foliage, print) that no brightness/texture model fully
    # predicts: on the Columbia camera originals that direction fired on 11 of 183
    # authentic images and separated nothing (AUC 0.55), so it is not used as evidence.
    stat, comp, info = cohesive_anomaly(-z, thr=Z_THR)
    kind, heat_z = "smoother", -z
    score = calibrate(stat, STAT_THRESHOLD, STAT_SOFTNESS)
    # Deliberately a weak vote. Smooth regions also occur naturally (defocus, sky, studio
    # backdrops): on public datasets this detector fired on 7 of 200 CoMoFoD originals and
    # 1 of 183 Columbia originals. At this confidence only a very strong anomaly can flag
    # an image on its own; otherwise it contributes a lead and a map.
    conf = 0.5
    if d.is_jpeg and d.jpeg_quality and d.jpeg_quality < 80:
        conf = 0.4
    H, W = d.rgb.shape[:2]
    heat = heat_from_z(heat_z, (H, W), lo=2.0, hi=6.0)
    med_sigma = float(np.nanmedian(sigma))
    reg_sigma = float(np.nanmedian(sigma[comp])) if comp.any() else med_sigma
    return Evidence(
        detector="image.noise", title="Sensor-noise consistency",
        category=Category.MANIPULATION, score=score, confidence=conf,
        summary=(f"A compact region ({info['frac'] * 100:.1f}% of the image) is markedly "
                 f"{kind} than the rest after accounting for brightness (noise σ≈{reg_sigma:.2f} "
                 f"vs {med_sigma:.2f}). Typical of pasted content or AI in-painting."
                 if score >= 0.5 else
                 f"Noise level is consistent across the image (statistic {stat:.3f} < "
                 f"{STAT_THRESHOLD})."),
        details={"block": block, "statistic": stat, "threshold": STAT_THRESHOLD,
                 "direction": kind, "median_sigma": med_sigma, "region_sigma": reg_sigma, **info},
        heatmap=heat if score >= 0.3 else heat * 0.5,
        regions=regions_from_mask(comp, block / s, block / s, score, f"{kind} noise")
        if score >= 0.5 else [])
