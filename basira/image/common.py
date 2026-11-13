"""Helpers shared by pixel-domain image detectors."""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..core.types import Region


def cohesive_anomaly(z: np.ndarray, thr: float = 2.5, min_blocks: int = 4
                     ) -> Tuple[float, np.ndarray, dict]:
    """Score how much of a block-level z-map forms *one compact* anomalous blob.

    Isolated outlier blocks happen in every natural image; a forged region shows
    up as a connected cluster. Statistic = area fraction of the largest
    component x clipped mean z inside it. Returns (stat, component mask, info).
    """
    z = np.nan_to_num(z, nan=0.0)
    m = (cv2.medianBlur((z > thr).astype(np.uint8) * 255, 3) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return 0.0, np.zeros_like(m, bool), {"frac": 0.0, "mean_z": 0.0, "n_blocks": 0}
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = labels == idx
    nb = int(comp.sum())
    if nb < min_blocks:
        return 0.0, comp, {"frac": nb / z.size, "mean_z": float(z[comp].mean()), "n_blocks": nb}
    frac = nb / z.size
    mean_z = float(z[comp].mean())
    # A blob covering most of the image is a global property, not a local edit.
    size_pen = 1.0 if frac < 0.45 else max(0.0, 1 - (frac - 0.45) / 0.3)
    stat = float(np.sqrt(frac) * min(mean_z / thr, 3.0) * size_pen)
    return stat, comp, {"frac": frac, "mean_z": mean_z, "n_blocks": nb}


def regions_from_mask(mask: np.ndarray, scale_y: float, scale_x: float, score: float,
                      label: str, max_regions: int = 4) -> List[Region]:
    m = mask.astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        out.append((a, Region(int(x * scale_x), int(y * scale_y), int(w * scale_x),
                              int(h * scale_y), score, label)))
    out.sort(key=lambda t: -t[0])
    return [r for _, r in out[:max_regions]]


def heat_from_z(z: np.ndarray, shape, lo: float = 1.5, hi: float = 5.0) -> np.ndarray:
    z = np.nan_to_num(z, nan=0.0)
    hm = np.clip((z - lo) / (hi - lo), 0, 1).astype(np.float32)
    hm = cv2.resize(hm, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(hm, (0, 0), max(1.0, min(shape) / 300))


def working_gray(rgb: np.ndarray, max_side: int = 2048) -> Tuple[np.ndarray, float]:
    """Float32 gray image, downscaled (INTER_AREA) if huge; returns (gray, scale)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = g.shape
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        g = cv2.resize(g, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return g, s
