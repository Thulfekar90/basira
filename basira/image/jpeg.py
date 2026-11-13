"""JPEG compression-history forensics.

``image.jpeg_dq`` — aligned double-quantisation analysis (after Lin et al.,
2009, "Fast, automatic and fine-grained tampered JPEG image detection via DCT
coefficient analysis"). When an image is saved twice, the histograms of its
quantised DCT coefficients become periodic. Regions pasted in *after* the first
save (or pasted with a shifted 8x8 grid) do not follow that periodicity, so each
8x8 block gets a likelihood ratio "single- vs double-compressed".
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..core.types import Category, Evidence
from ..core.utils import calibrate, robust_z
from .common import cohesive_anomaly, heat_from_z, regions_from_mask
from .load import ZIGZAG, ImageData

N_FREQ = 20              # first AC coefficients in zig-zag order
HIST_RANGE = 48
NOISE_FLOOR = 4.0        # |coefficient| below this is pixel-rounding noise between saves
PERIOD_MIN_STRENGTH = 0.30
MIN_PERIODIC_FREQS = 3
DQ_STAT_THRESHOLD = 0.30   # clean re-saved images measured <= 0.16, splices >= 0.44
DQ_STAT_SOFTNESS = 0.04


def _dct_matrix() -> np.ndarray:
    m = np.zeros((8, 8))
    for k in range(8):
        a = np.sqrt(1 / 8) if k == 0 else np.sqrt(2 / 8)
        m[k] = a * np.cos((2 * np.arange(8) + 1) * k * np.pi / 16)
    return m


DCT8 = _dct_matrix()


def block_dct(y: np.ndarray) -> np.ndarray:
    """Orthonormal 8x8 DCT of every block: returns (hb, wb, 8, 8)."""
    h, w = y.shape
    hb, wb = h // 8, w // 8
    b = (y[: hb * 8, : wb * 8] - 128.0).reshape(hb, 8, wb, 8).transpose(0, 2, 1, 3)
    return np.einsum("ij,abjk,lk->abil", DCT8, b, DCT8, optimize=True)


def _period(h: np.ndarray, center: int) -> Tuple[int, float]:
    """Detect a periodic pattern in a coefficient histogram. Returns (period, strength)."""
    hf = h.astype(np.float64)
    smooth = np.convolve(hf, np.ones(7) / 7, mode="same")
    idx = np.arange(len(h)) - center
    ok = (smooth > 4) & (np.abs(idx) >= 1)
    if ok.sum() < 12:
        return 1, 0.0
    hn = np.where(ok, hf / np.maximum(smooth, 1e-9) - 1.0, np.nan)
    total_var = np.nanvar(hn)
    if not np.isfinite(total_var) or total_var < 1e-4:
        return 1, 0.0
    best_p, best_s, strengths = 1, 0.0, {}
    for p in range(2, 11):
        means = []
        for ph in range(p):
            sel = hn[(idx % p) == ph]
            sel = sel[np.isfinite(sel)]
            if sel.size < 2:
                break
            means.append(sel.mean())
        else:
            strengths[p] = float(np.var(means) / total_var)
    if not strengths:
        return 1, 0.0
    smax = max(strengths.values())
    for p in sorted(strengths):  # smallest period close to the best (avoid multiples)
        if strengths[p] >= 0.8 * smax:
            best_p, best_s = p, strengths[p]
            break
    return (best_p, best_s) if best_s >= PERIOD_MIN_STRENGTH else (1, best_s)


def analyze_double_quantization(d: ImageData) -> Evidence:
    det, title = "image.jpeg_dq", "JPEG double-quantisation map"
    if not d.is_jpeg or d.quant is None or d.luma_y is None:
        return Evidence.not_applicable(det, title, Category.MANIPULATION, "not a JPEG file")
    y = d.luma_y
    if min(y.shape) < 64:
        return Evidence.not_applicable(det, title, Category.MANIPULATION, "image too small")
    C = block_dct(y)
    hb, wb = C.shape[:2]
    raw_blocks = y[: hb * 8, : wb * 8].reshape(hb, 8, wb, 8)
    sat = ((raw_blocks <= 1) | (raw_blocks >= 254)).mean(axis=(1, 3)) > 0.1
    Q = d.quant
    L = np.zeros((hb, wb))
    N = np.zeros((hb, wb))
    periodic: List[Dict] = []
    for zz in range(1, N_FREQ + 1):
        u, v = divmod(int(ZIGZAG[zz]), 8)
        q = Q[u, v]
        k = np.round(C[:, :, u, v] / q).astype(np.int64)
        use = (~sat) & (np.abs(k) <= HIST_RANGE)
        h = np.bincount(k[use] + HIST_RANGE, minlength=2 * HIST_RANGE + 1)
        p, strength = _period(h, HIST_RANGE)
        if p < 2:
            continue
        periodic.append({"freq": [u, v], "q2": float(q), "period": p, "strength": round(strength, 3)})
        # P_u(k) = h(k) / sum(h over its period window); P_t = 1/p. Windows are centred
        # on multiples of the period. The window around zero is skipped: tiny non-zero
        # coefficients there come from pixel rounding between the two saves, not from
        # a different compression history, and would light up every smooth area.
        idx = np.arange(len(h)) - HIST_RANGE
        widx = np.floor_divide(idx + p // 2, p)
        start = widx * p - p // 2 + HIST_RANGE
        csum = np.concatenate([[0], np.cumsum(h)])
        win = csum[np.clip(start + p, 0, len(h))] - csum[np.clip(start, 0, len(h))]
        pu = (h + 0.5) / (win + 0.5 * p)
        llr_tab = np.clip(np.log((1.0 / p) / pu), -3, 3)
        llr_tab[(widx == 0) | (win < 20)] = 0.0
        informative = use & (llr_tab[np.clip(k + HIST_RANGE, 0, len(h) - 1)] != 0) & \
            (np.abs(k) * q >= NOISE_FLOOR)
        L += np.where(informative, llr_tab[np.clip(k + HIST_RANGE, 0, len(h) - 1)], 0.0)
        N += informative
    info = {"periodic_frequencies": len(periodic), "periods": periodic[:10],
            "last_quality": d.jpeg_quality}
    if len(periodic) < MIN_PERIODIC_FREQS:
        return Evidence(
            detector=det, title=title, category=Category.MANIPULATION, score=0.0, confidence=0.1,
            summary="No double-compression periodicity found (image looks single-compressed, "
                    "or was resized / re-saved with a non-aligned grid). This detector cannot "
                    "localise edits in that case.", details=info)
    # Average evidence per informative coefficient over a 3x3 block neighbourhood.
    num = cv2.blur(L.astype(np.float32), (3, 3))
    den = cv2.blur(N.astype(np.float32), (3, 3))
    enough = den >= 2.5                       # >= ~22 informative coefficients nearby
    Ls = np.where(enough, num / (den + 0.5), np.nan)
    if np.isfinite(Ls).sum() < 50:
        return Evidence(detector=det, title=title, category=Category.MANIPULATION, score=0.0,
                        confidence=0.1, summary="Too few informative DCT coefficients to map "
                        "double compression.", details=info)
    z = np.where(enough, robust_z(Ls), 0.0)
    stat, comp, cinfo = cohesive_anomaly(z, thr=2.5)
    score = calibrate(stat, DQ_STAT_THRESHOLD, DQ_STAT_SOFTNESS)
    conf = 0.8 if len(periodic) >= 5 else 0.6
    H, W = d.rgb.shape[:2]
    heat = heat_from_z(z, (H, W), lo=2.5, hi=6.0)
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION, score=score, confidence=conf,
        summary=(f"The image was JPEG-compressed at least twice ({len(periodic)} DCT frequencies "
                 f"show double-quantisation periodicity), and a compact region "
                 f"({cinfo['frac'] * 100:.1f}% of blocks) does NOT share that history — "
                 f"consistent with content pasted in before the last save."
                 if score >= 0.5 else
                 f"The image was compressed at least twice ({len(periodic)} periodic DCT "
                 f"frequencies), and the double-compression traces are spatially consistent."),
        details={**info, "statistic": stat, "threshold": DQ_STAT_THRESHOLD, **cinfo},
        heatmap=heat if score >= 0.3 else heat * 0.5,
        regions=regions_from_mask(comp, 8, 8, score, "single-compressed block cluster")
        if score >= 0.5 else [])
