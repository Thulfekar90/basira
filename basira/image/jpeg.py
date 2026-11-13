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
from ..core.utils import calibrate
from .common import cohesive_anomaly, heat_from_z, regions_from_mask
from .load import ZIGZAG, ImageData

N_FREQ = 20              # first AC coefficients in zig-zag order
HIST_RANGE = 48
NOISE_FLOOR = 4.0        # |coefficient| below this is pixel-rounding noise between saves
PERIOD_MIN_STRENGTH = 3.0    # between-phase / within-phase variance of the folded histogram
MIN_PERIODIC_FREQS = 3
DQ_STAT_THRESHOLD = 0.06   # clean double-saved images measured <= 0.03 (real and synthetic)
DQ_STAT_SOFTNESS = 0.015


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
    """Detect the comb left by double quantisation in a coefficient histogram.

    For each candidate period p the histogram is divided by its own moving average over
    exactly one period (which flattens any comb of that period) and folded at p. A real
    comb gives phase means that differ strongly *and consistently*: the variance between
    phases must dominate the variance within phases, with a modulation of at least 30%.
    A smooth single-compression histogram fails both tests, whatever its Poisson noise.
    Returns (period, strength); period 1 means "no comb".
    """
    hf = h.astype(np.float64)
    idx = np.arange(len(h)) - center
    best_p, best_r = 1, 0.0
    for p in range(2, 11):
        kern = np.ones(p + 1)
        kern[0] = kern[-1] = 0.5              # centred window spanning one full period
        trend = np.convolve(hf, kern / p, mode="same")
        ok = (trend >= 8) & (np.abs(idx) >= 2)
        if ok.sum() < 4 * p:
            continue
        hn = hf / np.maximum(trend, 1e-9) - 1.0
        means, within = [], []
        for ph in range(p):
            sel = hn[ok & ((idx % p) == ph)]
            if sel.size < 3:
                break
            means.append(sel.mean())
            within.append(sel.var())
        else:
            means = np.asarray(means)
            ratio = float(means.var() / (np.mean(within) + 1e-4))
            if np.ptp(means) >= 0.3 and ratio > best_r * 1.25:   # prefer the fundamental
                best_p, best_r = p, ratio
    return (best_p, best_r) if best_r >= PERIOD_MIN_STRENGTH else (1, best_r)


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
    EU = np.zeros((hb, wb))
    ET = np.zeros((hb, wb))
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
        dead = (widx == 0) | (win < 20)
        llr_tab[dead] = 0.0
        # Expected LLR of a coefficient in each period window under both hypotheses:
        # "untouched" (follows the double-compression histogram) and "tampered"
        # (uniform inside the window). Normalising by these removes the dependence on
        # *which* windows a block's coefficients fall in, i.e. on local texture.
        wkey = (widx - widx.min()).astype(np.int64)
        eu_w = np.bincount(wkey, weights=pu * llr_tab)
        et_w = np.bincount(wkey, weights=llr_tab / p)
        eu_tab, et_tab = eu_w[wkey], et_w[wkey]
        eu_tab[dead], et_tab[dead] = 0.0, 0.0
        ki = np.clip(k + HIST_RANGE, 0, len(h) - 1)
        informative = use & ~dead[ki] & (np.abs(k) * q >= NOISE_FLOOR)
        L += np.where(informative, llr_tab[ki], 0.0)
        EU += np.where(informative, eu_tab[ki], 0.0)
        ET += np.where(informative, et_tab[ki], 0.0)
        N += informative
    info = {"periodic_frequencies": len(periodic), "periods": periodic[:10],
            "last_quality": d.jpeg_quality}
    if len(periodic) < MIN_PERIODIC_FREQS:
        return Evidence(
            detector=det, title=title, category=Category.MANIPULATION, score=0.0, confidence=0.1,
            summary="No double-compression periodicity found (image looks single-compressed, "
                    "or was resized / re-saved with a non-aligned grid). This detector cannot "
                    "localise edits in that case.", details=info)
    # Block score over a 3x3 neighbourhood: 0 = behaves like the doubly-compressed
    # majority, 1 = behaves like content compressed only once. It is an absolute scale,
    # so it also works when the edited area is large.
    k3 = (3, 3)
    num = cv2.blur((L - EU).astype(np.float32), k3)
    span = cv2.blur((ET - EU).astype(np.float32), k3)
    den = cv2.blur(N.astype(np.float32), k3)
    enough = (den >= 2.5) & (span > 0.5)      # >= ~22 informative coefficients nearby
    if enough.sum() < 50:
        return Evidence(detector=det, title=title, category=Category.MANIPULATION, score=0.0,
                        confidence=0.1, summary="Too few informative DCT coefficients to map "
                        "double compression.", details=info)
    frac_single = np.where(enough, np.clip(num / np.maximum(span, 1e-6), -0.5, 1.5), 0.0)
    z = frac_single * 5.0                     # cluster where the score exceeds 0.5
    stat, comp, cinfo = cohesive_anomaly(z, thr=2.5)
    score = calibrate(stat, DQ_STAT_THRESHOLD, DQ_STAT_SOFTNESS)
    conf = 0.8 if len(periodic) >= 5 else 0.6
    H, W = d.rgb.shape[:2]
    heat = heat_from_z(z, (H, W), lo=2.0, hi=5.0)
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
