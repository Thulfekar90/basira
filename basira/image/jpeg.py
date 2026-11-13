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
# Valley / peak mass of the comb. Direct double saves measure <= 0.02 (<= 0.26 when a
# splice fills the valleys); images processed between saves (CoMoFoD re-compressed
# originals) measure 0.21-0.54 and produced false alarms before this check existed.
COMB_MAX_VALLEY = 0.20
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
    n = len(h)
    idx = np.arange(n) - center
    pos = np.arange(n)
    ratios: Dict[int, float] = {}
    for p in range(2, 11):
        kern = np.ones(p + 1)
        kern[0] = kern[-1] = 0.5              # centred window spanning one full period
        trend = np.convolve(hf, kern / p, mode="same")
        # Bins within p of either end see a zero-padded average: leave them out.
        ok = (trend >= 8) & (np.abs(idx) >= 2) & (pos >= p) & (pos < n - p)
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
            if np.ptp(means) >= 0.3:
                ratios[p] = float(means.var() / (np.mean(within) + 1e-4))
    if not ratios:
        return 1, 0.0
    best_p = max(ratios, key=lambda k: ratios[k])
    best_r = ratios[best_p]
    # A comb of period d also folds perfectly at 2d, 3d...: resolve to the fundamental.
    for d in sorted(ratios):
        if d < best_p and best_p % d == 0 and ratios[d] >= 0.5 * best_r:
            best_p, best_r = d, ratios[d]
            break
    return (best_p, best_r) if best_r >= PERIOD_MIN_STRENGTH else (1, best_r)


def _llr_tables(h: np.ndarray, p: int, q: float, center: int):
    """Per-bin log-likelihood ratio "single vs double compressed" and its expectation
    under both hypotheses.

    The histogram is cut into windows of one period, centred on multiples of p. Inside
    a window, an untouched (doubly compressed) coefficient follows the measured comb
    P_u; a tampered one is uniform, P_t. Both are normalised over exactly the bins that
    are scored: bins below the pixel-rounding noise floor, the window around zero and
    windows cut off by the histogram range are dropped, and a window needs at least two
    scored bins to discriminate anything. Returns (llr, E_u[llr], E_t[llr], live).
    """
    n = len(h)
    idx = np.arange(n) - center
    widx = np.floor_divide(idx + p // 2, p)
    eligible = np.abs(idx) * q >= NOISE_FLOOR
    llr, eu, et = np.zeros(n), np.zeros(n), np.zeros(n)
    live = np.zeros(n, bool)
    for w in np.unique(widx):
        if w == 0:
            continue
        bins = np.flatnonzero(widx == w)
        if len(bins) != p:
            continue                           # incomplete window at the histogram edge
        keep = bins[eligible[bins]]
        if len(keep) < 2 or h[keep].sum() < 20:
            continue
        pu = (h[keep] + 0.5) / (h[keep].sum() + 0.5 * len(keep))
        pt = 1.0 / len(keep)
        lw = np.clip(np.log(pt / pu), -3, 3)
        llr[keep] = lw
        eu[keep] = float((pu * lw).sum())
        et[keep] = float((pt * lw).sum())
        live[keep] = True
    return llr, eu, et, live


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
    smeared = 0
    for zz in range(1, N_FREQ + 1):
        u, v = divmod(int(ZIGZAG[zz]), 8)
        q = Q[u, v]
        k = np.round(C[:, :, u, v] / q).astype(np.int64)
        use = (~sat) & (np.abs(k) <= HIST_RANGE)
        h = np.bincount(k[use] + HIST_RANGE, minlength=2 * HIST_RANGE + 1)
        p, strength = _period(h, HIST_RANGE)
        if p < 2:
            continue
        # Comb purity. After a direct second save the valleys of the comb are (almost)
        # empty. If the image was processed between the saves - colour adjustment,
        # filtering, partial re-alignment - the valleys fill up everywhere, the
        # single-vs-double model no longer describes untouched content, and local
        # differences in that smearing would be mistaken for edits. Skip such combs.
        idx0 = np.arange(len(h)) - HIST_RANGE
        sel = (np.abs(idx0) >= 2) & (np.abs(idx0) <= 20)
        mass = np.array([h[sel & ((idx0 % p) == a)].sum() for a in range(p)], dtype=np.float64)
        if mass.min() / max(mass.max(), 1.0) > COMB_MAX_VALLEY:
            smeared += 1
            continue
        periodic.append({"freq": [u, v], "q2": float(q), "period": p, "strength": round(strength, 3)})
        llr_tab, eu_tab, et_tab, live = _llr_tables(h, p, float(q), HIST_RANGE)
        ki = np.clip(k + HIST_RANGE, 0, len(h) - 1)
        informative = use & live[ki]
        L += np.where(informative, llr_tab[ki], 0.0)
        EU += np.where(informative, eu_tab[ki], 0.0)
        ET += np.where(informative, et_tab[ki], 0.0)
        N += informative
    info = {"periodic_frequencies": len(periodic), "smeared_combs": smeared,
            "periods": periodic[:10], "last_quality": d.jpeg_quality}
    if len(periodic) < MIN_PERIODIC_FREQS and smeared >= MIN_PERIODIC_FREQS:
        return Evidence(
            detector=det, title=title, category=Category.MANIPULATION, score=0.0, confidence=0.1,
            summary=f"The image was compressed more than once ({smeared} DCT frequencies show "
                    f"a comb), but the combs are smeared: it was processed between the saves "
                    f"(colour or tone adjustment, filtering). Edits cannot be localised from "
                    f"compression history in that case.", details=info)
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
