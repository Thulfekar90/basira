"""Frequency-domain cues of synthetic images and of camera pipelines.

``image.spectral`` — Generative decoders (GAN up-convolutions, latent-diffusion
VAEs that upsample 8x) imprint periodic correlations on the noise residual,
which appear as sharp peaks on a regular lattice in the residual's power
spectrum (Zhang et al. 2019; Corvi et al. 2023). Natural photos have a smooth
residual spectrum apart from the JPEG 8x8 lattice and the colour-filter-array
frequency, which we account for.

``image.cfa`` — Real cameras sample colour through a Bayer filter and
interpolate (demosaic). The interpolated green pixels are linear combinations
of their neighbours, which leaves a 2-periodic prediction-error pattern
(Popescu & Farid 2005). Its presence is evidence of a camera pipeline; its
local absence inside an otherwise demosaiced image localises spliced or
synthesised regions (Ferrara et al. 2012). Strong JPEG compression and
resizing destroy it, which lowers this detector's confidence.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import median_filter

from ..core.types import Category, Evidence
from ..core.utils import calibrate
from .common import cohesive_anomaly, heat_from_z, regions_from_mask
from .load import ImageData

TILE = 256
PEAK_THRESHOLD = 5.0       # dB above local background for a lattice peak to count
# dB above the off-lattice null reference. Camera originals are not flat here: demosaicing
# and in-camera sharpening leave lattice peaks of their own (183 Columbia originals:
# median 1.0 dB, 99th percentile 7.2 dB, maximum 7.6 dB), so the threshold sits above them.
SPEC_STAT_THRESHOLD = 10.0
SPEC_STAT_SOFTNESS = 1.5
CFA_RATIO_THRESHOLD = 1.06   # mean|e| ratio between the two green lattices
CFA_BLOCK = 32
CFA_INVERSION_MARGIN = 0.15  # block log-ratio below -margin = opposite Bayer phase
# Cluster statistic of phase-inverted blocks. Columbia: 0 of 183 originals reach 0.05.
CFA_STAT_THRESHOLD = 0.10
CFA_STAT_SOFTNESS = 0.025


def _residual(gray: np.ndarray) -> np.ndarray:
    return gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)


def residual_spectrum(gray: np.ndarray, tile: int = TILE, max_tiles: int = 24) -> np.ndarray:
    """Average log power spectrum of the noise residual over tiles (fftshifted, tile x tile)."""
    r = _residual(gray)
    h, w = r.shape
    ys = np.linspace(0, h - tile, max(1, min(6, h // tile))).astype(int)
    xs = np.linspace(0, w - tile, max(1, min(6, w // tile))).astype(int)
    win = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    acc = np.zeros((tile, tile), np.float64)
    n = 0
    for y in ys:
        for x in xs:
            t = r[y:y + tile, x:x + tile]
            t = (t - t.mean()) * win
            acc += np.abs(np.fft.fft2(t)) ** 2
            n += 1
            if n >= max_tiles:
                break
    acc /= max(n, 1)
    return np.fft.fftshift(10 * np.log10(acc + 1e-9))


def lattice_peaks(spec_db: np.ndarray, period: int = 8):
    """Peak height (dB over local median background) at the k/period lattice frequencies."""
    n = spec_db.shape[0]
    bg = median_filter(spec_db.astype(np.float32), size=9, mode="wrap")
    excess = spec_db - bg
    c = n // 2
    out = {}
    for ky in range(-period // 2, period // 2 + 1):
        for kx in range(-period // 2, period // 2 + 1):
            if (kx, ky) == (0, 0):
                continue
            y, x = c + ky * n // period, c + kx * n // period
            if 0 <= y < n and 0 <= x < n:
                y0, y1, x0, x1 = max(y - 1, 0), min(y + 2, n), max(x - 1, 0), min(x + 2, n)
                out[(kx, ky)] = float(excess[y0:y1, x0:x1].max())
    return out, excess


def lattice_statistic(candidates: dict, excess: np.ndarray) -> float:
    """Mean of the six strongest lattice peaks, in dB above an off-lattice null reference.

    Null reference: the same 3x3-max statistic at random off-lattice frequencies. The
    expected top-6 mean of that many null draws is about mean + 1.6 sd; only the part of
    the lattice response above it counts (otherwise plain noise images look "periodic").
    """
    if not candidates:
        return 0.0
    n = excess.shape[0]
    rs = np.random.RandomState(0)
    ry, rx = rs.randint(2, n - 2, 3000), rs.randint(2, n - 2, 3000)
    off = (ry % (n // 8) > 3) & (rx % (n // 8) > 3) & (np.hypot(ry - n // 2, rx - n // 2) > 8)
    null = np.array([excess[y - 1:y + 2, x - 1:x + 2].max() for y, x in zip(ry[off], rx[off])])
    null_ref = float(null.mean() + 1.6 * null.std()) if null.size > 50 else 0.0
    return float(np.mean(sorted(candidates.values(), reverse=True)[:6]) - null_ref)


def non_camera_peaks(peaks8: dict) -> dict:
    """Lattice frequencies not explained by a Bayer CFA (which lives at +-N/2)."""
    return {k: v for k, v in peaks8.items() if abs(k[0]) != 4 and abs(k[1]) != 4}


def cfa_map(rgb: np.ndarray, block: int = 32):
    """Ratio of green-channel prediction error between the two quincunx lattices."""
    g = rgb[..., 1].astype(np.float32)
    pred = (np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)) / 4
    e = np.abs(g - pred)
    e[:2], e[-2:], e[:, :2], e[:, -2:] = 0, 0, 0, 0
    yy, xx = np.indices(g.shape)
    lat = ((yy + xx) % 2 == 0)
    h, w = g.shape
    hb, wb = h // block, w // block
    sl = (slice(0, hb * block), slice(0, wb * block))
    E = e[sl].reshape(hb, block, wb, block)
    Lm = lat[sl].reshape(hb, block, wb, block)
    a = (E * Lm).sum(axis=(1, 3)) / np.maximum(Lm.sum(axis=(1, 3)), 1)
    b = (E * ~Lm).sum(axis=(1, 3)) / np.maximum((~Lm).sum(axis=(1, 3)), 1)
    texture = (a + b) / 2
    glob = float(e[lat].mean() / max(e[~lat].mean(), 1e-6))
    # Orientation-aware local ratio: acquired lattice / interpolated lattice.
    ratio = np.where(glob >= 1, a / np.maximum(b, 1e-3), b / np.maximum(a, 1e-3))
    return ratio, texture, max(glob, 1 / max(glob, 1e-6))


def analyze_spectral(d: ImageData) -> Evidence:
    det, title = "image.spectral", "Generator fingerprint (residual spectrum)"
    gray = cv2.cvtColor(d.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    if min(h, w) < TILE:
        return Evidence.not_applicable(det, title, Category.AI_GENERATION,
                                       f"image smaller than {TILE}px")
    spec = residual_spectrum(gray)
    peaks8, excess = lattice_peaks(spec, 8)
    peaks4 = {k: v for k, v in peaks8.items() if k[0] % 2 == 0 and k[1] % 2 == 0}
    # Frequencies that a *camera + JPEG* pipeline explains: the 8x8 JPEG lattice (when
    # JPEG) and the CFA Nyquist frequencies (±4 on the 8-lattice).
    if d.is_jpeg:
        jpeg_strength = float(np.median(list(peaks8.values())))
        candidates = {}   # JPEG blocking produces exactly this lattice — can't separate
    else:
        jpeg_strength = 0.0
        candidates = non_camera_peaks(peaks8)
    strong = {k: v for k, v in candidates.items() if v > PEAK_THRESHOLD}
    stat = lattice_statistic(candidates, excess)
    # Off-lattice periodic spikes (any strong isolated peak), e.g. other upsampling factors.
    ex = excess.copy()
    c = ex.shape[0] // 2
    ex[c - 3:c + 4, c - 3:c + 4] = 0
    off_peak = float(np.percentile(ex, 99.97))

    ratio, texture, cfa_glob = cfa_map(d.rgb)
    camera_like = cfa_glob > CFA_RATIO_THRESHOLD
    score = calibrate(stat, SPEC_STAT_THRESHOLD, SPEC_STAT_SOFTNESS)
    if d.is_jpeg or not d.lossless:
        conf = 0.1          # block-based lossy codecs occupy the same lattice
    elif d.palette:
        conf = 0.15         # palette quantisation / dithering is periodic too
    else:
        conf = 0.5
    spec_vis = np.clip((spec - np.percentile(spec, 1)) / (np.ptp(spec) + 1e-6) * 255, 0, 255)
    spec_vis = cv2.applyColorMap(spec_vis.astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    return Evidence(
        detector=det, title=title, category=Category.AI_GENERATION, score=score, confidence=conf,
        summary=(f"The noise residual has {len(strong)} sharp periodic peaks on the 1/8 "
                 f"frequency lattice ({stat:.1f} dB above the off-lattice background) — the signature of a "
                 f"generator's upsampling layers." if score >= 0.5 else
                 ("JPEG blocking occupies the same frequency lattice, so generator "
                  "periodicity cannot be measured reliably on this file."
                  if d.is_jpeg else
                  f"No generator-like periodic peaks in the residual spectrum (mean lattice "
                  f"excess {stat:.1f} dB < {SPEC_STAT_THRESHOLD} dB).")) +
                (" Camera demosaicing traces are present." if camera_like and score < 0.5 else ""),
        details={"lattice_excess_db": stat, "strong_peaks": len(strong),
                 "threshold_db": SPEC_STAT_THRESHOLD, "offlattice_p99.97_db": off_peak,
                 "jpeg_lattice_db": jpeg_strength, "cfa_ratio": cfa_glob,
                 "peaks": {f"{k[0]},{k[1]}": round(v, 2) for k, v in sorted(
                     peaks8.items(), key=lambda kv: -kv[1])[:12]}},
        visuals={"residual_spectrum": spec_vis})


def _inversion_heat(z: np.ndarray, shape) -> np.ndarray:
    """Solid map of the phase-inverted area: flat or dark blocks inside a pasted region
    carry no CFA reading of their own, so gaps between inverted blocks are closed."""
    inv = (z > 2.0).astype(np.uint8)
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    heat = cv2.resize(inv.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(heat, (0, 0), max(2.0, min(shape) / 120))


def analyze_cfa(d: ImageData) -> Evidence:
    det, title = "image.cfa", "Camera demosaicing (CFA) consistency"
    if min(d.rgb.shape[:2]) < 128:
        return Evidence.not_applicable(det, title, Category.MANIPULATION, "image too small")
    ratio, texture, glob = cfa_map(d.rgb, CFA_BLOCK)
    info = {"global_ratio": glob, "threshold": CFA_RATIO_THRESHOLD}
    if glob < CFA_RATIO_THRESHOLD:
        why = ("JPEG compression at this quality usually erases them" if d.is_jpeg and
               (d.jpeg_quality or 100) < 92 else
               "the image may have been resized, rendered or generated rather than captured")
        return Evidence(
            detector=det, title=title, category=Category.MANIPULATION, score=0.0, confidence=0.05,
            summary=f"No global demosaicing pattern (ratio {glob:.3f}); {why}. Local CFA "
                    f"consistency cannot be checked.", details=info)
    # A generator's periodic imprint can mimic the 2-periodic CFA pattern.
    if not d.is_jpeg and min(d.rgb.shape[:2]) >= TILE:
        gray = cv2.cvtColor(d.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        peaks, excess = lattice_peaks(residual_spectrum(gray), 8)
        if lattice_statistic(non_camera_peaks(peaks), excess) > SPEC_STAT_THRESHOLD:
            return Evidence.not_applicable(det, title, Category.MANIPULATION,
                                           "periodic generator-like artefacts mask the CFA pattern",
                                           **info)
    # In a single-camera image the *same* green lattice is the acquired one everywhere, so
    # the block ratio stays above 1. A region whose ratio is clearly below 1 has the
    # opposite Bayer phase: it was demosaiced on a different grid, i.e. it comes from
    # another image (or was shifted by an odd number of pixels). Regions pasted with the
    # same phase are invisible to this test, which bounds its recall near one half.
    lr = np.log(np.maximum(ratio, 1e-3))
    valid = texture > np.percentile(texture, 20)
    z = np.where(valid, (-lr) / CFA_INVERSION_MARGIN * 2.0, 0.0)   # z > 2 <=> lr < -margin
    stat, comp, cinfo = cohesive_anomaly(z, thr=2.0, min_blocks=4)
    score = calibrate(stat, CFA_STAT_THRESHOLD, CFA_STAT_SOFTNESS)
    conf = float(np.clip((glob - 1) * 4, 0.3, 0.75))
    H, W = d.rgb.shape[:2]
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION, score=score, confidence=conf,
        summary=(f"The image carries camera demosaicing traces (ratio {glob:.3f}), but a compact "
                 f"region ({cinfo['frac'] * 100:.1f}% of blocks) has the opposite colour-filter "
                 f"phase: it was demosaiced on a different pixel grid, so it comes from another "
                 f"image or was moved." if score >= 0.5 else
                 f"Camera demosaicing traces are present with one consistent colour-filter phase "
                 f"(ratio {glob:.3f}). A region pasted with the same phase would not show here."),
        details={**info, "statistic": stat, "stat_threshold": CFA_STAT_THRESHOLD, **cinfo},
        heatmap=_inversion_heat(z, (H, W)) if score >= 0.5 else heat_from_z(z, (H, W), 1.5, 5.0) * 0.5,
        regions=regions_from_mask(comp, CFA_BLOCK, CFA_BLOCK, score, "opposite CFA phase")
        if score >= 0.5 else [])
