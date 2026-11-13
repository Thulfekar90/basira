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
from ..core.utils import calibrate, robust_z
from .common import cohesive_anomaly, heat_from_z, regions_from_mask
from .load import ImageData

TILE = 256
PEAK_THRESHOLD = 5.0       # dB above local background for a lattice peak to count
SPEC_STAT_THRESHOLD = 4.0
SPEC_STAT_SOFTNESS = 1.5
CFA_RATIO_THRESHOLD = 1.06   # mean|e| ratio between the two green lattices


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
        candidates = {k: v for k, v in peaks8.items() if abs(k[0]) != 4 and abs(k[1]) != 4}
    strong = {k: v for k, v in candidates.items() if v > PEAK_THRESHOLD}
    stat = float(np.mean(sorted(candidates.values(), reverse=True)[:6])) if candidates else 0.0
    # Off-lattice periodic spikes (any strong isolated peak), e.g. other upsampling factors.
    ex = excess.copy()
    c = ex.shape[0] // 2
    ex[c - 3:c + 4, c - 3:c + 4] = 0
    off_peak = float(np.percentile(ex, 99.97))

    ratio, texture, cfa_glob = cfa_map(d.rgb)
    camera_like = cfa_glob > CFA_RATIO_THRESHOLD
    score = calibrate(stat, SPEC_STAT_THRESHOLD, SPEC_STAT_SOFTNESS)
    if camera_like:
        score *= 0.6
    if d.is_jpeg:
        conf = 0.1
    elif d.lossless:
        conf = 0.5
    else:
        conf = 0.3
    spec_vis = np.clip((spec - np.percentile(spec, 1)) / (np.ptp(spec) + 1e-6) * 255, 0, 255)
    spec_vis = cv2.applyColorMap(spec_vis.astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    return Evidence(
        detector=det, title=title, category=Category.AI_GENERATION, score=score, confidence=conf,
        summary=(f"The noise residual has {len(strong)} sharp periodic peaks on the 1/8 "
                 f"frequency lattice (mean excess {stat:.1f} dB) — the signature of a "
                 f"generator's upsampling layers." if score >= 0.5 else
                 ("JPEG blocking occupies the same frequency lattice, so generator "
                  "periodicity cannot be measured reliably on this file."
                  if d.is_jpeg else
                  f"No generator-like periodic peaks in the residual spectrum (mean lattice "
                  f"excess {stat:.1f} dB < {SPEC_STAT_THRESHOLD} dB).")) +
                (" Camera demosaicing traces are present." if camera_like else ""),
        details={"lattice_excess_db": stat, "strong_peaks": len(strong),
                 "threshold_db": SPEC_STAT_THRESHOLD, "offlattice_p99.97_db": off_peak,
                 "jpeg_lattice_db": jpeg_strength, "cfa_ratio": cfa_glob,
                 "peaks": {f"{k[0]},{k[1]}": round(v, 2) for k, v in sorted(
                     peaks8.items(), key=lambda kv: -kv[1])[:12]}},
        visuals={"residual_spectrum": spec_vis})


def analyze_cfa(d: ImageData) -> Evidence:
    det, title = "image.cfa", "Camera demosaicing (CFA) consistency"
    if min(d.rgb.shape[:2]) < 128:
        return Evidence.not_applicable(det, title, Category.MANIPULATION, "image too small")
    ratio, texture, glob = cfa_map(d.rgb)
    info = {"global_ratio": glob, "threshold": CFA_RATIO_THRESHOLD}
    if glob < CFA_RATIO_THRESHOLD:
        why = ("JPEG compression at this quality usually erases them" if d.is_jpeg and
               (d.jpeg_quality or 100) < 92 else
               "the image may have been resized, rendered or generated rather than captured")
        return Evidence(
            detector=det, title=title, category=Category.MANIPULATION, score=0.0, confidence=0.05,
            summary=f"No global demosaicing pattern (ratio {glob:.3f}); {why}. Local CFA "
                    f"consistency cannot be checked.", details=info)
    # A generator's periodic imprint can mimic the 2-periodic CFA pattern: if the residual
    # spectrum shows non-camera lattice peaks, the CFA reading is not trustworthy.
    if not d.is_jpeg and min(d.rgb.shape[:2]) >= TILE:
        gray = cv2.cvtColor(d.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        peaks, _ = lattice_peaks(residual_spectrum(gray), 8)
        other = sorted((v for k, v in peaks.items() if abs(k[0]) != 4 and abs(k[1]) != 4),
                       reverse=True)[:6]
        if other and float(np.mean(other)) > SPEC_STAT_THRESHOLD:
            return Evidence.not_applicable(det, title, Category.MANIPULATION,
                                           "periodic generator-like artefacts mask the CFA pattern",
                                           **info)
    lr = np.log(np.maximum(ratio, 1e-3))
    valid = texture > np.percentile(texture, 20)
    z = -robust_z(np.where(valid, lr, np.nan))    # positive where CFA trace is missing
    z = np.where(valid, z, 0.0)
    stat, comp, cinfo = cohesive_anomaly(z, thr=2.5)
    score = calibrate(stat, 0.22, 0.08)
    conf = float(np.clip((glob - 1) * 4, 0.2, 0.7))
    H, W = d.rgb.shape[:2]
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION, score=score, confidence=conf,
        summary=(f"The image carries camera demosaicing traces (ratio {glob:.3f}) except in a "
                 f"compact region ({cinfo['frac'] * 100:.1f}% of blocks) where they vanish — "
                 f"that region was likely pasted, retouched or synthesised." if score >= 0.5 else
                 f"Camera demosaicing traces are present and spatially consistent "
                 f"(ratio {glob:.3f})."),
        details={**info, "statistic": stat, **cinfo},
        heatmap=heat_from_z(z, (H, W), 1.5, 5.0),
        regions=regions_from_mask(comp, 32, 32, score, "missing CFA trace") if score >= 0.5 else [])
