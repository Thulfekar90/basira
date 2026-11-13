"""Numerical and imaging helpers shared across engines."""
from __future__ import annotations

import base64
import io
import re
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibrate(x: float, threshold: float, softness: float) -> float:
    """Logistic map so that ``x == threshold`` -> 0.5.

    ``softness`` is the distance from the threshold that moves the score to
    ~0.73 (one logistic unit). Every detector exposes its raw statistic in
    ``details`` and converts it with this function, so thresholds stay
    auditable in one place per detector.
    """
    if not np.isfinite(x):
        return 0.0
    z = (x - threshold) / max(softness, 1e-9)
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -40, 40))))


def robust_z(values: np.ndarray, axis=None) -> np.ndarray:
    """Median / MAD z-score (robust to the anomaly we are looking for)."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).any():
        return np.zeros_like(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(values, axis=axis, keepdims=axis is not None)
        mad = np.nanmedian(np.abs(values - med), axis=axis, keepdims=axis is not None)
    return (values - med) / (1.4826 * mad + 1e-9)


def normalize01(a: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


# --------------------------------------------------------------------------- #
# Block processing
# --------------------------------------------------------------------------- #
def block_reduce(a: np.ndarray, block: int, fn=np.mean) -> np.ndarray:
    """Reduce a 2-D array over non-overlapping ``block x block`` tiles."""
    h, w = a.shape[:2]
    hb, wb = h // block, w // block
    if hb == 0 or wb == 0:
        return np.zeros((max(hb, 1), max(wb, 1)), dtype=np.float32)
    t = a[: hb * block, : wb * block].reshape(hb, block, wb, block)
    return fn(fn(t, axis=3), axis=1) if fn in (np.mean, np.max, np.min) else \
        np.apply_over_axes(fn, t, [1, 3]).reshape(hb, wb)


def upsample_to(a: np.ndarray, shape: Tuple[int, int], smooth: bool = True) -> np.ndarray:
    """Resize a low-resolution map to ``shape`` (H, W)."""
    import cv2

    interp = cv2.INTER_LINEAR if smooth else cv2.INTER_NEAREST
    out = cv2.resize(np.asarray(a, dtype=np.float32), (shape[1], shape[0]), interpolation=interp)
    return out


def to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim == 2:
        return rgb
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def largest_component_stats(mask: np.ndarray) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
    """Return (area fraction of largest connected component, bbox x,y,w,h)."""
    import cv2

    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return 0.0, None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return 0.0, None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[idx]
    return float(area) / m.size, (int(x), int(y), int(w), int(h))


def mask_regions(mask: np.ndarray, min_frac: float = 0.002, max_regions: int = 5
                 ) -> List[Tuple[int, int, int, int, float]]:
    """Bounding boxes of the biggest connected components: (x, y, w, h, area_frac)."""
    import cv2

    m = mask.astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        frac = area / m.size
        if frac >= min_frac:
            out.append((int(x), int(y), int(w), int(h), float(frac)))
    out.sort(key=lambda r: -r[4])
    return out[:max_regions]


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
_INFERNO = np.array([
    [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110], [147, 38, 103],
    [188, 55, 84], [221, 81, 58], [243, 120, 25], [252, 165, 10], [246, 215, 70],
    [252, 255, 164]], dtype=np.float32)


def colormap(a: np.ndarray) -> np.ndarray:
    """Map 0..1 float array to an inferno-like RGB uint8 image."""
    a = np.clip(np.nan_to_num(np.asarray(a, dtype=np.float32)), 0, 1)
    pos = a * (len(_INFERNO) - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(_INFERNO) - 1)
    t = (pos - i0)[..., None]
    rgb = _INFERNO[i0] * (1 - t) + _INFERNO[i1] * t
    return rgb.astype(np.uint8)


def overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a heatmap over an image; low values stay transparent."""
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, -1)
    rgb = rgb[..., :3].astype(np.float32)
    if heat.shape[:2] != rgb.shape[:2]:
        heat = upsample_to(heat, rgb.shape[:2])
    cm = colormap(heat).astype(np.float32)
    a = (alpha * np.clip(heat, 0, 1) ** 0.8)[..., None]
    gray = to_gray(rgb)[..., None] * np.ones((1, 1, 3), np.float32)
    base = 0.6 * gray + 0.4 * rgb
    return np.clip(base * (1 - a) + cm * a, 0, 255).astype(np.uint8)


def thumbnail(rgb: np.ndarray, max_side: int = 960) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return rgb
    import cv2

    return cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def png_b64(img: np.ndarray, max_side: int = 960) -> str:
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else \
            np.clip(img, 0, 255).astype(np.uint8)
    img = thumbnail(img, max_side)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def jpeg_b64(img: np.ndarray, max_side: int = 960, quality: int = 85) -> str:
    img = thumbnail(np.asarray(img).astype(np.uint8), max_side)
    buf = io.BytesIO()
    Image.fromarray(img).convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# Byte-level signature scanning (metadata, XMP, C2PA, encoders)
# --------------------------------------------------------------------------- #
# (pattern, label, kind) — kind: "ai" (generator), "edit" (editing tool),
# "c2pa" (content credentials), "transcode" (re-encoding pipeline)
SIGNATURES: List[Tuple[bytes, str, str]] = [
    # Content credentials / IPTC digital source type
    (rb"trainedAlgorithmicMedia", "IPTC DigitalSourceType = trainedAlgorithmicMedia", "ai"),
    (rb"compositeWithTrainedAlgorithmicMedia", "IPTC DigitalSourceType = compositeWithTrainedAlgorithmicMedia", "ai"),
    (rb"algorithmicMedia", "IPTC DigitalSourceType = algorithmicMedia", "ai"),
    (rb"compositeSynthetic", "IPTC DigitalSourceType = compositeSynthetic", "ai"),
    (rb"c2pa\.ai_generated|c2pa\.ai_training", "C2PA AI assertion", "ai"),
    (rb"jumb[\s\S]{0,64}c2pa|c2pa\.claim|c2pa\.manifest|urn:uuid:[0-9a-f-]{36}[\s\S]{0,40}c2pa", "C2PA manifest (JUMBF)", "c2pa"),
    # Image generators
    (rb"Stable Diffusion|stable-diffusion|StableDiffusion|sd_xl|SDXL", "Stable Diffusion", "ai"),
    (rb"ComfyUI|\"class_type\"\s*:", "ComfyUI workflow", "ai"),
    (rb"AUTOMATIC1111|Negative prompt:|Sampler: [A-Za-z0-9+ ]+, CFG scale", "AUTOMATIC1111 / SD WebUI parameters", "ai"),
    (rb"InvokeAI|invokeai_metadata|sd-metadata", "InvokeAI", "ai"),
    (rb"NovelAI", "NovelAI", "ai"),
    (rb"Midjourney|midjourney", "Midjourney", "ai"),
    (rb"DALL[\-\xb7 ]?E|dall-e|OpenAI", "OpenAI / DALL-E", "ai"),
    (rb"Adobe Firefly|firefly", "Adobe Firefly", "ai"),
    (rb"Imagen|Made with Google AI|SynthID", "Google Imagen / Gemini", "ai"),
    (rb"Bing Image Creator|Microsoft Designer", "Microsoft Designer / Bing", "ai"),
    (rb"Leonardo\.Ai|leonardo\.ai", "Leonardo.Ai", "ai"),
    (rb"ideogram", "Ideogram", "ai"),
    (rb"FLUX\.1|black-forest-labs", "FLUX (Black Forest Labs)", "ai"),
    (rb"Grok[ -]?(?:Imagine|\d)|xAI Aurora|\bx\.ai\b", "xAI Grok / Aurora", "ai"),
    # Video generators
    (rb"\bSora\b|OpenAI Sora", "OpenAI Sora", "ai"),
    (rb"RunwayML|Runway Gen|runwayml", "Runway", "ai"),
    (rb"Pika Labs|pika\.art", "Pika", "ai"),
    (rb"Kling ?AI|klingai|kling\.kuaishou", "Kling", "ai"),
    (rb"Google Veo|Veo [23]\b", "Google Veo", "ai"),
    (rb"Luma Dream Machine|lumalabs", "Luma", "ai"),
    (rb"HeyGen|Synthesia|d-id\.com", "Avatar generator (HeyGen / Synthesia / D-ID)", "ai"),
    # Voice generators
    (rb"ElevenLabs|elevenlabs", "ElevenLabs", "ai"),
    (rb"[Rr]esemble\.ai|Resemble AI", "Resemble AI", "ai"),
    (rb"Murf\.ai|PlayHT|play\.ht|WellSaid|Speechify", "Commercial TTS service", "ai"),
    # Editing software
    (rb"Adobe Photoshop", "Adobe Photoshop", "edit"),
    (rb"Adobe Lightroom|Lightroom", "Adobe Lightroom", "edit"),
    (rb"GIMP \d|Created with GIMP|gimp:|GIMP_", "GIMP", "edit"),
    (rb"Affinity Photo", "Affinity Photo", "edit"),
    (rb"Pixelmator", "Pixelmator", "edit"),
    (rb"Snapseed", "Snapseed", "edit"),
    (rb"Picsart|PicsArt", "Picsart", "edit"),
    (rb"\bCanva\b", "Canva", "edit"),
    (rb"FaceApp", "FaceApp", "edit"),
    (rb"Facetune", "Facetune", "edit"),
    (rb"Adobe Premiere|Premiere Pro", "Adobe Premiere", "edit"),
    (rb"After Effects", "Adobe After Effects", "edit"),
    (rb"DaVinci Resolve", "DaVinci Resolve", "edit"),
    (rb"Final Cut", "Final Cut Pro", "edit"),
    (rb"CapCut|capcut", "CapCut", "edit"),
    (rb"iMovie", "iMovie", "edit"),
    (rb"Audacity", "Audacity", "edit"),
    (rb"Adobe Audition", "Adobe Audition", "edit"),
    (rb"REAPER|Reaper", "REAPER", "edit"),
    # Transcoders (weak: re-encoding, not necessarily manipulation)
    (rb"Lavf\d+|Lavc\d+|libavformat", "FFmpeg (libavformat)", "transcode"),
    (rb"HandBrake", "HandBrake", "transcode"),
    (rb"LAME\d|LAME3", "LAME MP3 encoder", "transcode"),
]


def scan_signatures(data: bytes, kinds: Optional[Iterable[str]] = None) -> List[Dict[str, str]]:
    """Find known generator / editor / provenance markers in raw bytes."""
    found: Dict[str, Dict[str, str]] = {}
    allowed = set(kinds) if kinds else None
    for pat, label, kind in SIGNATURES:
        if allowed and kind not in allowed:
            continue
        m = re.search(pat, data)
        if m and label not in found:
            lo = max(0, m.start() - 24)
            ctx = data[lo: m.end() + 40]
            found[label] = {"label": label, "kind": kind, "offset": str(m.start()),
                            "context": _printable(ctx)}
    return list(found.values())


def _printable(b: bytes) -> str:
    s = b.decode("latin-1", "replace")
    return "".join(ch if 32 <= ord(ch) < 127 else "·" for ch in s)


def read_head_tail(path: str, head: int = 4 << 20, tail: int = 2 << 20) -> bytes:
    """Read the first and last few MB — metadata lives there for every container we handle."""
    import os

    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size <= head + tail:
            return f.read()
        a = f.read(head)
        f.seek(size - tail)
        return a + f.read()


def iptc_source_types(data: bytes) -> List[str]:
    """Extract IPTC digital-source-type codes (used by C2PA and XMP)."""
    codes = re.findall(rb"digitalsourcetype/([A-Za-z]+)", data)
    return sorted({c.decode("ascii", "ignore") for c in codes})
