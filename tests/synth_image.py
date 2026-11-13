"""Deterministic synthetic 'camera' images and forgeries with ground-truth masks.

The scene model is a colour 1/f random field plus shapes and gradients. The
capture model is: Bayer RGGB mosaic -> additive sensor noise -> bilinear
demosaic -> JPEG with a standard libjpeg table. That gives every detector the
traces it looks for in real photos (CFA, a homogeneous noise floor, a JPEG
history), so forgeries are realistic tests rather than strawmen.
"""
from __future__ import annotations

import io
import os
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def scene(h: int = 768, w: int = 1024, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1
    img = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        spec = (rng.randn(h, w) + 1j * rng.randn(h, w)) / f ** 1.1
        field = np.real(np.fft.ifft2(spec))
        img[..., c] = (field - field.mean()) / field.std()
    mix = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]], np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    # Texture strength varies smoothly across the frame, like real scenes do
    # (sky vs foliage), so detectors cannot assume homogeneous texture.
    env = cv2.GaussianBlur(rng.rand(h, w).astype(np.float32), (0, 0), min(h, w) / 6)
    env = 0.25 + 1.5 * (env - env.min()) / (np.ptp(env) + 1e-6)
    img = (img @ mix.T) * (38 * env[..., None]) + np.array([120, 115, 105], np.float32)
    img += (yy / h * 40 - 20)[..., None]
    if rng.rand() < 0.5:                       # a smooth "sky" band
        sky = int(h * rng.uniform(0.15, 0.35))
        grad = np.linspace(200, 160, sky)[:, None, None] * np.array([0.8, 0.9, 1.0])
        a = np.clip((sky - yy[:sky]) / 40.0, 0, 1)[..., None]
        img[:sky] = img[:sky] * (1 - a) + grad * a
    for _ in range(14):
        c = tuple(int(v) for v in rng.randint(30, 225, 3))
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        if rng.rand() < 0.5:
            cv2.circle(img, (cx, cy), int(rng.randint(15, 90)), c, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (cx, cy), (cx + rng.randint(20, 160), cy + rng.randint(20, 120)),
                          c, -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return np.clip(img, 0, 255)


def capture(sc: np.ndarray, noise: float = 2.5, seed: int = 1) -> np.ndarray:
    """Bayer RGGB sampling + sensor noise + bilinear demosaic."""
    rng = np.random.RandomState(seed)
    h, w = sc.shape[:2]
    bayer = np.zeros((h, w), np.float32)
    bayer[0::2, 0::2] = sc[0::2, 0::2, 0]
    bayer[0::2, 1::2] = sc[0::2, 1::2, 1]
    bayer[1::2, 0::2] = sc[1::2, 0::2, 1]
    bayer[1::2, 1::2] = sc[1::2, 1::2, 2]
    bayer += rng.randn(h, w).astype(np.float32) * noise * np.sqrt(np.maximum(bayer, 1) / 120)
    b8 = np.clip(bayer, 0, 255).astype(np.uint8)
    # OpenCV names Bayer patterns by the 2nd row; RGGB corresponds to COLOR_BayerBG2RGB.
    return cv2.cvtColor(b8, cv2.COLOR_BayerBG2RGB)


def jpeg(rgb: np.ndarray, q: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def save_jpeg(rgb: np.ndarray, path: str, q: int, exif: bytes = None) -> str:
    kw = {"quality": q}
    if exif is not None:
        kw["exif"] = exif
    Image.fromarray(rgb).save(path, format="JPEG", **kw)
    return path


def camera_photo(seed: int = 0, h: int = 768, w: int = 1024, noise: float = 2.5) -> np.ndarray:
    return capture(scene(h, w, seed), noise=noise, seed=seed + 100)


# ----------------------------------------------------------------------------- #
# Forgeries — each returns (path, ground-truth mask). Everything is seed-driven.
# ----------------------------------------------------------------------------- #
def _box(rng, h, w, lo=130, hi=260):
    s = int(rng.randint(lo, hi))
    y = int(rng.randint(16, h - s - 16))
    x = int(rng.randint(16, w - s - 16))
    return y, x, s


def make_authentic(out: str, seed: int = 0) -> Tuple[str, np.ndarray]:
    rng = np.random.RandomState(1000 + seed)
    img = camera_photo(seed, noise=float(rng.choice([1.0, 2.5, 4.0])))
    q = int(rng.choice([72, 80, 85, 90, 95]))
    p = save_jpeg(img, os.path.join(out, f"authentic_{seed}.jpg"), q)
    return p, np.zeros(img.shape[:2], bool)


def make_resaved(out: str, seed: int = 0) -> Tuple[str, np.ndarray]:
    """Authentic but saved twice (e.g. downloaded and re-saved): must NOT be flagged."""
    rng = np.random.RandomState(2000 + seed)
    img = jpeg(camera_photo(seed + 300, noise=float(rng.choice([1.0, 2.5, 4.0]))),
               int(rng.choice([70, 78, 85])))
    p = save_jpeg(img, os.path.join(out, f"resaved_{seed}.jpg"), int(rng.choice([88, 92, 95])))
    return p, np.zeros(img.shape[:2], bool)


def make_splice(out: str, seed: int = 3) -> Tuple[str, np.ndarray]:
    rng = np.random.RandomState(3000 + seed)
    host = jpeg(camera_photo(seed, noise=2.5), int(rng.choice([70, 75, 80]))).copy()
    donor = jpeg(camera_photo(seed + 50, noise=float(rng.choice([0.6, 5.0]))), 95)
    h, w = host.shape[:2]
    y, x, s = _box(rng, h, w)
    dy, dx = int(rng.randint(0, h - s)), int(rng.randint(0, w - s))
    host[y:y + s, x:x + s] = donor[dy:dy + s, dx:dx + s]
    mask = np.zeros((h, w), bool)
    mask[y:y + s, x:x + s] = True
    return save_jpeg(host, os.path.join(out, f"splice_{seed}.jpg"), int(rng.choice([88, 92]))), mask


def make_copymove(out: str, seed: int = 5) -> Tuple[str, np.ndarray]:
    rng = np.random.RandomState(4000 + seed)
    img = jpeg(camera_photo(seed), 85).copy()
    h, w = img.shape[:2]
    for _ in range(100):
        sy, sx, s = _box(rng, h, w, 110, 190)
        ty, tx = int(rng.randint(16, h - s - 16)), int(rng.randint(16, w - s - 16))
        if abs(ty - sy) > s or abs(tx - sx) > s:
            break
    img[ty:ty + s, tx:tx + s] = img[sy:sy + s, sx:sx + s]
    mask = np.zeros((h, w), bool)
    mask[ty:ty + s, tx:tx + s] = True
    mask[sy:sy + s, sx:sx + s] = True
    return save_jpeg(img, os.path.join(out, f"copymove_{seed}.jpg"), 90), mask


def make_inpaint(out: str, seed: int = 7) -> Tuple[str, np.ndarray]:
    """Simulated generative fill: a region replaced by smooth, noise-free content."""
    rng = np.random.RandomState(5000 + seed)
    img = camera_photo(seed, noise=float(rng.choice([2.5, 4.0]))).copy()
    h, w = img.shape[:2]
    clean = scene(h, w, seed)                                  # same scene, no sensor noise
    y, x, s = _box(rng, h, w)
    patch = cv2.GaussianBlur(clean[y:y + s, x:x + s], (0, 0), 1.2)
    img[y:y + s, x:x + s] = np.clip(patch, 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), bool)
    mask[y:y + s, x:x + s] = True
    return save_jpeg(img, os.path.join(out, f"inpaint_{seed}.jpg"), int(rng.choice([90, 95]))), mask


def make_grid_shift(out: str, seed: int = 9) -> Tuple[str, np.ndarray]:
    rng = np.random.RandomState(6000 + seed)
    host = jpeg(camera_photo(seed), 72).copy()
    donor = jpeg(camera_photo(seed + 1), 72)
    h, w = host.shape[:2]
    y, x, s = _box(rng, h, w, 180, 280)
    oy, ox = int(rng.randint(1, 8)), int(rng.randint(1, 8))
    y, x = (y // 8) * 8 + oy, (x // 8) * 8 + ox                # grid shifted by (ox, oy)
    sy, sx = (y // 8) * 8, (x // 8) * 8
    host[y:y + s, x:x + s] = donor[sy:sy + s, sx:sx + s]
    mask = np.zeros((h, w), bool)
    mask[y:y + s, x:x + s] = True
    return save_jpeg(host, os.path.join(out, f"gridshift_{seed}.jpg"), 92), mask


def make_generated(out: str, seed: int = 11) -> Tuple[str, np.ndarray]:
    """A toy 'decoder' output: latent upsampled x8 through a fixed learned-like kernel."""
    rng = np.random.RandomState(seed)
    lat = cv2.GaussianBlur(scene(96, 128, seed), (0, 0), 0.6)
    up = cv2.resize(lat, (1024, 768), interpolation=cv2.INTER_CUBIC)
    kernel = 1 + 0.035 * rng.randn(8, 8, 3).astype(np.float32)   # checkerboard-like imprint
    tiled = np.tile(kernel, (768 // 8, 1024 // 8, 1))
    img = np.clip(up * tiled, 0, 255).astype(np.uint8)
    p = os.path.join(out, f"generated_{seed}.png")
    Image.fromarray(img).save(p)
    return p, np.ones(img.shape[:2], bool)


def make_sd_png(out: str, seed: int = 12) -> str:
    img = camera_photo(seed, 512, 512)
    meta = PngInfo()
    meta.add_text("parameters", "a photo of a cat\nNegative prompt: blurry\nSteps: 30, "
                  "Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1234, Size: 512x512, "
                  "Model: sd_xl_base_1.0")
    p = os.path.join(out, f"sd_params_{seed}.png")
    Image.fromarray(img).save(p, pnginfo=meta)
    return p


def iou(heat: np.ndarray, mask: np.ndarray, thr: float = 0.5) -> float:
    if heat is None:
        return 0.0
    if heat.shape != mask.shape:
        heat = cv2.resize(heat, (mask.shape[1], mask.shape[0]))
    pred = heat >= thr
    inter = (pred & mask).sum()
    union = (pred | mask).sum()
    return float(inter / union) if union else 0.0


def hit_ratio(heat: np.ndarray, mask: np.ndarray) -> float:
    """Mean heat inside the true mask divided by mean heat outside (localisation contrast)."""
    if heat is None:
        return 0.0
    if heat.shape != mask.shape:
        heat = cv2.resize(heat, (mask.shape[1], mask.shape[0]))
    return float(heat[mask].mean() / (heat[~mask].mean() + 1e-3))
