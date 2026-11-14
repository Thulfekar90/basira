"""Deterministic synthetic videos and temporal forgeries with ground truth."""
from __future__ import annotations

import os
import subprocess
from typing import List, Optional, Tuple

import cv2
import numpy as np

from basira.core.io import ffmpeg_exe

W, H, FPS = 640, 360, 30


def _background(seed: int, w: int = W * 3, h: int = H * 2) -> np.ndarray:
    rng = np.random.RandomState(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1
    out = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        field = np.real(np.fft.ifft2((rng.randn(h, w) + 1j * rng.randn(h, w)) / f ** 1.2))
        out[..., c] = (field - field.mean()) / field.std() * 40 + 118
    return np.clip(out, 0, 255)


def make_frames(n: int = 240, seed: int = 0, noise: float = 3.0, camera=None, jolt=None,
                object_speed: float = 1.0) -> List[np.ndarray]:
    """A slow camera pan over a textured scene with moving objects and sensor noise.

    ``camera(t)`` overrides the horizontal position of the camera (pixels) and ``jolt(t)``
    adds a vertical displacement, for whip pans, abrupt pan starts and knocks;
    ``object_speed`` scales the motion of the discs (slow motion).
    """
    rng = np.random.RandomState(seed)
    bg = _background(seed)
    objs = [(rng.randint(40, W - 40), rng.randint(40, H - 40), rng.uniform(-3, 3),
             rng.uniform(-2, 2), rng.randint(14, 36), tuple(int(v) for v in rng.randint(30, 230, 3)))
            for _ in range(5)]
    nrng = np.random.RandomState(seed + 7)
    frames = []
    for t in range(n):
        x0 = int(20 + 1.6 * t) if camera is None else int(camera(t))
        y0 = int(30 + 18 * np.sin(t / 40.0) + 30) + (int(jolt(t)) if jolt is not None else 0)
        x0 = int(np.clip(x0, 0, bg.shape[1] - W))
        y0 = int(np.clip(y0, 0, bg.shape[0] - H))
        fr = bg[y0:y0 + H, x0:x0 + W].copy()
        for (cx, cy, vx, vy, r, col) in objs:
            px = int((cx + vx * object_speed * t) % W)
            py = int((cy + vy * object_speed * t) % H)
            cv2.circle(fr, (px, py), int(r), col, -1, cv2.LINE_AA)
        fr += nrng.randn(H, W, 3).astype(np.float32) * noise
        frames.append(np.clip(fr, 0, 255).astype(np.uint8))
    return frames


def encode(frames: List[np.ndarray], path: str, gop: int = 30, crf: int = 20, bframes: int = 2,
           metadata: Optional[dict] = None, extra: Optional[List[str]] = None) -> str:
    exe = ffmpeg_exe()
    h, w = frames[0].shape[:2]
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0", "-bf", str(bframes),
           "-crf", str(crf), "-preset", "veryfast"]
    for k, v in (metadata or {}).items():
        cmd += ["-metadata", f"{k}={v}"]
    cmd += (extra or []) + [path]
    p = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[:400])
    return path


def reencode(src: str, dst: str, gop: int = 30, crf: int = 22, bframes: int = 2) -> str:
    exe = ffmpeg_exe()
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
           "-bf", str(bframes), "-crf", str(crf), "-preset", "veryfast", dst]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[:400])
    return dst


# ---- forgeries: return (path, [true event times in seconds]) ----
def make_clean(out: str, seed: int = 0) -> Tuple[str, List[float]]:
    return encode(make_frames(240, seed), os.path.join(out, f"clean_{seed}.mp4")), []


def make_deletion(out: str, seed: int = 1) -> Tuple[str, List[float]]:
    fr = make_frames(260, seed)
    cut = 120
    fr = fr[:cut] + fr[cut + 22:]
    return encode(fr, os.path.join(out, f"deletion_{seed}.mp4")), [cut / FPS]


def make_repeat(out: str, seed: int = 2) -> Tuple[str, List[float]]:
    fr = make_frames(240, seed)
    fr[150:180] = [f.copy() for f in fr[80:110]]       # replay an earlier half second
    return encode(fr, os.path.join(out, f"repeat_{seed}.mp4")), [150 / FPS]


def make_freeze(out: str, seed: int = 3) -> Tuple[str, List[float]]:
    fr = make_frames(240, seed)
    fr[100:118] = [fr[100].copy() for _ in range(18)]
    return encode(fr, os.path.join(out, f"freeze_{seed}.mp4")), [100 / FPS]


def make_insert(out: str, seed: int = 4) -> Tuple[str, List[float]]:
    """A segment from another camera (much noisier source) cut into the clip."""
    a = make_frames(240, seed, noise=2.0)
    b = make_frames(240, seed + 31, noise=9.0)
    a[100:160] = b[100:160]
    return encode(a, os.path.join(out, f"insert_{seed}.mp4")), [100 / FPS, 160 / FPS]


def make_double(out: str, seed: int = 5, gop1: int = 12, gop2: int = 30) -> Tuple[str, List[float]]:
    first = encode(make_frames(300, seed), os.path.join(out, f"_first_{seed}.mp4"), gop=gop1, crf=23)
    return reencode(first, os.path.join(out, f"double_{seed}.mp4"), gop=gop2), []


def make_tagged(out: str, seed: int = 6) -> Tuple[str, List[float]]:
    return encode(make_frames(90, seed), os.path.join(out, f"tagged_{seed}.mp4"),
                  metadata={"comment": "Generated with OpenAI Sora"}), []
