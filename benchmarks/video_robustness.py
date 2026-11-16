"""Video robustness matrix: real footage that you supply, forgeries with ground truth,
and the post-processing every shared video goes through.

    python benchmarks/video_robustness.py --source A.mov [--source B.mov ...]
                                          [--quick] [--work DIR] [--jobs N] [--json OUT]
    python benchmarks/video_robustness.py            # no footage: synthetic fallback

Data
----
Nothing is bundled and nothing is downloaded. ``--source`` videos are only read; every
derived file goes to ``--work`` (a temporary directory by default, removed at the end).
A ``--work`` directory you name is kept, and a second run with the same directory reuses
its clips and only repeats the analysis (minutes instead of the half hour it takes to cut
and encode about 600 files from two 4K sources).

*Camera-like clips.* From each source, 8 s segments are cut at three positions and turned
into what a camera would have written: ordinary frame rate (a 240 fps source is
decimated, averaging half of each frame interval to imitate a 180 degree shutter),
1080p or 720p, 30 or 25 fps, H.264 with or without B-frames, one key frame per 1-2 s.
Three framings per source: the full picture, a 2x detail crop (looks like another shot),
and a slow virtual pan (the crop window travels across the real picture).

*Forgeries.* The camera file is decoded again (as an editor would), edited, and exported
(H.264 CRF 20). Ground truth is known exactly:

    del15    0.5 s removed mid-shot              del5     0.17 s removed mid-shot
    replay   1 s of earlier footage shown again  freeze   0.6 s of frozen picture
    insert   1.5 s from another source cut in    speed    2 s played at double speed

*Natural events* (must NOT be reported): hard cuts between shots, cuts between two
similar-looking shots, dissolve, fade through black, photo flashes, auto-exposure ramp,
whip pan, pan that starts abruptly, a knock against the camera, 4x slow motion, slow
motion made by showing each frame twice, 24->30 fps pulldown; and three synthetic ones:
locked-off shot with one object that moves, stops and leaves; a screen recording with
long static stretches, scrolling and page changes; periodic motion.

*Delivery conditions*, applied to every clean, natural and forged clip (libx264,
preset veryfast, no B-frames unless stated):

    orig     as exported                         crf23/28/35  re-encoded at that CRF
    half     downscaled to 50 %, CRF 23          fps-a    30->25 fps (25->30 for 25 fps clips)
    fps-b    ->24 fps                            bframes  re-encoded with 3 B-frames
    msg      "messaging app": 480p, CRF 32, one key frame per 2 s

Reported
--------
* clean and natural clips: how many are flagged (verdict level >= 1) or carry any
  suspicious time range. Both count as false positives.
* forged clips: how many are located, i.e. a reported time range has an edge within
  0.25 s of a true edit boundary (ranges covering more than half the clip do not count),
  and how many reported ranges lie nowhere near an edit ("stray").

A forged clip in the ``orig`` column has been compressed twice and a clean one once; in
every other column both have been re-encoded, which is the fair comparison.
Only the picture is analysed (``analyze_audio_track=False``).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from basira.core.io import ffmpeg_exe  # noqa: E402

TOL = 0.25                      # seconds: how close a reported range must be to the truth
CLIP_SECONDS = 8.0
CONDITIONS = ("orig", "crf23", "crf28", "crf35", "half", "fps-a", "fps-b", "bframes", "msg")
QUICK_CONDITIONS = ("orig", "crf28", "fps-a", "msg")
FORGERIES = ("del15", "del5", "replay", "freeze", "insert", "speed")
# (short side, fps, B-frames, key-frame interval in seconds, CRF) of the simulated cameras
CAMERAS = ((1080, 30, 0, 1.0, 18), (720, 30, 2, 2.0, 20), (720, 25, 0, 1.0, 19),
           (720, 30, 0, 1.0, 20), (1080, 25, 2, 2.0, 19), (720, 30, 2, 1.0, 18))
FF = ["-hide_banner", "-loglevel", "error", "-nostdin", "-y"]


# --------------------------------------------------------------------------- frame I/O
def _run(cmd: List[str], data: Optional[bytes] = None) -> bytes:
    p = subprocess.run(cmd, input=data, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[-600:])
    return p.stdout


def _frames_from_pipe(cmd: List[str], w: int, h: int, n: int) -> np.ndarray:
    """Read up to n raw I420 frames from an ffmpeg command writing to stdout."""
    size = w * h * 3 // 2
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Read frame by frame: a single read(size * n) asks for size * n bytes up front, which
    # is hundreds of GB for the default n and raises MemoryError on Linux.
    chunks: List[bytes] = []
    for _ in range(n):
        frame = p.stdout.read(size)
        if len(frame) < size:
            break
        chunks.append(frame)
    buf = b"".join(chunks)
    p.stdout.close()
    p.kill()
    p.wait()
    k = len(buf) // size
    if k < 2:
        raise RuntimeError("could not decode frames: " + p.stderr.read().decode("utf-8", "replace")[-300:])
    return np.frombuffer(buf[:k * size], np.uint8).reshape(k, h * 3 // 2, w).copy()


def decode_file(path: str, w: int, h: int, n: int = 10 ** 6) -> np.ndarray:
    return _frames_from_pipe([ffmpeg_exe(), *FF, "-i", path, "-f", "rawvideo", "-pix_fmt",
                              "yuv420p", "-"], w, h, n)


def encode(frames, path: str, fps: int, crf: int = 20, gop: Optional[int] = None,
           bf: int = 0) -> str:
    """Encode an iterable of I420 frames (each (h*3/2, w) uint8)."""
    it = iter(frames)
    first = next(it)
    h, w = first.shape[0] * 2 // 3, first.shape[1]
    cmd = [ffmpeg_exe(), *FF, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r",
           str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
           "-crf", str(crf), "-bf", str(bf)]
    if gop:
        cmd += ["-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"]
    p = subprocess.Popen(cmd + [path], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        p.stdin.write(first.tobytes())
        for f in it:
            p.stdin.write(f.tobytes())
        p.stdin.close()
    except BrokenPipeError:
        pass
    err = p.stderr.read()
    if p.wait() != 0:
        raise RuntimeError(err.decode("utf-8", "replace")[-400:])
    return path


def deliver(src: str, dst: str, cond: str, fps: int) -> str:
    """Re-encode ``src`` the way a platform or a messaging app would."""
    if cond == "orig":
        return src
    vf, crf, bf, extra = None, 23, 0, []
    if cond.startswith("crf"):
        crf = int(cond[3:])
    elif cond == "half":
        vf = "scale=trunc(iw/4)*2:trunc(ih/4)*2"
    elif cond == "fps-a":
        vf = "fps=%d" % (25 if fps == 30 else 30)
    elif cond == "fps-b":
        vf = "fps=24"
    elif cond == "bframes":
        bf = 3
    elif cond == "msg":
        vf, crf = "scale=-2:480", 32
        extra = ["-g", str(2 * fps), "-keyint_min", str(2 * fps), "-sc_threshold", "0"]
    else:
        raise ValueError(cond)
    cmd = [ffmpeg_exe(), *FF, "-i", src]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", str(crf),
            "-bf", str(bf), "-an", *extra, dst]
    _run(cmd)
    return dst


# --------------------------------------------------------------------------- sources
def _even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


class RealSource:
    """A real video file; segments are cut out with ffmpeg and never stored in the repo."""

    def __init__(self, path: str):
        self.path, self.name = path, os.path.splitext(os.path.basename(path))[0]
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.w, self.h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / max(self.fps, 1e-6)
        cap.release()

    def size(self, short: int) -> Tuple[int, int]:
        if self.w >= self.h:
            return _even(short * self.w / self.h), _even(short)
        return _even(short), _even(short * self.h / self.w)

    def segment(self, start: float, seconds: float, fps: int, size: Tuple[int, int],
                mode: str = "full", slow: int = 1) -> np.ndarray:
        """``seconds`` of footage starting at ``start`` as I420 frames at ``fps``.

        A source that is much faster than ``fps`` is decimated by an integer step and half
        of each step is averaged (motion blur of a 180 degree shutter); the time base is
        then relabelled, which can change the speed by a few per cent (240 -> 25 fps uses
        every 10th frame). ``slow`` > 1 keeps more source frames: real slow motion.
        """
        w, h = size
        step = max(1, int(round(self.fps / fps / slow)))
        n = int(round(seconds * fps))
        total = n * step                       # source frames consumed
        cw, ch = _even(self.w / 2), _even(self.h / 2)
        x0, y0 = int(self.w * 0.04), int(self.h * 0.46)
        mid, vf = total // 2, []
        if mode == "detail":
            vf.append(f"crop={cw}:{ch}:{x0}:{y0}")
        elif mode == "detail2":
            vf.append(f"crop={cw}:{ch}:{self.w - cw - x0}:{int(self.h * 0.06)}")
        elif mode == "pan":                    # slow constant pan across the picture
            v = (self.w - cw) * 0.9 / total
            vf.append(f"crop={cw}:{ch}:{x0 // 2}+{v:.5f}*n:{y0}")
        elif mode == "whip":                   # 0.4 s whip pan in the middle of the clip
            dn = max(2, int(0.4 * fps * step))
            u = f"clip((n-{mid})/{dn}\\,0\\,1)"
            vf.append(f"crop={cw}:{ch}:{x0 // 2}+{(self.w - cw) * 0.9:.1f}*(3*pow({u}\\,2)-2*pow({u}\\,3)):{y0}")
        elif mode == "panstart":               # locked off, then the pan starts abruptly
            v = (self.w - cw) * 0.9 / (total - mid)
            vf.append(f"crop={cw}:{ch}:{x0 // 2}+{v:.5f}*max(0\\,n-{mid}):{y0}")
        elif mode == "bump":                   # a knock: sudden jolt, damped 8 Hz wobble
            a, tau, per = self.h * 0.03, 0.12 * fps * step, fps * step / 8.0
            vf.append(f"crop={cw}:{ch}:{x0}:{y0 - int(a) - 2}+{a:.1f}*gte(n\\,{mid})*"
                      f"exp(-(n-{mid})/{tau:.2f})*cos(2*PI*(n-{mid})/{per:.2f})")
        elif mode != "full":
            raise ValueError(mode)
        if step >= 2:
            vf.append(f"tmix=frames={max(1, step // 2)}")
            vf.append(f"select=not(mod(n\\,{step}))")
        vf.append(f"scale={w}:{h}:flags=area:force_original_aspect_ratio=increase")
        vf.append(f"crop={w}:{h}")
        cmd = [ffmpeg_exe(), *FF, "-ss", f"{start:.3f}", "-i", self.path, "-vf", ",".join(vf),
               "-fps_mode", "passthrough", "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt",
               "yuv420p", "-"]
        return _frames_from_pipe(cmd, w, h, n)


class SynthSource:
    """Fallback when no footage is given: the generator of ``tests/synth_video.py``."""

    def __init__(self, seed: int):
        self.seed, self.name, self.fps, self.duration = seed, f"synthetic-{seed}", 30.0, 60.0
        self.w, self.h = 640, 360

    def size(self, short: int) -> Tuple[int, int]:
        return 640, 360

    def segment(self, start: float, seconds: float, fps: int, size: Tuple[int, int],
                mode: str = "full", slow: int = 1) -> np.ndarray:
        import synth_video as V

        n = int(round(seconds * fps))
        mid = n // 2
        seed = self.seed + int(start) * 7 + {"detail": 100, "detail2": 200}.get(mode, 0)
        speed = 1.6 / slow
        path = {
            "whip": lambda t: 20 + 900 * (3 * np.clip((t - mid) / 12, 0, 1) ** 2
                                          - 2 * np.clip((t - mid) / 12, 0, 1) ** 3),
            "panstart": lambda t: 20 + 3.0 * max(0, t - mid),
            "bump": lambda t: 200.0,
        }.get(mode, lambda t: 20 + speed * t)
        jolt = (lambda t: 12.0 * np.exp(-(t - mid) / 3.6) * np.cos(2 * np.pi * (t - mid) / 3.75)
                if t >= mid else 0.0) if mode == "bump" else None
        fr = V.make_frames(n, seed, camera=path, jolt=jolt, object_speed=1.0 / slow)
        return np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2YUV_I420) for f in fr])


# --------------------------------------------------------------------------- pixel helpers
def _luma(f: np.ndarray) -> np.ndarray:
    return f[:f.shape[0] * 2 // 3]


def gain(f: np.ndarray, g: float) -> np.ndarray:
    out = f.copy()
    y = _luma(out)
    y[:] = np.clip((y.astype(np.float32) - 16) * g + 16, 0, 255).astype(np.uint8)
    return out


def blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return np.clip(a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t + 0.5, 0, 255
                   ).astype(np.uint8)


def black_like(f: np.ndarray) -> np.ndarray:
    out = np.full_like(f, 128)
    _luma(out)[:] = 16
    return out


# --------------------------------------------------------------------------- forgeries
def forge(kind: str, a: np.ndarray, other: np.ndarray, fps: int):
    """Returns (list of frames, [(t0, t1) true edit boundaries in seconds])."""
    n = len(a)
    idx: List[Any] = list(range(n))
    if kind in ("del15", "del5"):
        k, cut = int(round((0.5 if kind == "del15" else 1 / 6.0) * fps)), n // 2
        idx = idx[:cut] + idx[cut + k:]
        truth = [(cut / fps, cut / fps)]
    elif kind == "replay":
        src, dst, ln = int(0.30 * n), int(0.62 * n), fps
        idx[dst:dst + ln] = range(src, src + ln)
        truth = [(dst / fps, (dst + ln) / fps)]
    elif kind == "freeze":
        at, ln = int(0.42 * n), int(round(0.6 * fps))
        idx[at:at + ln] = [at] * ln
        truth = [(at / fps, (at + ln) / fps)]
    elif kind == "insert":
        at, ln = int(0.40 * n), min(len(other), int(round(1.5 * fps)))
        idx[at:at + ln] = [-(i + 1) for i in range(ln)]          # negative: from ``other``
        truth = [(at / fps, (at + ln) / fps)]
    elif kind == "speed":
        at, ln = int(0.35 * n), 2 * fps
        idx = idx[:at] + idx[at:at + ln:2] + idx[at + ln:]
        truth = [(at / fps, (at + ln // 2) / fps)]
    else:
        raise ValueError(kind)
    return [a[i] if i >= 0 else other[-i - 1] for i in idx], truth


# --------------------------------------------------------------------------- natural events
def natural_from_shots(kind: str, s: Dict[str, np.ndarray], fps: int) -> List[np.ndarray]:
    """Legitimate events built from shots ``s`` (keys: a, b = different sources, c, d = two
    similar-looking framings of the first source)."""
    a, b = s["a"], s["b"]
    third = len(a) // 3
    if kind == "cuts":                       # three shots, hard cuts
        return list(a[:third]) + list(b[:third]) + list(s["c"][:third])
    if kind == "cuts-similar":               # same place, same light, different framing
        return list(s["c"][:third]) + list(s["d"][:third]) + list(a[third:2 * third])
    if kind == "dissolve":
        ln = fps // 2
        mix = [blend(a[third + i], b[i], (i + 1) / (ln + 1)) for i in range(ln)]
        return list(a[:third]) + mix + list(b[ln:ln + third])
    if kind == "fade-black":
        ln, k = fps // 3, black_like(a[0])
        out_ = [blend(a[third + i], k, (i + 1) / ln) for i in range(ln)]
        in_ = [blend(k, b[i], (i + 1) / ln) for i in range(ln)]
        return list(a[:third]) + out_ + [k] * (fps // 5) + in_ + list(b[ln:ln + third])
    if kind == "flashes":                    # photo flashes: 1 frame, 2 frames, decaying
        out = [f for f in a]
        n = len(out)
        out[n // 4] = gain(out[n // 4], 1.9)
        for j in (0, 1):
            out[n // 2 + j] = gain(out[n // 2 + j], 1.7)
        for j, g in enumerate((2.0, 1.5, 1.2)):
            out[3 * n // 4 + j] = gain(out[3 * n // 4 + j], g)
        return out
    if kind == "exposure":                   # auto-exposure settles to a brighter level
        n, ln = len(a), fps // 2
        return [gain(f, 1.0 + 0.3 * float(np.clip((i - n // 2) / ln, 0, 1))) for i, f in enumerate(a)]
    if kind == "dup-slowmo":                 # 50 % slow motion by showing each frame twice
        return [f for f in a[:len(a) // 2] for _ in (0, 1)]
    if kind == "pulldown":                   # 24 -> 30 fps: every 4th frame shown twice
        return [f for i, f in enumerate(a[:len(a) * 4 // 5]) for _ in range(2 if i % 4 == 3 else 1)]
    raise ValueError(kind)


def _noisy(bg: np.ndarray, rng: np.random.RandomState, sigma: float) -> np.ndarray:
    out = bg.copy()
    y = _luma(out)
    y[:] = np.clip(y + rng.randn(*y.shape) * sigma, 0, 255).astype(np.uint8)
    return out


def tripod_object(bg: np.ndarray, fps: int, seconds: float = 8.0, seed: int = 0) -> List[np.ndarray]:
    """Locked-off shot (one still + fresh sensor noise per frame). An object enters, crosses
    half the picture, stands still for 2 s, walks on and leaves; then nothing moves."""
    rng = np.random.RandomState(seed)
    n, (h, w) = int(seconds * fps), (bg.shape[0] * 2 // 3, bg.shape[1])
    size = h // 5
    tex = cv2.GaussianBlur(rng.rand(size, size).astype(np.float32) * 255, (0, 0), 2)
    tex = np.clip((tex - tex.mean()) * 5 + 90, 0, 255)
    walk = 1.2 * fps                                   # frames for each walking stretch
    frames = []
    for t in range(n):
        f = _noisy(bg, rng, 2.0)
        if t < walk:
            x = -size + (w // 2 + size) * t / walk
        elif t < walk + 2 * fps:
            x = w // 2
        else:
            x = w // 2 + (w // 2 + size) * (t - walk - 2 * fps) / walk
        x = int(x)
        x0, x1 = max(0, x), min(w, x + size)
        if x1 > x0:
            y0 = h // 2
            f[y0:y0 + size, x0:x1] = np.clip(tex[:, x0 - x:x1 - x] + rng.randn(size, x1 - x0) * 2,
                                             0, 255).astype(np.uint8)
        frames.append(f)
    return frames


def screen_recording(fps: int = 30, seconds: float = 10.0, seed: int = 0, w: int = 1280,
                     h: int = 720) -> List[np.ndarray]:
    """A desktop capture: pages of text-like bars, long stretches where nothing changes,
    a moving pointer, typing, smooth scrolling and instant page changes."""
    rng = np.random.RandomState(seed)

    def page(k: int) -> np.ndarray:
        r = np.random.RandomState(seed * 100 + k)
        y = np.full((h * 3, w), 242, np.uint8)
        for row in range(40, h * 3 - 20, 26):
            x = 60
            while x < w - 120 and r.rand() > 0.06:
                ln = r.randint(20, 110)
                y[row:row + 12, x:x + ln] = r.randint(30, 90)
                x += ln + 14
        return y

    pages = [page(k) for k in range(3)]
    cur, scroll, typed = 0, 0, 0
    px, py = w // 2, h // 2
    n = int(seconds * fps)
    events = {int(0.20 * n): "scroll", int(0.35 * n): "pointer", int(0.50 * n): "page",
              int(0.62 * n): "type", int(0.80 * n): "scroll", int(0.90 * n): "page"}
    todo: List[str] = []
    frames = []
    for t in range(n):
        if t in events:
            ev = events[t]
            if ev == "page":
                cur, scroll, typed = (cur + 1) % 3, 0, 0
            else:
                todo = [ev] * {"scroll": 8, "pointer": 20, "type": 30}[ev]
        if todo:
            ev = todo.pop()
            if ev == "scroll":
                scroll = min(scroll + 30, 2 * h)
            elif ev == "pointer":
                px, py = px + 9, py - 5
            elif ev == "type" and len(todo) % 3 == 0:
                typed += 1
        y = pages[cur][scroll:scroll + h].copy()
        y[:28] = 70                                              # title bar
        y[h - 40:h - 28, 60:60 + 11 * typed] = 40                # typed text
        y[py:py + 14, px:px + 9] = 0                             # pointer
        f = np.full((h * 3 // 2, w), 128, np.uint8)
        f[:h] = y
        frames.append(f)
    return frames


def periodic_motion(bg: np.ndarray, fps: int, seconds: float = 8.0, seed: int = 0) -> List[np.ndarray]:
    """Locked-off shot of something that repeats: a pendulum and a blinking light."""
    rng = np.random.RandomState(seed)
    n, (h, w) = int(seconds * fps), (bg.shape[0] * 2 // 3, bg.shape[1])
    r = h // 9
    frames = []
    for t in range(n):
        f = _noisy(bg, rng, 2.0)
        y = _luma(f)
        cx = int(w / 2 + (w / 4) * np.sin(2 * np.pi * t / fps))          # 1 s period
        cv2.circle(y, (cx, h // 2), r, 40, -1, cv2.LINE_AA)
        if (t // (fps // 2)) % 2 == 0:
            cv2.circle(y, (w // 6, h // 5), r // 2, 250, -1, cv2.LINE_AA)
        frames.append(f)
    return frames


# --------------------------------------------------------------------------- build + analyse
def _analyse(path: str) -> Dict[str, Any]:
    from basira.video import analyze_video

    t0 = time.time()
    r = analyze_video(path, analyze_audio_track=False)
    return {"level": int(r.verdict_level), "manip": float(r.scores.get("manipulation", 0.0)),
            "prov": float(r.scores.get("provenance", 0.0)),
            "segments": [(float(s.start), float(s.end), s.label) for s in r.fused_segments],
            "det": {e.detector: (round(e.score, 3), round(e.confidence, 2),
                                 [(round(s.start, 3), round(s.end, 3)) for s in e.segments])
                    for e in r.evidences},
            "duration": float(r.info.get("duration", 0.0)), "elapsed": time.time() - t0}


def located(segments, truth, duration: float) -> Tuple[bool, int]:
    """(some range edge within TOL of a true boundary, number of ranges far from any)."""
    hit, stray = False, 0
    for s0, s1, _ in segments:
        near = False
        for t0, t1 in truth:
            edge = min(abs(s0 - t0), abs(s0 - t1), abs(s1 - t0), abs(s1 - t1)) <= TOL
            inside = s0 - TOL <= t0 and t1 <= s1 + TOL
            if (edge or inside) and (s1 - s0) <= 0.5 * max(duration, 1e-6):
                hit = near = True
            elif s0 - 1.0 <= t1 and t0 <= s1 + 1.0:
                near = True
        stray += not near
    return hit, stray


def run_item(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Worker: build one clip (clean / natural / forged), deliver it under every
    condition, analyse each file."""
    work, fps, name = job["work"], job["fps"], job["name"]
    out = []
    try:
        truth = [tuple(t) for t in job.get("truth", [])]
        base = job.get("camera_file") or os.path.join(work, name + ".mp4")
        if not os.path.exists(base):
            a = np.load(job["frames_npy"], mmap_mode="r")
            if job["kind"] == "natural":
                encode(a, base, fps, crf=job["crf"], gop=job["gop"], bf=job["bf"])
            else:
                frames, _ = forge(job["what"], a, np.load(job["other_npy"], mmap_mode="r"), fps)
                encode(frames, base, fps, crf=20, bf=job["bf"])    # an editor's export
        for cond in job["conditions"]:
            path = os.path.join(work, f"{name}__{cond}.mp4") if cond != "orig" else base
            if not os.path.exists(path):
                deliver(base, path, cond, fps)
            res = _analyse(path)
            hit, stray = located(res["segments"], truth, res["duration"])
            out.append({"clip": job["clip"], "source": job["source"], "kind": job["kind"],
                        "what": job["what"], "cond": cond, "truth": truth, "located": hit,
                        "stray": stray, "path": path, **res})
            if path != base and not job.get("keep"):
                os.remove(path)
    except Exception as e:                    # one broken clip must not hide the others
        out.append({"clip": job["clip"], "source": job["source"], "kind": job["kind"],
                    "what": job["what"], "cond": "ERROR", "error": repr(e)[:300]})
    return out


def _camera_roundtrip(frames: np.ndarray, path: str, fps: int, crf: int, gop: int, bf: int
                      ) -> np.ndarray:
    encode(frames, path, fps, crf=crf, gop=gop, bf=bf)
    h, w = frames.shape[1] * 2 // 3, frames.shape[2]
    return decode_file(path, w, h)


def build_jobs(sources: Sequence[Any], work: str, conditions: Sequence[str], quick: bool,
               log=print) -> List[Dict[str, Any]]:
    """Cut the camera-like clips (sequentially: decoding the sources is the slow part)
    and describe every clean / natural / forged clip as a job for the worker pool."""
    jobs: List[Dict[str, Any]] = []
    synthetic = isinstance(sources[0], SynthSource)
    modes = ("full", "detail", "pan")
    k = 0
    for si, src in enumerate(sources):
        oth = sources[(si + 1) % len(sources)]
        starts = np.linspace(0.06, 0.80, 3) * max(src.duration - CLIP_SECONDS - 1, 1.0)
        for j, start in enumerate(starts[:1] if quick else starts):
            short, fps, bf, gop_s, crf = CAMERAS[k % len(CAMERAS)]
            size = src.size(short)
            clip = f"c{k}"
            log(f"  clip {clip}: {src.name} @{start:.1f}s {modes[j]} {size[0]}x{size[1]} "
                f"{fps}fps bf={bf}")
            raw = src.segment(float(start), CLIP_SECONDS, fps, size, modes[j])
            cam = os.path.join(work, f"{clip}_clean.mp4")
            dec = _camera_roundtrip(raw, cam, fps, crf, int(gop_s * fps), bf)
            # The inserted stretch comes from another source with its own camera encoding.
            o_start = float(0.5 * max(oth.duration - 3.0, 1.0)) if oth is not src else \
                float((start + 0.5 * src.duration) % max(src.duration - 3.0, 1.0))
            o_raw = oth.segment(o_start, 2.0, fps, size, "full" if oth is not src else "detail2")
            o_dec = _camera_roundtrip(o_raw, os.path.join(work, f"{clip}_other.mp4"), fps, 23,
                                      fps, 0)
            npy, onpy = os.path.join(work, f"{clip}.npy"), os.path.join(work, f"{clip}_o.npy")
            np.save(npy, dec)
            np.save(onpy, o_dec)
            common = {"work": work, "fps": fps, "clip": clip, "source": src.name, "bf": bf,
                      "conditions": list(conditions), "frames_npy": npy, "other_npy": onpy}
            jobs.append({**common, "kind": "clean", "what": "clean", "name": f"{clip}_clean",
                         "camera_file": cam})
            for what in FORGERIES:
                truth = forge(what, dec[:, :2, :2], o_dec[:, :2, :2], fps)[1]
                jobs.append({**common, "kind": "forged", "what": what, "truth": truth,
                             "name": f"{clip}_{what}"})
            k += 1

    # ---- natural events ----
    log("  natural events")
    a_src, b_src = sources[0], sources[(1) % len(sources)]
    fps, short = 30, 720
    size = a_src.size(short)
    nat = {"work": work, "fps": fps, "conditions": list(conditions), "crf": 20, "gop": fps,
           "bf": 0, "kind": "natural"}
    da, db = max(a_src.duration - 10, 1.0), max(b_src.duration - 10, 1.0)

    def add(what: str, frames, source: str, tag: str = "") -> None:
        name = f"nat_{what}{tag}"
        npy = os.path.join(work, name + ".npy")
        np.save(npy, np.asarray(frames))
        jobs.append({**nat, "what": what, "name": name, "clip": name, "source": source,
                     "frames_npy": npy})

    shots = {"a": a_src.segment(0.15 * da, 6.0, fps, size, "full"),
             "b": b_src.segment(0.55 * db, 6.0, fps, size, "full" if b_src is not a_src else "detail2"),
             "c": a_src.segment(0.45 * da, 6.0, fps, size, "detail"),
             "d": a_src.segment(0.70 * da, 6.0, fps, size, "detail2")}
    label = "synthetic" if synthetic else "real"
    for what in ("cuts", "cuts-similar", "dissolve", "fade-black", "flashes", "exposure",
                 "dup-slowmo", "pulldown"):
        add(what, natural_from_shots(what, shots, fps), label)
    if not quick:
        if len(sources) > 1:                 # the same events on the second source
            shots2 = {"a": shots["b"], "b": shots["a"],
                      "c": b_src.segment(0.25 * db, 6.0, fps, size, "detail"),
                      "d": b_src.segment(0.80 * db, 6.0, fps, size, "detail2")}
            for what in ("cuts-similar", "flashes", "exposure", "pulldown"):
                add(what, natural_from_shots(what, shots2, fps), label, "_2")
        for si, src in enumerate(sources[:2]):
            d = max(src.duration - 10, 1.0)
            for mode in ("whip", "panstart", "bump"):
                add(mode, src.segment(0.3 * d, 6.0, fps, src.size(short), mode), label, f"_{si}")
            add("slowmo-4x", src.segment(0.6 * d, 6.0, fps, src.size(short), "full", slow=4),
                label, f"_{si}")
    still = shots["a"][0]
    add("tripod-object", tripod_object(still, fps), "synthetic motion on a " + label + " still")
    add("periodic", periodic_motion(still, fps), "synthetic motion on a " + label + " still")
    add("screen-recording", screen_recording(fps), "synthetic")
    return jobs


# --------------------------------------------------------------------------- report
def table(rows: List[Dict[str, Any]], conditions: Sequence[str]) -> str:
    lines = []
    errs = [r for r in rows if r["cond"] == "ERROR"]
    rows = [r for r in rows if r["cond"] != "ERROR"]
    head = f"{'':24s}" + "".join(f"{c:>9s}" for c in conditions)

    def cell(sel, fn) -> str:
        return "".join(f"{fn([r for r in sel if r['cond'] == c]):>9s}" for c in conditions)

    def frac(pred):
        return lambda rs: f"{sum(1 for r in rs if pred(r))}/{len(rs)}" if rs else "-"

    clean = [r for r in rows if r["kind"] == "clean"]
    lines += ["FALSE POSITIVES: clips flagged (verdict level >= 1 or any time range) / clips",
              head, f"{'clean, one shot':24s}" + cell(clean, frac(lambda r: r["level"] >= 1 or r["segments"]))]
    nat = [r for r in rows if r["kind"] == "natural"]
    for what in sorted({r["what"] for r in nat}):
        sel = [r for r in nat if r["what"] == what]
        lines.append(f"{what:24s}" + cell(sel, frac(lambda r: r["level"] >= 1 or r["segments"])))
    lines += ["", f"FORGERIES LOCATED within {TOL} s / clips", head]
    forged = [r for r in rows if r["kind"] == "forged"]
    for what in FORGERIES:
        sel = [r for r in forged if r["what"] == what]
        if sel:
            lines.append(f"{what:24s}" + cell(sel, frac(lambda r: r["located"])))
    lines += ["", "FORGED CLIPS with verdict 'Suspicious' or stronger (level >= 2) / clips", head]
    for what in FORGERIES:
        sel = [r for r in forged if r["what"] == what]
        if sel:
            lines.append(f"{what:24s}" + cell(sel, frac(lambda r: r["level"] >= 2)))
    lines += ["", "STRAY time ranges in forged clips (nowhere near the edit), total", head,
              f"{'all forgeries':24s}" + cell(forged, lambda rs: str(sum(r["stray"] for r in rs)) if rs else "-")]
    fired: Dict[str, int] = defaultdict(int)
    for r in clean + nat:
        for det, (score, conf, _) in r["det"].items():
            if score >= 0.5 and conf > 0:
                fired[det] += 1
    lines += ["", f"Detectors above threshold on the {len(clean) + len(nat)} clean/natural files: "
              + (", ".join(f"{d} x{n}" for d, n in sorted(fired.items())) or "none")]
    if rows:
        el = np.array([r["elapsed"] for r in rows])
        lines.append(f"Analysis time per file: median {np.median(el):.2f} s, max {el.max():.2f} s "
                     f"({len(rows)} files)")
    for e in errs[:10]:
        lines.append(f"ERROR {e['clip']} {e['what']}: {e['error']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", action="append", default=[], help="a real video (repeatable)")
    ap.add_argument("--quick", action="store_true", help="one clip per source, 4 conditions")
    ap.add_argument("--work", help="directory for derived files (kept); default: temporary")
    ap.add_argument("--jobs", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--json", help="write every per-file result to this JSON file")
    args = ap.parse_args(argv)
    if args.jobs < 1:
        ap.error("--jobs must be positive")
    if ffmpeg_exe() is None:
        print("ffmpeg is not available (pip install imageio-ffmpeg): nothing to measure.")
        return 1
    missing = [s for s in args.source if not os.path.isfile(s)]
    for s in missing:
        print(f"ERROR requested source is not a file: {s}")
    if missing:
        print(f"INCOMPLETE: {len(missing)} requested source(s) missing; no substitute corpus used.")
        return 1
    paths = args.source
    if paths:
        sources: List[Any] = [RealSource(p) for p in paths]
        print(f"Real footage: {len(sources)} source(s): " + ", ".join(
            f"{s.name} ({s.w}x{s.h}, {s.fps:.0f} fps, {s.duration:.0f} s)" for s in sources))
    else:
        sources = [SynthSource(0), SynthSource(50)]
        print("No --source given: falling back to the SYNTHETIC generator (tests/synth_video.py).\n"
              "These numbers say little about real footage.")
    conditions = QUICK_CONDITIONS if args.quick else CONDITIONS
    work = args.work or tempfile.mkdtemp(prefix="basira_video_")
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    manifest = os.path.join(work, "jobs.json")
    config_path = os.path.join(work, "run-config.json")
    config = {"mode": "real" if paths else "synthetic", "quick": args.quick,
              "conditions": list(conditions),
              "sources": [{"path": os.path.abspath(p), "size": os.stat(p).st_size,
                           "mtime_ns": os.stat(p).st_mtime_ns} for p in paths]}
    try:
        if os.path.exists(manifest):         # --work DIR from an earlier run: reuse its clips
            try:
                with open(config_path) as f:
                    previous_config = json.load(f)
            except (OSError, ValueError):
                previous_config = None
            if previous_config != config:
                print("ERROR: cached corpus does not match this invocation (or lacks source metadata). "
                      "Use a new --work directory; no cached source was substituted.")
                return 1
            with open(manifest) as f:
                jobs = json.load(f)
            print(f"  reusing the clips in {work}")
        else:
            jobs = build_jobs(sources, work, conditions, args.quick)
            for j in jobs:
                j["keep"] = bool(args.work)
            with open(manifest, "w") as f:
                json.dump(jobs, f)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        print(f"  {len(jobs)} clips x {len(conditions)} conditions, {args.jobs} worker(s) "
              f"(built in {time.time() - t0:.0f} s)")
        if args.jobs > 1:
            import multiprocessing as mp

            with mp.get_context("spawn").Pool(args.jobs) as pool:
                rows = [r for part in pool.imap_unordered(run_item, jobs) for r in part]
        else:
            rows = [r for job in jobs for r in run_item(job)]
        for f in os.listdir(work):           # raw frames are only needed to build the clips
            if f.endswith(".npy"):
                os.remove(os.path.join(work, f))
    finally:
        if not args.work:
            shutil.rmtree(work, ignore_errors=True)
    print()
    print(table(rows, conditions))
    print(f"\nTotal {time.time() - t0:.0f} s")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f)
    errors = sum(1 for r in rows if "error" in r)
    if not rows or errors:
        print(f"INCOMPLETE: {errors} failed clip(s), {len(rows)} result row(s).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
