"""Temporal video forensics: one decode pass, then detectors on per-frame signals.

Signals per frame: a 96x54 gray thumbnail, the difference to the previous
frame, a gray histogram and a sensor-noise estimate. From these:

* ``video.freeze``      — runs of duplicated frames inside moving footage,
* ``video.repeat``      — an earlier stretch of footage replayed later (loop / copy-paste in time),
* ``video.jump``        — an intra-scene temporal jump (frames removed),
* ``video.source``      — stretches whose noise level does not match the rest
                           (footage from another camera or pipeline cut in). Weak,
                           corroborating evidence only: video codecs flatten most of the
                           noise difference between sources (a 4.5x difference in sensor
                           noise survives H.264 at CRF 20 as roughly 1.3x).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..core.types import Category, Evidence, Segment
from ..core.utils import calibrate

THUMB = (96, 54)
DUP_EPS = 0.35                  # mean abs gray difference below which frames are "the same"
REPEAT_SIM = 0.992
REPEAT_MIN_RUN = 8
JUMP_RATIO = 4.0
SCENE_HIST = 0.35
NOISE_STEP_THRESHOLD = 0.28     # |delta log sigma| between a stretch and the rest
NOISE_STEP_SOFTNESS = 0.06
KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


@dataclass
class FrameScan:
    fps: float
    n_total: int
    width: int
    height: int
    stride: int
    idx: np.ndarray                 # original frame index of each analysed frame
    thumbs: np.ndarray              # (N, 54, 96) float32
    diff: np.ndarray                # (N,) mean abs difference to previous analysed frame
    hist: np.ndarray                # (N,) histogram distance to previous frame, 0..1
    noise: np.ndarray               # (N,) noise sigma estimate
    keyframes: List[Tuple[int, np.ndarray]] = field(default_factory=list)  # (frame idx, RGB)
    analysed_frames: int = 0
    truncated: bool = False
    codec: str = ""

    pts: Optional[np.ndarray] = None  # presentation time (s) of each analysed frame

    @property
    def t(self) -> np.ndarray:
        """Presentation timestamps when the decoder reports them (correct for
        variable-frame-rate footage), otherwise frame index / nominal fps."""
        if self.pts is not None and len(self.pts) == len(self.idx):
            return self.pts
        return self.idx / max(self.fps, 1e-6)

    @property
    def duration(self) -> float:
        return self.n_total / max(self.fps, 1e-6)


def _noise_sigma(gray: np.ndarray) -> float:
    h, w = gray.shape
    if max(h, w) > 1280:            # centre crop keeps the cost flat for 4K footage
        y0, x0 = (h - min(h, 720)) // 2, (w - min(w, 1280)) // 2
        gray = gray[y0:y0 + 720, x0:x0 + 1280]
    g = gray.astype(np.float32)
    res = np.abs(cv2.filter2D(g, cv2.CV_32F, KERNEL))
    grad = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    ok = grad < max(np.percentile(grad, 60), 6.0)
    return float(np.sqrt(np.pi / 2) / 6.0 * res[ok].mean()) if ok.any() else 0.0


def scan_frames(path: str, max_frames: int = 3000, n_key: int = 5,
                max_decode: int = 5400) -> FrameScan:
    """Decode once. Every frame is analysed when the clip has <= ``max_frames`` frames;
    longer clips are analysed up to ``max_decode`` decoded frames (see ``truncated``)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open this video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not (1.0 <= fps <= 480.0):
        fps = 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
    # Decode budget is expressed in 1080p-equivalent frames so 4K clips stay responsive.
    max_decode = max(600, int(max_decode * min(1.0, (1920 * 1080) / max(w * h, 1))))
    span = min(n_total, max_decode) if n_total > 0 else max_decode
    stride = max(1, int(np.ceil(span / max_frames)))
    key_targets = set(np.linspace(0.12, 0.88, n_key) * max(span, 1) // 1)
    idx, thumbs, noise, keys, pts = [], [], [], [], []
    i = 0
    while True:
        if not cap.grab():
            break
        want_key = i in key_targets
        if i % stride == 0 or want_key:
            ok, bgr = cap.retrieve()
            if not ok or bgr is None:
                i += 1
                continue
            if i % stride == 0:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                thumbs.append(cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32))
                noise.append(_noise_sigma(gray))
                idx.append(i)
                pts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            if want_key:
                keys.append((i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        i += 1
        if i >= max_decode:
            break
    truncated = n_total > i > 0
    cap.release()
    if len(idx) < 2:
        raise RuntimeError("could not decode enough frames")
    T = np.stack(thumbs)
    d = np.zeros(len(T), np.float32)
    d[1:] = np.abs(T[1:] - T[:-1]).mean(axis=(1, 2))
    hs = np.stack([np.histogram(x, bins=32, range=(0, 256))[0] for x in T]).astype(np.float32)
    hs /= hs.sum(axis=1, keepdims=True) + 1e-9
    hd = np.zeros(len(T), np.float32)
    hd[1:] = 0.5 * np.abs(hs[1:] - hs[:-1]).sum(axis=1)
    if not keys:
        cap = cv2.VideoCapture(path)
        ok, bgr = cap.read()
        cap.release()
        if ok:
            keys.append((0, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    pts_arr: Optional[np.ndarray] = np.asarray(pts, dtype=np.float64)
    if len(pts_arr) < 2 or not np.all(np.isfinite(pts_arr)) or pts_arr[-1] <= 0 or \
            np.any(np.diff(pts_arr) < 0):
        pts_arr = None                      # decoder gave no usable timestamps
    return FrameScan(fps=fps, n_total=max(n_total, i), width=w, height=h, stride=stride,
                     pts=pts_arr,
                     analysed_frames=i, truncated=truncated,
                     idx=np.asarray(idx), thumbs=T, diff=d, hist=hd, noise=np.asarray(noise),
                     keyframes=keys, codec=codec)


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[start, end) index ranges where mask is True."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _series(fs: FrameScan, y: np.ndarray, ylabel: str, markers=()) -> dict:
    step = max(1, len(y) // 1500)
    return {"x": [float(v) for v in fs.t[::step]], "y": [float(v) for v in y[::step]],
            "xlabel": "time (s)", "ylabel": ylabel, "markers": [float(m) for m in markers]}


def detect_freeze(fs: FrameScan) -> Evidence:
    det, title = "video.freeze", "Frozen / duplicated frames"
    d = fs.diff
    motion = float(np.median(d[1:]))
    if fs.stride > 1:
        return Evidence.not_applicable(det, title, Category.MANIPULATION,
                                       "video was sub-sampled; consecutive frames not compared")
    if motion < 4 * DUP_EPS:
        return Evidence.not_applicable(det, title, Category.MANIPULATION,
                                       "footage is (nearly) static, duplicates are natural",
                                       median_motion=motion)
    dup = d < DUP_EPS
    dup[0] = False
    runs = [(a, b) for a, b in _runs(dup)]
    long_runs = [(a, b) for a, b in runs if b - a >= 3]
    singles = [a for a, b in runs if b - a < 3]
    cadence = False
    if len(singles) >= 6:
        gaps = np.diff(singles)
        cadence = float(np.std(gaps)) < 0.25 * float(np.mean(gaps)) + 1.0
    segs = [Segment(float(fs.t[a - 1]), float(fs.t[b - 1]), min(1.0, 0.6 + 0.02 * (b - a)),
                    f"{b - a + 1} identical frames") for a, b in long_runs]
    longest = max((b - a for a, b in long_runs), default=0)
    score = calibrate(longest, 4, 1.5) if long_runs else (0.35 if cadence else 0.05)
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION, score=score, confidence=0.7,
        summary=(f"{len(long_runs)} stretch(es) where the picture freezes while the surrounding "
                 f"footage moves (longest {longest + 1} frames). Frozen frames are used to pad "
                 f"removed content or hide an event." if long_runs else
                 ("Single duplicated frames at a regular cadence — a frame-rate conversion, "
                  "not an edit." if cadence else "No frozen stretches in moving footage.")),
        details={"median_motion": motion, "long_runs": len(long_runs), "longest": int(longest),
                 "single_duplicates": len(singles), "regular_cadence": cadence},
        segments=segs,
        series={"frame_difference": _series(fs, d, "mean |Δ|", [s.start for s in segs])})


def detect_repeat(fs: FrameScan) -> Evidence:
    det, title = "video.repeat", "Replayed footage (temporal copy-paste)"
    n = len(fs.thumbs)
    if n < 3 * REPEAT_MIN_RUN:
        return Evidence.not_applicable(det, title, Category.MANIPULATION, "video too short")
    if float(np.median(fs.diff[1:])) < 4 * DUP_EPS:
        return Evidence.not_applicable(det, title, Category.MANIPULATION,
                                       "footage is (nearly) static")
    small = np.stack([cv2.resize(x, (16, 9), interpolation=cv2.INTER_AREA) for x in fs.thumbs])
    v = small.reshape(n, -1)
    v = v - v.mean(axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    sim = v @ v.T
    hit = sim > REPEAT_SIM
    min_gap = max(REPEAT_MIN_RUN, int(0.4 * fs.fps / fs.stride))
    found = []
    for k in range(min_gap, n - REPEAT_MIN_RUN):
        diag = np.diagonal(hit, k)
        if diag.sum() < REPEAT_MIN_RUN:
            continue
        for a, b in _runs(diag):
            if b - a < REPEAT_MIN_RUN:
                continue
            # The content must *change* during the run; otherwise it is just a static shot.
            if sim[a, b - 1] > REPEAT_SIM or fs.diff[a + 1:b].mean() < 2 * DUP_EPS:
                continue
            found.append((b - a, a, a + k, float(np.diagonal(sim, k)[a:b].mean())))
    found.sort(reverse=True)
    kept: List[Tuple[int, int, int, float]] = []
    for f in found:
        if all(abs(f[2] - g[2]) > g[0] for g in kept):
            kept.append(f)
    kept = kept[:5]
    segs = [Segment(float(fs.t[j]), float(fs.t[min(j + ln, n - 1)]), min(1.0, 0.6 + 0.01 * ln),
                    f"replays {fs.t[i]:.2f}s–{fs.t[min(i + ln, n - 1)]:.2f}s")
            for ln, i, j, _ in kept]
    longest = kept[0][0] if kept else 0
    score = calibrate(longest, REPEAT_MIN_RUN + 2, 2.0) if kept else 0.03
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION, score=score, confidence=0.75,
        summary=(f"{len(kept)} stretch(es) of moving footage appear twice (longest "
                 f"{longest * fs.stride} frames). Replaying earlier frames is a classic way to "
                 f"cover a removed event; loops and repeated on-screen animations also match."
                 if kept else "No stretch of moving footage is repeated."),
        details={"repeats": [{"frames": int(ln * fs.stride), "source_s": float(fs.t[i]),
                              "copy_s": float(fs.t[j]), "similarity": s} for ln, i, j, s in kept]},
        segments=segs)


def detect_jump(fs: FrameScan) -> Tuple[Evidence, List[int]]:
    """Intra-scene temporal jumps. Also returns indices of all cuts (scene cuts + jumps)."""
    det, title = "video.jump", "Temporal jumps (removed frames)"
    d = fs.diff.astype(np.float64)
    n = len(d)
    jumps, cuts = [], []
    for i in range(2, n - 2):
        lo, hi = max(1, i - 12), min(n, i + 13)
        nb = np.concatenate([d[lo:i], d[i + 1:hi]])
        base = max(float(np.median(nb)), DUP_EPS)
        if d[i] > JUMP_RATIO * base and d[i] > 2.5 * max(d[i - 1], d[i + 1], DUP_EPS):
            cuts.append(i)
            if fs.hist[i] < SCENE_HIST:
                jumps.append((i, d[i] / base))
    segs = [Segment(float(fs.t[i - 1]), float(fs.t[i]), float(np.clip(0.5 + r / 40, 0.55, 1.0)),
                    f"motion jumps {r:.0f}x within the same scene") for i, r in jumps]
    best = max((r for _, r in jumps), default=0.0)
    score = calibrate(best, JUMP_RATIO, 1.0) if jumps else 0.03
    if fs.stride > 1:
        conf = 0.3
    else:
        conf = 0.6
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION, score=score, confidence=conf,
        summary=(f"{len(jumps)} abrupt jump(s) in motion while the scene stays the same — "
                 f"consistent with frames having been removed. A deliberate jump cut in an "
                 f"edited video looks identical." if jumps else
                 f"No intra-scene temporal jumps ({len(cuts)} ordinary scene cut(s))."),
        details={"jumps": len(jumps), "scene_cuts": len(cuts) - len(jumps),
                 "max_ratio": float(best)},
        segments=segs,
        series={"frame_difference": _series(fs, fs.diff, "mean |Δ|", [s.end for s in segs])}), cuts


def detect_source(fs: FrameScan, cuts: List[int]) -> Evidence:
    det, title = "video.source", "Source consistency (noise level over time)"
    n = len(fs.noise)
    ln = np.log(np.maximum(fs.noise, 1e-3))
    bounds = [0] + sorted(set(cuts)) + [n]
    shots = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 6]
    if len(shots) < 2:
        return Evidence(detector=det, title=title, category=Category.MANIPULATION, score=0.03,
                        confidence=0.3, summary="A single continuous shot: nothing to compare "
                        "its noise level against.", details={"shots": len(shots)},
                        series={"noise_sigma": _series(fs, fs.noise, "noise σ")})
    med = np.array([np.median(ln[a:b]) for a, b in shots])
    length = np.array([b - a for a, b in shots], dtype=float)
    order = np.argsort(med)
    cum = np.cumsum(length[order]) / length.sum()
    ref = med[order][np.searchsorted(cum, 0.5)]          # duration-weighted median
    dev = np.abs(med - ref)
    segs, worst = [], 0.0
    for (a, b), dv, m in zip(shots, dev, med):
        if dv > NOISE_STEP_THRESHOLD and (b - a) < 0.5 * n:
            worst = max(worst, float(dv))
            ratio = float(np.exp(m - ref))
            segs.append(Segment(float(fs.t[a]), float(fs.t[b - 1]),
                                calibrate(dv, NOISE_STEP_THRESHOLD, NOISE_STEP_SOFTNESS),
                                f"noise level {ratio:.1f}x the rest of the video"))
    score = calibrate(worst, NOISE_STEP_THRESHOLD, NOISE_STEP_SOFTNESS) if segs else \
        calibrate(float(dev.max()), NOISE_STEP_THRESHOLD, NOISE_STEP_SOFTNESS) * 0.6
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION, score=score, confidence=0.35,
        summary=(f"{len(segs)} stretch(es) have a noise level that does not match the rest of "
                 f"the video — footage from a different camera, generator or processing chain "
                 f"was likely cut in. Multi-camera productions legitimately look like this."
                 if segs else f"Noise level is consistent across {len(shots)} shots."),
        details={"shots": len(shots), "max_log_deviation": float(dev.max()),
                 "threshold": NOISE_STEP_THRESHOLD},
        segments=segs,
        series={"noise_sigma": _series(fs, fs.noise, "noise σ",
                                       [s.start for s in segs] + [s.end for s in segs])})
