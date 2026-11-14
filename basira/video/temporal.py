"""Temporal video forensics: one decode pass, then detectors on per-frame signals.

Signals per analysed frame: a 96x54 gray thumbnail, the mean absolute difference to the
previous thumbnail ("d"), a gray-histogram distance and (on a subset of frames) a
sensor-noise estimate. From these:

* ``video.freeze``  — the picture stops while the scene was moving and then *jumps*:
                       frozen frames standing in for removed footage,
* ``video.repeat``  — an earlier stretch of moving footage shown again,
* ``video.jump``    — a discontinuity inside one shot (frames removed),
* ``video.source``  — informational: noise level per shot.

What these detectors cannot see (measured with ``benchmarks/video_robustness.py``):
a removal in a shot that barely moves, a change of playback speed, footage cut in with
ordinary hard cuts (an edited video looks exactly like that), and anything in a stretch
where nothing moves. Scene cuts, fades, flashes and frame-rate conversion are recognised
and not reported.

All thresholds are relative to the motion of the clip itself ("level": the upper-quartile
``d`` within one second), because real footage ranges from d = 0.15 (locked-off camera,
only waves move; 1080p) to d > 3 (pan), which no absolute threshold covers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.types import Category, Evidence, Segment
from ..core.utils import calibrate

THUMB = (96, 54)
ANALYSIS_FPS = 60.0             # faster footage is decoded but only every n-th frame analysed
NOISE_SAMPLES = 60              # frames per clip on which the noise level is estimated (it only
                                # feeds the informational video.source; was every frame and half
                                # of the scan time)
MAX_FRAME_PIXELS = 40_000_000  # larger frames are refused: one 16384x8640 VP9 frame took 107-147 s
                                # and 1.4 GB for a short clip (fuzz specialist); 8K is 33 MP
KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)

# --- duplicates / liveness (d = mean abs gray difference of consecutive thumbnails) ---
DUP_EPS = 0.35                  # absolute ceiling for "the same picture" (legacy value)
DUP_REL = 0.25                  # ... and at most this fraction of the local motion level
                                # (frozen frames measure 0.00-0.08 after H.264; a locked-off
                                # real shot at 1080p moves at 0.16)
LIVE_MIN = 0.08                 # motion level below which a stretch is treated as static
STATIC_FRACTION = 0.5           # more duplicate frames than this: nothing to judge
# --- jumps ---
JUMP_RATIO = 4.0                # d spike vs local level (real footage: 0.5 s removed gives
                                # 8-18x, 0.17 s gives 4-7x, key frames of a clean stream up to 9x
                                # -- hence the peak test below)
JUMP_SOFTNESS = 0.5             # ratio 5 -> score 0.88: "Suspicious" on its own
JUMP_ISOLATION = 2.5            # ... and vs both neighbouring frames
JUMP_PEAK_RATIO = 3.0           # the strongest local change must spike too: a key frame of a
                                # low-bitrate stream refreshes the whole picture a little
                                # (mean d x8 on real footage at CRF 32) but moves nothing
                                # (99.5th percentile x2), a removal moves something (x4-8)
SAME_SCENE_NCC = 0.80           # shift-compensated correlation above which two frames are one shot
SCENE_HIST = 0.35               # legacy histogram distance, reported only
# --- freeze ---
FREEZE_MIN_FRAMES = 3
FREEZE_MIN_SECONDS = 0.10
FREEZE_RESUME_RATIO = 2.0       # jump (vs local level) when the picture starts moving again
# --- repeat ---
REPEAT_MIN_RUN = 8              # frames
REPEAT_MIN_SECONDS = 0.25
REPEAT_REL = 1.5                # a replayed frame is about as close to its far-away source as to
                                # its own temporal neighbours (rms on 32x18 signatures). In 261
                                # clean / natural real files (and 52 synthetic) no frame run comes
                                # anywhere near: thresholds from 1.0 to 3.0 all give 0 false
                                # positives. Replays measure 0.1-0.7 as exported and up to 2.9
                                # after CRF 35 / 480p; 1.25-2.0 finds the most (48-50 of 54),
                                # above 2.5 a replay starts matching at several lags.
# --- source ---
NOISE_STEP_THRESHOLD = 0.28     # |delta log sigma| between a shot and the rest
NOISE_STEP_SOFTNESS = 0.06


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
    key_pts: Dict[int, float] = field(default_factory=dict)

    def time_of(self, frame_index: int) -> float:
        """Presentation time of a frame (exact for key frames, else nearest analysed frame)."""
        if frame_index in self.key_pts:
            return float(self.key_pts[frame_index])
        if self.pts is not None and len(self.idx):
            return float(self.pts[int(np.argmin(np.abs(self.idx - frame_index)))])
        return frame_index / max(self.fps, 1e-6)

    @property
    def analysed_seconds(self) -> float:
        if self.pts is not None and len(self.pts):
            return float(max([self.pts[-1]] + list(self.key_pts.values())))
        return self.analysed_frames / max(self.fps, 1e-6)

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

    @property
    def rate(self) -> float:
        """Analysed frames per second of video."""
        t = self.t
        if len(t) > 2:
            step = float(np.median(np.diff(t)))
            if step > 0:
                return 1.0 / step
        return self.fps / max(self.stride, 1)


def _noise_sigma(gray: np.ndarray) -> float:
    h, w = gray.shape
    if max(h, w) > 1280:            # centre crop keeps the cost flat for 4K footage
        y0, x0 = (h - min(h, 720)) // 2, (w - min(w, 1280)) // 2
        gray = gray[y0:y0 + 720, x0:x0 + 1280]
    g = gray.astype(np.float32)
    res = np.abs(cv2.filter2D(g, cv2.CV_32F, KERNEL))
    grad = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    # The percentile only separates flat from textured pixels: a 1-in-16 sample is enough
    # and avoids sorting a million values per frame (this was half of the scan time).
    ok = grad < max(float(np.percentile(grad[::4, ::4], 60)), 6.0)
    return float(np.sqrt(np.pi / 2) / 6.0 * res[ok].mean()) if ok.any() else 0.0


def scan_frames(path: str, max_frames: int = 3000, n_key: int = 5,
                max_decode: int = 5400) -> FrameScan:
    """Decode once. Every frame is analysed when the clip has <= ``max_frames`` frames and
    at most ``ANALYSIS_FPS`` frames per second; longer clips are analysed up to
    ``max_decode`` decoded frames (see ``truncated``)."""
    from ..core.io import _enforce_opencv_whitelist, decoder_path

    _enforce_opencv_whitelist()               # re-assert: the environment may have changed
    # "file:" + absolute path: a relative name like "pipe:0.mp4" must not select a protocol.
    cap = cv2.VideoCapture(decoder_path(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open this video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not (1.0 <= fps <= 480.0):
        fps = 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
    if w * h > MAX_FRAME_PIXELS:
        cap.release()
        raise RuntimeError(f"frames of {w}x{h} exceed the {MAX_FRAME_PIXELS // 1_000_000} MP limit; "
                           f"not decoded")
    # Decode budget is expressed in 1080p-equivalent frames so 4K clips stay responsive.
    max_decode = max(600, int(max_decode * min(1.0, (1920 * 1080) / max(w * h, 1))))
    span = min(n_total, max_decode) if n_total > 0 else max_decode
    # Converting a decoded frame to BGR costs as much as decoding it (4K HEVC), so frames
    # beyond ANALYSIS_FPS are grabbed (decoded) but not retrieved.
    stride = max(1, int(np.ceil(span / max_frames)), int(fps / ANALYSIS_FPS + 0.01))
    noise_every = max(1, int(np.ceil(span / stride / NOISE_SAMPLES)))
    key_targets = set(np.linspace(0.12, 0.88, n_key) * max(span, 1) // 1)
    idx, thumbs, noise, keys, pts = [], [], [], [], []
    key_pts: Dict[int, float] = {}
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
                noise.append(_noise_sigma(gray) if len(idx) % noise_every == 0 else np.nan)
                idx.append(i)
                pts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            if want_key:
                keys.append((i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                key_pts[i] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
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
    nz = np.asarray(noise, dtype=np.float64)
    have = np.isfinite(nz)
    if have.any() and not have.all():         # frames in between take the nearest estimate
        pos = np.where(have)[0]
        nz = nz[pos[np.argmin(np.abs(np.arange(len(nz))[:, None] - pos[None, :]), axis=1)]]
    nz = np.where(np.isfinite(nz), nz, 0.0)
    if not keys:
        _enforce_opencv_whitelist()
        cap = cv2.VideoCapture(decoder_path(path), cv2.CAP_FFMPEG)
        ok, bgr = cap.read()
        cap.release()
        if ok:
            keys.append((0, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    pts_arr: Optional[np.ndarray] = np.asarray(pts, dtype=np.float64)
    if len(pts_arr) < 2 or not np.all(np.isfinite(pts_arr)) or pts_arr[-1] <= 0 or \
            np.any(np.diff(pts_arr) < 0):
        pts_arr = None                      # decoder gave no usable timestamps
    return FrameScan(fps=fps, n_total=max(n_total, i), width=w, height=h, stride=stride,
                     pts=pts_arr, key_pts=key_pts if pts_arr is not None else {},
                     analysed_frames=i, truncated=truncated,
                     idx=np.asarray(idx), thumbs=T, diff=d, hist=hd, noise=nz,
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


# ----------------------------------------------------------------------------- shared signals
@dataclass
class Motion:
    d: np.ndarray                   # mean abs difference to the previous analysed frame
    peak: np.ndarray                # 99.5th percentile of the same difference
    level: np.ndarray               # upper-quartile d of the last second of moving frames
    dup: np.ndarray                 # frame shows the same picture as the previous one
    live: np.ndarray                # indices of frames that are not duplicates
    win: int                        # frames in one second
    static_fraction: float
    cadence: bool                   # duplicates come as regular single frames


def motion(fs: FrameScan) -> Motion:
    cached = getattr(fs, "_motion", None)
    if cached is not None:
        return cached
    d = np.asarray(fs.diff, dtype=np.float64)
    n = len(d)
    peak = np.zeros(n)
    if n > 1:
        ad = np.abs(fs.thumbs[1:] - fs.thumbs[:-1]).reshape(n - 1, -1)
        k = max(1, ad.shape[1] // 200)
        peak[1:] = np.partition(ad, -k, axis=1)[:, -k]
    win = int(np.clip(round(fs.rate), 5, 120))
    # The level is causal and built from non-duplicate frames only: "how much did the
    # scene move during the last second in which it moved". A symmetric window would let
    # a busy stretch ahead (a dissolve, a cut) turn the calm frames before it into
    # "duplicates".
    level = np.zeros(n)
    dup = np.zeros(n, bool)
    hist: List[float] = []
    start = float(np.percentile(d[1:win + 1], 75)) if n > 1 else 0.0
    for i in range(1, n):
        ref = float(np.percentile(hist[-win:], 75)) if len(hist) >= 5 else start
        level[i] = ref
        dup[i] = d[i] < min(DUP_REL * ref, DUP_EPS)
        if not dup[i]:
            hist.append(d[i])
    level[0] = level[1] if n > 1 else 0.0
    runs = _runs(dup)
    singles = [a for a, b in runs if b - a <= 2]
    cadence = False
    if len(singles) >= 6:
        gaps = np.diff(singles)
        cadence = float(np.std(gaps)) < 0.25 * float(np.mean(gaps)) + 1.0
    fs._motion = Motion(d=d, peak=peak, level=level, dup=dup, live=np.where(~dup)[0], win=win,
                        static_fraction=float(dup[1:].mean()) if n > 1 else 0.0, cadence=cadence)
    return fs._motion


def _normalised(x: np.ndarray) -> np.ndarray:
    """Thumbnail without its mean and contrast (floor of 4 gray levels for flat frames)."""
    return (x - x.mean()) / max(float(x.std()), 4.0)


def _same_scene(a: np.ndarray, b: np.ndarray) -> float:
    """Best normalised correlation of ``b``'s centre inside ``a`` (and vice versa), i.e.
    allowing the camera to have moved by up to 1/8 of the picture."""
    mx, my = THUMB[0] // 8, THUMB[1] // 8
    best = -1.0
    for x, y in ((a, b), (b, a)):
        tpl = y[my:-my, mx:-mx]
        if float(tpl.std()) < 1.0 or float(x.std()) < 1.0:
            return 0.0                       # featureless frame (black, fog): no statement
        r = cv2.matchTemplate(x, tpl, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(r.max()))
    return best


def _transient(T: np.ndarray, i: int, d_i: float) -> bool:
    """The picture is back where it was within a few frames (a flash, a glitch): either
    the frames after i match the frame before it, or i itself matches a frame
    shortly before the excursion started."""
    for k in (1, 2, 3):
        if i + k < len(T) and float(np.abs(T[i + k] - T[i - 1]).mean()) < 0.5 * d_i:
            return True
        if i - 1 - k >= 0 and float(np.abs(T[i] - T[i - 1 - k]).mean()) < 0.5 * d_i:
            return True
    return False


def _spikes(fs: FrameScan, m: Motion) -> List[Tuple[int, float, float]]:
    """(frame, ratio, same-scene correlation) of isolated one-frame discontinuities in
    moving footage, measured on the non-duplicate frames."""
    lv = m.live
    out = []
    if len(lv) < 6:
        return out
    T = fs.thumbs
    dl = m.d[lv]
    for q in range(2, len(lv) - 2):
        i = int(lv[q])
        lo, hi = max(1, q - 12), min(len(lv), q + 13)
        nb = np.concatenate([dl[lo:q], dl[q + 1:hi]])
        base = float(np.median(nb))
        if base < LIVE_MIN:
            continue
        ratio = dl[q] / base
        if ratio < JUMP_RATIO or dl[q] < JUMP_ISOLATION * max(dl[q - 1], dl[q + 1]):
            continue
        pk = m.peak[lv]
        if pk[q] < JUMP_PEAK_RATIO * max(float(np.median(np.concatenate([pk[lo:q], pk[q + 1:hi]]))), 1.0):
            continue                          # key-frame refresh of the codec, nothing moved
        if _transient(T, i, float(dl[q])):
            continue
        # Global brightness changes (exposure step, a light switched on) are not edits:
        # the spike must survive removing mean and contrast of each frame.
        j = int(lv[q - 1])                    # the previous moving frame, as reference
        spike = float(np.abs(_normalised(T[i]) - _normalised(T[i - 1])).mean())
        usual = float(np.abs(_normalised(T[j]) - _normalised(T[j - 1])).mean())
        norm_ratio = spike / max(usual, 1e-6)
        out.append((i, float(min(ratio, norm_ratio)), _same_scene(T[i - 1], T[i])))
    return out


# ----------------------------------------------------------------------------- detectors
def detect_freeze(fs: FrameScan) -> Evidence:
    det, title = "video.freeze", "Frozen / duplicated frames"
    m = motion(fs)
    if m.static_fraction > STATIC_FRACTION and not m.cadence:
        return Evidence.not_applicable(det, title, Category.MANIPULATION,
                                       "footage is (nearly) static, duplicates are natural",
                                       static_fraction=m.static_fraction)
    if float(np.median(m.level)) < LIVE_MIN:
        return Evidence.not_applicable(det, title, Category.MANIPULATION,
                                       "footage is (nearly) static, duplicates are natural",
                                       median_motion=float(np.median(m.d[1:])))
    T, t, n = fs.thumbs, fs.t, len(m.d)
    min_run = max(FREEZE_MIN_FRAMES, int(np.ceil(FREEZE_MIN_SECONDS * fs.rate)))
    found, held = [], 0
    for a, b in _runs(m.dup):
        if b - a < min_run - 1 or b >= n or a < 2:
            continue                          # b - a duplicates = b - a + 1 identical frames
        before = float(np.median(m.d[max(1, a - 3):a]))
        lvl = float(m.level[a - 1])
        if lvl < LIVE_MIN or before < 0.5 * lvl:
            held += 1                         # the scene had come to rest by itself
            continue
        resume = float(m.d[b]) / max(lvl, 1e-6)
        if m.peak[b] < JUMP_PEAK_RATIO * max(float(np.median(m.peak[max(1, a - 12):a])), 1.0):
            resume = 0.0                      # key-frame refresh, not a jump
        if float(T[a - 1].std()) < 2.0:
            held += 1                         # black / blank frames (fade, title gap)
            continue
        if resume < FREEZE_RESUME_RATIO or _same_scene(T[b - 1], T[b]) < SAME_SCENE_NCC:
            held += 1                         # motion continues where it stopped, or a cut
            continue
        found.append((a, b, resume))
    segs = [Segment(float(t[a - 1]), float(t[b - 1]), min(1.0, 0.6 + 0.02 * (b - a)),
                    f"{b - a + 1} identical frames, then a jump") for a, b, _ in found]
    longest = max((b - a for a, b, _ in found), default=0)
    score = calibrate(longest, min_run, 1.5) if found else (0.35 if m.cadence else 0.05)
    conf = 0.7 if fs.stride == 1 else 0.5
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION, score=score, confidence=conf,
        summary=(f"{len(found)} stretch(es) where the picture freezes in moving footage and "
                 f"then jumps (longest {longest + 1} frames). Frozen frames are used to pad "
                 f"removed content or hide an event; a stalled video call or stream looks the same."
                 if found else
                 ("Single duplicated frames at a regular cadence — a frame-rate conversion or "
                  "slow motion, not an edit." if m.cadence else
                  "No frozen stretch followed by a jump. A freeze that replaces a moment in "
                  "which nothing moved cannot be seen.")),
        details={"median_motion": float(np.median(m.d[1:])), "long_runs": len(found),
                 "longest": int(longest), "held_pictures": held,
                 "duplicate_fraction": m.static_fraction, "regular_cadence": m.cadence},
        segments=segs,
        series={"frame_difference": _series(fs, m.d, "mean |Δ|", [s.start for s in segs])})


def detect_repeat(fs: FrameScan) -> Evidence:
    det, title = "video.repeat", "Replayed footage (temporal copy-paste)"
    m = motion(fs)
    lv = m.live
    n = len(lv)
    min_run = max(REPEAT_MIN_RUN, int(np.ceil(REPEAT_MIN_SECONDS * fs.rate)))
    if n < 3 * min_run:
        return Evidence.not_applicable(det, title, Category.MANIPULATION, "video too short")
    if m.static_fraction > STATIC_FRACTION and not m.cadence or float(np.median(m.level)) < LIVE_MIN:
        return Evidence.not_applicable(det, title, Category.MANIPULATION,
                                       "footage is (nearly) static")
    T = fs.thumbs[lv]
    v = T.reshape(n, THUMB[1] // 3, 3, THUMB[0] // 3, 3).mean(axis=(2, 4)).reshape(n, -1)
    v = v.astype(np.float64) - v.mean(axis=0, keepdims=True)   # keeps the subtraction below exact
    sq = (v * v).sum(axis=1)
    msd = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (v @ v.T), 0.0) / v.shape[1]
    adj = np.full(n, np.inf)                 # distance to the nearer temporal neighbour
    step = np.diagonal(msd, 1)
    adj[1:] = step
    adj[:-1] = np.minimum(adj[:-1], step)
    live_floor = (0.5 * LIVE_MIN) ** 2
    hit = msd < (REPEAT_REL ** 2) * np.minimum(adj[:, None], adj[None, :])
    hit &= (adj[:, None] > live_floor) & (adj[None, :] > live_floor)
    # Frame-rate conversion drops or repeats frames at different places in the source and
    # in the copy, so the lag wobbles by one frame and single frames have no twin.
    near = hit.copy()
    near[:, 1:] |= hit[:, :-1]
    near[:, :-1] |= hit[:, 1:]
    min_gap = max(min_run, int(0.4 * fs.rate))
    found = []
    for k in range(min_gap, n - min_run):
        diag = np.diagonal(near, k)
        if diag.sum() < min_run:
            continue
        closed = diag.copy()
        closed[1:-1] |= diag[:-2] & diag[2:]          # bridge one missing frame
        for a, b in _runs(closed):
            if b - a >= min_run and diag[a:b].mean() >= 0.7:
                found.append((b - a, a, a + k))
    found.sort(reverse=True)
    kept: List[Tuple[int, int, int]] = []
    lags: List[List[int]] = []
    for f in found:
        for g, lg in zip(kept, lags):
            if abs(f[2] - g[2]) <= g[0] or abs(f[1] - g[1]) <= g[0] or abs(f[2] - g[1]) <= g[0] \
                    or abs(f[1] - g[2]) <= g[0]:
                if all(abs((f[2] - f[1]) - x) > 3 for x in lg):
                    lg.append(f[2] - f[1])
                break
        else:
            kept.append(f)
            lags.append([f[2] - f[1]])
    # The same footage at three or more places is a loop or periodic motion (a pendulum, a
    # blinking light, an animated logo): reported, but not as an edit.
    periodic = [len(lg) >= 2 for lg in lags]
    t = fs.t
    once = [f for f, p in zip(kept, periodic) if not p][:5]
    segs = [Segment(float(t[lv[j]]), float(t[lv[min(j + ln, n - 1)]]), min(1.0, 0.6 + 0.01 * ln),
                    f"replays {t[lv[i]]:.2f}s–{t[lv[min(i + ln, n - 1)]]:.2f}s")
            for ln, i, j in once]
    longest = once[0][0] if once else 0
    score = calibrate(longest, min_run + 2, 2.0) if once else (0.3 if any(periodic) else 0.03)
    # A copy that begins or ends with a discontinuity was spliced in; one that flows in and
    # out of its surroundings may be motion that happens to repeat.
    edges = {e for ln, i, j in once for c in (j, j + ln) for e in range(c - 2, c + 3) if 0 < e < n}
    spliced = any(m.d[lv[e]] >= JUMP_RATIO * max(m.level[lv[e]], LIVE_MIN) for e in edges)
    conf = 0.9 if spliced else 0.75
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION, score=score,
        confidence=conf if fs.stride == 1 else 0.55,
        summary=(f"{len(once)} stretch(es) of moving footage appear twice (longest "
                 f"{longest * fs.stride} frames). Replaying earlier frames is a classic way to "
                 f"cover a removed event." if once else
                 ("The same motion repeats three or more times: a loop or periodic motion. "
                  "Footage looped to hide an event looks the same and is not flagged here."
                  if any(periodic) else "No stretch of moving footage is repeated.")),
        details={"repeats": [{"frames": int(ln * fs.stride), "source_s": float(t[lv[i]]),
                              "copy_s": float(t[lv[j]])} for ln, i, j in once],
                 "periodic": int(sum(periodic)), "bounded_by_discontinuity": bool(spliced)},
        segments=segs)


def detect_jump(fs: FrameScan) -> Tuple[Evidence, List[int]]:
    """Discontinuities inside a shot. Also returns the indices of all cuts (scene cuts + jumps)."""
    det, title = "video.jump", "Temporal jumps (removed frames)"
    m = motion(fs)
    spikes = _spikes(fs, m)
    jumps = [(i, r) for i, r, ncc in spikes if ncc >= SAME_SCENE_NCC and r >= JUMP_RATIO]
    cuts = [i for i, _, _ in spikes]
    t = fs.t
    segs = [Segment(float(t[i - 1]), float(t[i]), float(np.clip(0.5 + r / 40, 0.55, 1.0)),
                    f"motion jumps {r:.0f}x within the same scene") for i, r in jumps]
    best = max((r for _, r in jumps), default=0.0)
    score = calibrate(best, JUMP_RATIO, JUMP_SOFTNESS) if jumps else 0.03
    quiet = float((m.level < LIVE_MIN).mean())
    conf = 0.6 if fs.stride == 1 else 0.4
    blind = (f" {quiet:.0%} of the video is too static to judge." if quiet > 0.2 else "")
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION, score=score, confidence=conf,
        summary=(f"{len(jumps)} abrupt jump(s) in motion while the scene stays the same — "
                 f"consistent with frames having been removed. A deliberate jump cut in an "
                 f"edited video looks identical." if jumps else
                 f"No jump inside a shot ({len(cuts)} ordinary scene cut(s)). Removals of a few "
                 f"frames, or in a shot that barely moves, cannot be seen.") + blind,
        details={"jumps": len(jumps), "scene_cuts": len(cuts) - len(jumps),
                 "max_ratio": float(best), "static_fraction": quiet},
        segments=segs,
        series={"frame_difference": _series(fs, fs.diff, "mean |Δ|", [s.end for s in segs])}), cuts


def detect_source(fs: FrameScan, cuts: List[int]) -> Evidence:
    """Noise level per shot. INFORMATIONAL: confidence 0, no time ranges, never part of the
    verdict. On real footage the noise estimate follows the content of a shot (and what
    the codec did to it) far more than the camera: it singled out a shot in 27 of 27
    legitimate multi-shot clips and in 54 of 54 clips with foreign footage cut in
    (benchmarks/video_robustness.py), i.e. it cannot tell the two apart."""
    det, title = "video.source", "Source consistency (noise level over time)"
    n = len(fs.noise)
    ln = np.log(np.maximum(fs.noise, 1e-3))
    bounds = [0] + sorted(set(cuts)) + [n]
    shots = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 6]
    series = {"noise_sigma": _series(fs, fs.noise, "noise σ", [float(fs.t[a]) for a, _ in shots[1:]])}
    if len(shots) < 2:
        return Evidence(detector=det, title=title, category=Category.MANIPULATION, score=0.0,
                        confidence=0.0, summary="Informational: a single continuous shot, "
                        "nothing to compare its noise level against.",
                        details={"shots": len(shots), "informational": True}, series=series)
    med = np.array([np.median(ln[a:b]) for a, b in shots])
    length = np.array([b - a for a, b in shots], dtype=float)
    order = np.argsort(med)
    cum = np.cumsum(length[order]) / length.sum()
    ref = med[order][np.searchsorted(cum, 0.5)]          # duration-weighted median
    dev = np.abs(med - ref)
    odd = [(float(fs.t[a]), float(fs.t[b - 1]), float(np.exp(mm - ref)))
           for (a, b), dv, mm in zip(shots, dev, med) if dv > NOISE_STEP_THRESHOLD]
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION,
        score=calibrate(float(dev.max()), NOISE_STEP_THRESHOLD, NOISE_STEP_SOFTNESS),
        confidence=0.0,
        summary=(f"Informational, not part of the verdict: {len(odd)} of {len(shots)} shots have "
                 f"a noise level unlike the rest. Shots of different content, and ordinary "
                 f"edits, look like this as often as foreign footage does." if odd else
                 f"Informational: noise level is similar across {len(shots)} shots."),
        details={"shots": len(shots), "max_log_deviation": float(dev.max()),
                 "threshold": NOISE_STEP_THRESHOLD, "informational": True,
                 "odd_shots": [{"start": a, "end": b, "noise_ratio": r} for a, b, r in odd]},
        series=series)
