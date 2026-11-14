"""audio.synthetic -- weak DSP cues of neural TTS / vocoder output.

Honest scope: trained detectors (e.g. ASVspoof-style countermeasures) beat any
hand-written heuristic on this task. These cues are cheap, explainable and
model-free, so they are reported with low-to-moderate confidence and are meant
to be read together with ``audio.bandwidth`` and ``audio.metadata``.

Cues
----
1. **Unnaturally clean pauses** (weight ``W_CLEAN``): a microphone in a room
   always records a noise floor (typically -70..-40 dBFS). Generated speech is
   usually rendered on digital silence, so its pauses sit below
   ``CLEAN_PAUSE_DB`` (or are exactly zero). ``clean_fraction`` = share of pause
   frames (frames ``PAUSE_BELOW_LOUD_DB`` below the loud level) that are that
   clean. Noise-gated podcasts and heavily denoised recordings show the same
   cue.
2. **Constant floor across pauses** (weight ``W_FLAT``): spread (std, dB) of
   the per-pause median level; real rooms vary by a dB or more between pauses
   (HVAC, movement), synthetic or gated pauses do not. Very weak alone.
3. **Persistent narrow spectral lines** above ``LINE_MIN_HZ`` (weight
   ``W_LINES``): vocoders with fixed frame / upsampling structure can leave
   stationary tones present in every part of the file. Mains harmonics are
   excluded; CRT / switching-supply whine produces the same cue. Skipped for
   lossy files (codec artefacts).

``raw = W_CLEAN * clean_fraction + W_FLAT * flat_cue + W_LINES * min(1, n_lines / 3)``;
``score = calibrate(raw, RAW_THRESHOLD, RAW_SOFTNESS)``.

Measured on real text-to-speech (21 macOS ``say`` voices, benchmarks/audio_robustness.py):
the cues exceed their threshold for 8 voices -- those rendered on digital silence. The
other 13 leave dither-level noise (-82..-88 dBFS) or breath-like sound in their pauses
and pass. After the speech is played into a (simulated) room no voice is detected, and
after resampling to 16 kHz none reaches "Suspicious". So this detector can only add
weight to a suspicion; a quiet result says nothing about whether the speech is human,
and its summary says so.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..core.types import Category, Evidence
from ..core.utils import calibrate
from ._dsp import SILENCE_DB, frame_levels, make_series, power_frames, runs, to_db

DETECTOR = "audio.synthetic"
TITLE = "Synthetic-speech cues (heuristic)"

PAUSE_BELOW_LOUD_DB = 30.0
MIN_PAUSE_FRAMES = 100       # 1 s of pauses in total
MIN_PAUSE_RUN = 10           # 100 ms: one "pause" for the floor-spread cue
CLEAN_PAUSE_DB = -90.0
FLAT_SPREAD_DB = 1.0         # spread below this -> flat cue rises towards 1
LINE_MIN_HZ = 4000.0
LINE_DB = 15.0               # line height over the local (±150 Hz) median
LINE_PARTS = 4
W_CLEAN, W_FLAT, W_LINES = 0.6, 0.15, 0.25
RAW_THRESHOLD = 0.45
RAW_SOFTNESS = 0.1
BASE_CONFIDENCE = 0.35


def _persistent_lines(x: np.ndarray, sr: int) -> List[float]:
    """High-frequency narrow lines present in every one of LINE_PARTS parts."""
    from scipy.ndimage import median_filter

    nfft = int(2 ** np.round(np.log2(sr / 5.86)))   # ~5.9 Hz bins (8192 @ 48k)
    if len(x) < LINE_PARTS * 2 * nfft or sr / 2 <= LINE_MIN_HZ + 500:
        return []
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    df = freqs[1]
    k = max(5, int(round(300.0 / df)) | 1)
    hi = (freqs >= LINE_MIN_HZ) & (freqs <= 0.97 * sr / 2)
    mains = np.zeros_like(hi)
    for base in (50.0, 60.0):
        mains |= np.abs((freqs / base) - np.round(freqs / base)) * base <= 2 * df
    masks = []
    edges = np.linspace(0, len(x), LINE_PARTS + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        acc, n = None, 0
        for _, p in power_frames(x[a:b], nfft, nfft // 2, chunk_frames=256):
            s = p.sum(axis=0, dtype=np.float64)
            acc = s if acc is None else acc + s
            n += len(p)
        if acc is None:
            return []
        S = to_db(acc / max(n, 1))
        exc = S - median_filter(S, size=k, mode="nearest")
        m = (exc >= LINE_DB) & hi & ~mains & (S > S[hi].max() - 80)
        # tolerate ±1 bin of drift between parts
        masks.append(m | np.roll(m, 1) | np.roll(m, -1))
    common = np.logical_and.reduce(masks)
    st, en = runs(common)
    return [float(freqs[(a + b - 1) // 2]) for a, b in zip(st, en)]


def detect_synthetic(x: np.ndarray, sr: int, *, lossy: bool = False) -> Evidence:
    dur = len(x) / sr
    if dur < 2.0:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.AI_GENERATION, "too short")
    t, db, _ = frame_levels(x, sr)
    live = db > SILENCE_DB + 1
    if live.sum() < 50:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.AI_GENERATION, "no signal")
    loud = float(np.percentile(db[live], 95))
    # ignore leading / trailing material below the pause threshold
    act = np.flatnonzero(db >= loud - PAUSE_BELOW_LOUD_DB)
    if act.size == 0:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.AI_GENERATION, "no active audio")
    inner = np.zeros(len(db), bool)
    inner[act[0]:act[-1] + 1] = True
    pause = inner & (db < loud - PAUSE_BELOW_LOUD_DB)
    if pause.sum() < MIN_PAUSE_FRAMES:
        return Evidence.not_applicable(
            DETECTOR, TITLE, Category.AI_GENERATION,
            f"no pauses to examine ({pause.sum() * 10} ms of frames {PAUSE_BELOW_LOUD_DB:.0f} dB "
            f"below the loud level); the cues need speech with pauses")

    clean_frac = float(np.mean(db[pause] < CLEAN_PAUSE_DB))
    st, en = runs(pause)
    keep = (en - st) >= MIN_PAUSE_RUN
    pause_levels = [float(np.median(np.maximum(db[a:b], -140.0))) for a, b in zip(st[keep], en[keep])]
    spread = float(np.std(pause_levels)) if len(pause_levels) >= 3 else float("nan")
    flat_cue = float(np.clip((FLAT_SPREAD_DB - spread) / FLAT_SPREAD_DB, 0, 1)) \
        if np.isfinite(spread) else 0.0
    # Lossy codecs create their own stationary artefacts: skip the line cue there.
    lines = [] if lossy else _persistent_lines(x, sr)
    line_cue = min(1.0, len(lines) / 3.0)

    raw = W_CLEAN * clean_frac + W_FLAT * flat_cue + W_LINES * line_cue
    score = calibrate(raw, RAW_THRESHOLD, RAW_SOFTNESS)
    conf = BASE_CONFIDENCE * float(np.clip(len(pause_levels) / 5.0, 0.4, 1.0))

    cues = []
    if clean_frac >= 0.5:
        cues.append(f"{clean_frac:.0%} of pause frames are below {CLEAN_PAUSE_DB:.0f} dBFS "
                    f"(no room noise)")
    if flat_cue >= 0.5:
        cues.append(f"the pause floor is unusually constant (spread {spread:.2f} dB)")
    if lines:
        cues.append(f"{len(lines)} persistent narrow line(s) at "
                    + ", ".join(f"{f / 1000:.2f} kHz" for f in lines[:5]))
    if cues and score >= 0.5:
        head = "Synthetic-speech cues: " + "; ".join(cues) + "."
    elif cues:
        head = "Only a weak cue, below threshold: " + "; ".join(cues) + "."
    else:
        head = (f"No synthetic-speech cues: pauses carry a natural noise floor "
                f"({clean_frac:.0%} unnaturally clean), floor spread "
                f"{spread:.2f} dB, no persistent high-frequency lines.")
    if score >= 0.5:
        tail = ("These DSP heuristics are weak compared with trained detectors; noise gates and "
                "denoisers produce the same clean pauses.")
    else:
        tail = ("This is not evidence of human speech: most text-to-speech voices tested (13 of "
                "21) pass this check, and all of them do once re-recorded in a room.")
    summary = f"{head} Combined cue {raw:.2f} (threshold {RAW_THRESHOLD:.2f}). {tail}"
    pt = t[pause]
    series = {"pause_levels": make_series(pt, np.maximum(db[pause], -140.0),
                                          "pause frame level (dBFS)")}
    details: Dict[str, object] = {
        "raw": raw, "threshold": RAW_THRESHOLD, "clean_pause_fraction": clean_frac,
        "pause_floor_spread_db": spread, "n_pauses": len(pause_levels),
        "pause_seconds": float(pause.sum()) * 0.01, "loud_level_dbfs": loud,
        "persistent_lines_hz": lines[:20], "line_cue_skipped_lossy": lossy,
        "median_pause_level_dbfs": float(np.median(pause_levels)) if pause_levels else None,
    }
    return Evidence(detector=DETECTOR, title=TITLE, category=Category.AI_GENERATION, score=score,
                    confidence=conf, summary=summary, details=details, series=series)
