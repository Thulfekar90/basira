"""audio.silence -- digital silence embedded in a live recording.

A microphone never outputs a run of identical samples: even a "silent" room
produces a noise floor of tens of LSB. A run of exact zeros (or of one
constant value) lasting ``MIN_RUN_S`` or longer *inside* a recording whose
surroundings do have a floor is therefore a digital artefact -- typically a
deleted passage replaced by silence, an inserted gap, or a pasted segment
that was gated.

Method
------
* Sample-level runs where ``|x[n] - x[n-1]| <= FLAT_EPS`` lasting
  ``MIN_RUN_S`` or longer; runs touching the start / end of the file
  (leading / trailing silence) are ignored.
* For each run, the *context floor* is the ``CONTEXT_PCT`` percentile of 20 ms
  frame levels within ``CONTEXT_S`` seconds on both sides (non-silent frames
  only). ``raw`` = the loudest such context floor (dBFS):
  ``score = calibrate(raw, CONTEXT_FLOOR_DB, CONTEXT_SOFTNESS_DB)``. A context
  floor near the 16-bit dither level (around -96 dBFS) can legitimately
  quantise to zeros; a floor of -60 dBFS cannot.
* If digital silence is the *norm* for this file's pauses (at least
  ``SYSTEMATIC_FRACTION`` of all pause time), the file was gated or
  synthesised as a whole; isolated runs are then not evidence of an edit and
  the score is capped (``audio.synthetic`` uses that cue instead).
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.types import Category, Evidence, Segment
from ..core.utils import calibrate
from ._dsp import SILENCE_DB, frame_levels, make_series, runs

DETECTOR = "audio.silence"
TITLE = "Embedded digital silence"

FLAT_EPS = 2e-6              # ~1/16 LSB at 16 bit: "identical" samples
MIN_RUN_S = 0.02
CONTEXT_S = 1.5
CONTEXT_PCT = 10.0
MIN_CONTEXT_FRAMES = 10
CONTEXT_FLOOR_DB = -85.0     # context floor above this cannot quantise to zeros
CONTEXT_SOFTNESS_DB = 5.0
PAUSE_BELOW_LOUD_DB = 30.0   # pause frames: this far below the loud level
SYSTEMATIC_FRACTION = 0.5
SYSTEMATIC_MIN_RUNS = 3
SYSTEMATIC_SCORE_CAP = 0.35


def detect_silence(x: np.ndarray, sr: int) -> Evidence:
    n = len(x)
    dur = n / sr
    if dur < 0.5:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.MANIPULATION, "too short")
    flat = np.abs(np.diff(x)) <= FLAT_EPS
    st, en = runs(flat)              # flat[k] covers samples k, k+1
    lens = en - st + 1               # samples
    keep = lens >= int(MIN_RUN_S * sr)
    st, en = st[keep], en[keep]
    nz = np.flatnonzero(~flat)
    if nz.size == 0:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.MANIPULATION,
                                       "the whole file is constant / digital silence")
    first, last = nz[0], nz[-1] + 1
    inner = (st > first) & (en < last)
    lead_trail = int((~inner).sum())
    st, en = st[inner], en[inner]

    t, db, _ = frame_levels(x, sr)
    ok = db > SILENCE_DB + 1
    loud = float(np.percentile(db[ok], 95)) if ok.any() else SILENCE_DB
    # frames overlapping any run are excluded from context estimates
    in_run = np.zeros(len(t), bool)
    for a, b in zip(st, en):
        in_run[(t >= a / sr - 0.01) & (t <= b / sr + 0.01)] = True
    usable = ok & ~in_run

    run_info = []
    for a, b in zip(st, en):
        t0, t1 = float(a / sr), float((b + 1) / sr)
        ctx = usable & (((t >= t0 - CONTEXT_S) & (t < t0)) | ((t > t1) & (t <= t1 + CONTEXT_S)))
        floor = float(np.percentile(db[ctx], CONTEXT_PCT)) if ctx.sum() >= MIN_CONTEXT_FRAMES \
            else float("nan")
        run_info.append({"start": t0, "end": t1, "value": float(x[a]), "context_floor_db": floor})

    # how much of this file's pause time is digital silence?
    span = (t >= first / sr) & (t <= last / sr)
    pause_live = float(np.sum(usable & span & (db < loud - PAUSE_BELOW_LOUD_DB))) * 0.01
    flat_time = float(sum(r["end"] - r["start"] for r in run_info))
    frac = flat_time / (flat_time + pause_live) if flat_time + pause_live > 0 else 0.0
    systematic = frac >= SYSTEMATIC_FRACTION and len(run_info) >= SYSTEMATIC_MIN_RUNS

    floors = [r["context_floor_db"] for r in run_info if np.isfinite(r["context_floor_db"])]
    raw = max(floors) if floors else float("-inf")
    score = calibrate(raw, CONTEXT_FLOOR_DB, CONTEXT_SOFTNESS_DB) if floors else 0.0
    segments: List[Segment] = []
    for r in run_info:
        if np.isfinite(r["context_floor_db"]):
            sc = calibrate(r["context_floor_db"], CONTEXT_FLOOR_DB, CONTEXT_SOFTNESS_DB)
            if systematic:
                sc = min(sc, SYSTEMATIC_SCORE_CAP)
            if sc >= 0.5:
                segments.append(Segment(r["start"], r["end"], sc,
                                        f"digital silence {1000 * (r['end'] - r['start']):.0f} ms "
                                        f"(context floor {r['context_floor_db']:.0f} dBFS)"))
    if not run_info:
        summary = (f"No embedded digital silence (runs of identical samples >= "
                   f"{MIN_RUN_S * 1000:.0f} ms) inside the recording"
                   + (f"; {lead_trail} leading/trailing run(s) ignored." if lead_trail else "."))
        confidence = 0.7
    elif systematic:
        score = min(score, SYSTEMATIC_SCORE_CAP)
        confidence = 0.3
        summary = (f"{len(run_info)} digital-silence runs, but they make up {frac:.0%} of all "
                   f"pause time: the whole file was noise-gated or synthesised, so individual "
                   f"runs are not evidence of an insertion (see the synthetic-speech cue).")
    else:
        confidence = 0.9 if floors else 0.4
        worst = max(run_info, key=lambda r: np.nan_to_num(r["context_floor_db"], nan=-999))
        summary = (f"{len(run_info)} run(s) of exactly identical samples inside the recording "
                   f"(total {flat_time * 1000:.0f} ms, {frac:.0%} of pause time). The loudest "
                   f"surrounding noise floor is {raw:.0f} dBFS (threshold {CONTEXT_FLOOR_DB:.0f}); "
                   f"a live microphone cannot drop to digital zero there, which points to an "
                   f"inserted or deleted passage, e.g. at {worst['start']:.2f} s.")
        if not floors:
            summary = (f"{len(run_info)} run(s) of identical samples, but the surrounding floor "
                       f"could not be measured.")
    # display: frame level with the runs as markers
    series = {"frame_level": make_series(t, np.maximum(db, -140.0), "frame level (dBFS)",
                                         [r["start"] for r in run_info[:50]])}
    details = {
        "raw_context_floor_db": raw if floors else None, "threshold_db": CONTEXT_FLOOR_DB,
        "n_runs": len(run_info), "ignored_leading_trailing": lead_trail,
        "digital_silence_s": flat_time, "pause_fraction_digital": frac,
        "systematic": systematic, "runs": run_info[:100],
    }
    return Evidence(detector=DETECTOR, title=TITLE, category=Category.MANIPULATION, score=score,
                    confidence=confidence, summary=summary, details=details,
                    segments=segments, series=series)
