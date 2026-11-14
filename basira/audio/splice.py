"""audio.splice -- background consistency and sample-level discontinuities.

Every recording carries the acoustic "room" it was made in: the level and the
spectral shape of its background noise. Material spliced in from another
recording usually brings a different background, and a hard cut in a quiet
passage leaves a click.

Background change points
------------------------
The room is only audible in pauses, and only once the reverberation of the
last word has died away. So the detector does not compare arbitrary windows
(continuous speech would read as a "louder room"); it collects *pause
observations* and looks for a change of state between them:

* 20 ms frames (50 % overlap): level above ``FLOOR_BAND_MIN_HZ`` and 24
  log-spaced band levels.
* A frame is *settled* when the level stays within ``SETTLE_RANGE_DB`` over
  ``+-SETTLE_HALF_S`` -- a decaying reverberation tail or a syllable is not.
  A pause observation is a run of at least ``RUN_MIN_S`` of settled frames
  that lie ``MEASURABLE_DB`` or more below the loud level of the file. It
  yields a floor level and a band-shape fingerprint (level removed).
* Consecutive observations that agree are grouped into a *state*. A state
  supported by less than ``STATE_MIN_S`` of settled frames (a held fricative,
  a short background sound) is discarded.
* Two adjacent states differ when the fingerprint moves by more than
  ``max(SHAPE_SHIFT_DB, SHAPE_REL x this file's typical pause-to-pause
  difference)`` (RMS dB over bands), or when the level moves by more than
  ``max(FLOOR_SHIFT_DB, LEVEL_REL x typical)`` in at least
  ``LEVEL_MIN_PAUSES`` separate pauses on each side *and* the fingerprint moved
  at least ``LEVEL_NEEDS_SHAPE`` of the way to its own threshold. A level change
  alone is never enough: automatic gain control (every phone has one) moves the
  floor of an untouched recording by many dB, but leaves its spectral shape alone.
* ``raw`` = largest change / its threshold (1.0 = threshold). The edit lies
  between the last pause of the old state and the first pause of the new one,
  and that interval is what is reported.

What this cannot see: an insert that contains no settled pause (in a
reverberant room, short gaps between words never reach the floor), material
from a room with the same background, and anything in recordings without
pauses (music, continuous noise) -- there it reports "not measurable".

Clicks (waveform discontinuities)
---------------------------------
In quiet passages (floor-level frames at least ``CLICK_QUIET_BELOW_DB`` below
the loud level) an order-``LPC_ORDER`` linear predictor is fitted per block; an
isolated prediction-error spike larger than ``CLICK_RATIO`` x the block's
robust residual sigma is a discontinuity no acoustic source produces (a cut
between unrelated samples). Joins inside speech are deliberately *not*
tested: glottal pulses of a creaky voice produce the same isolated spikes
(13-21 sigma on synthetic vocal fry, measured with tests/synth_audio.py), so
the test would not stay quiet on clean speech.

Exactly-zero frames, and frames within ``SILENCE_GUARD_S`` of them, are
ignored here (``audio.silence`` handles them).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..core.types import Category, Evidence, Segment
from ..core.utils import calibrate
from ._dsp import SILENCE_DB, make_series, power_frames, runs

DETECTOR = "audio.splice"
TITLE = "Background / splice consistency"

MIN_SECONDS = 4.0
N_BANDS = 24
SILENCE_GUARD_S = 0.15       # frames this close to digital silence are ignored
FLOOR_PCT = 5.0              # display only: noise floor per 1 s window

# Pause observations. Measured on the dev set of benchmarks/audio_robustness.py
# (text-to-speech in simulated rooms, RT60 0.2-0.6 s, with and without AGC):
FLOOR_BAND_MIN_HZ = 250.0    # level for pause finding: above rumble and mains hum
SETTLE_HALF_S = 0.03         # a frame is settled if the level range over +-this ...
SETTLE_RANGE_DB = 3.0        # ... stays within this (stationary noise: about 1.5 dB)
RUN_MIN_S = 0.10             # shortest settled run that counts as a pause observation
OBS_MAX_S = 0.5              # longer pauses are split into observations of this length
MEASURABLE_DB = 15.0         # a pause lies at least this far below the loud level
STATE_MIN_S = 0.2            # settled time a state needs; held fricatives are shorter
MIN_PAUSES = 3               # fewer separate pauses than this: not measurable
# Change thresholds. Clean dev recordings: fingerprints of consecutive pauses differ
# by 0.3-1.4 dB RMS and their level by up to 2.5 dB (9 dB for a single pause under AGC).
SHAPE_SHIFT_DB = 3.0         # absolute minimum fingerprint change (RMS dB over bands)
FLOOR_SHIFT_DB = 5.0         # absolute minimum floor-level change
SHAPE_REL = 3.0              # ... and this many times the file's typical difference
LEVEL_REL = 4.0
LEVEL_MIN_PAUSES = 2         # separate pauses each state needs before its level counts
LEVEL_NEEDS_SHAPE = 0.5      # a level change counts only if the spectrum moved by at least
                             # this fraction of its own threshold: gain control (AGC), or a
                             # speaker moving closer, shifts the floor of a clean recording
                             # by 9-12 dB (simulated AGC, dev set) but not its spectrum
CHANGE_SOFTNESS = 0.25
CONFIDENCE_MAX = 0.6         # a background change is "Suspicious" at most: a real room can
                             # change too (a window opens, traffic), and no real recordings
                             # were available to measure how often

LPC_ORDER = 16
CLICK_BLOCK_S = 0.04
CLICK_QUIET_BELOW_DB = 25.0  # quiet = this far below the file's loud level
CLICK_RATIO = 10.0           # residual spike / robust sigma (Gaussian max over 10 min ~ 5.6)
CLICK_SOFTNESS = 1.5
CLICK_MAX_OUTLIERS = 3       # an isolated spike, not a noisy / impulsive block


def _band_frames(x: np.ndarray, sr: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """~20 ms frames, 50 % hop -> (times, total level dBFS, level above
    FLOOR_BAND_MIN_HZ, band levels dB, index of the first band above that frequency)."""
    nfft = int(2 ** np.round(np.log2(0.02 * sr)))
    hop = nfft // 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    nyq = sr / 2
    edges = np.geomspace(100.0, 0.95 * nyq, N_BANDS + 1)
    idx = np.searchsorted(freqs, edges)
    idx = np.maximum.accumulate(np.maximum(idx, 1))
    first = int(min(np.searchsorted(edges[:-1], FLOOR_BAND_MIN_HZ), N_BANDS - 4))
    lv, bands = [], []
    for _, p in power_frames(x, nfft, hop):
        cs = np.cumsum(p, axis=1, dtype=np.float64)
        b = cs[:, idx[1:] - 1] - cs[:, idx[:-1] - 1]
        lv.append(cs[:, -1])
        bands.append(b)
    tot = np.concatenate(lv)
    B = np.concatenate(bands)
    t = (np.arange(len(tot)) * hop + nfft / 2) / sr
    # power_frames normalises bins to (2/sum(w))^2; Parseval turns the bin sum into the
    # frame's mean square (dBFS): offset = sum(w)^2 / (2 * nfft * sum(w^2)).
    w = np.hanning(nfft)
    off = 10 * np.log10(w.sum() ** 2 / (2 * nfft * np.sum(w * w)))
    upper = B[:, first:].sum(axis=1)
    with np.errstate(divide="ignore"):
        ldb = np.where(tot > 0, 10 * np.log10(tot + 1e-30) + off, SILENCE_DB)
        sdb = np.where(upper > 0, 10 * np.log10(upper + 1e-30) + off, SILENCE_DB)
        bdb = 10 * np.log10(np.maximum(B, 1e-14))
    return t, ldb, sdb, bdb, first


def _pause_observations(t: np.ndarray, sdb: np.ndarray, bdb: np.ndarray, first: int,
                        dead: np.ndarray) -> Tuple[List[Dict[str, object]], float]:
    """Settled low-level runs -> [{t0, t1, level, shape, n, pause}], loud level."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d

    ok = ~dead & (sdb > SILENCE_DB + 1)
    if ok.sum() < 20 or len(t) < 2:
        return [], 0.0
    hop_s = float(t[1] - t[0])
    loud = float(np.percentile(sdb[ok], 95))
    k = max(1, int(round(SETTLE_HALF_S / hop_s)))
    lvl = np.where(ok, sdb, np.inf)                   # dead frames break a run
    rng = maximum_filter1d(lvl, 2 * k + 1, mode="nearest") \
        - minimum_filter1d(np.where(ok, sdb, -np.inf), 2 * k + 1, mode="nearest")
    cand = ok & (rng <= SETTLE_RANGE_DB) & (sdb <= loud - MEASURABLE_DB)
    st, en = runs(cand)
    min_run = max(3, int(round(RUN_MIN_S / hop_s)))
    max_obs = max(min_run, int(round(OBS_MAX_S / hop_s)))
    obs: List[Dict[str, object]] = []
    pause, last_end = 0, None
    speech = ok & (sdb > loud - MEASURABLE_DB)
    for a, b in zip(st, en):
        if b - a < min_run:
            continue
        # a new pause only if something loud happened since the previous run
        if last_end is not None and speech[last_end:a].any():
            pause += 1
        last_end = b
        parts = max(1, int(round((b - a) / max_obs)))
        cuts = np.linspace(a, b, parts + 1).astype(int)
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            sh = bdb[c0:c1, first:].mean(axis=0)
            obs.append({"t0": float(t[c0]), "t1": float(t[c1 - 1]),
                        "level": float(np.median(sdb[c0:c1])), "shape": sh - sh.mean(),
                        "n": int(c1 - c0), "pause": pause})
    return obs, loud


def _shape_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _states(obs: List[Dict[str, object]], thr_l: float, thr_s: float, min_frames: int
            ) -> List[Dict[str, object]]:
    """Group consecutive agreeing observations; drop states with too little support."""
    def group(items):
        out: List[Dict[str, object]] = []
        for o in items:
            if out:
                cur = out[-1]
                if abs(o["level"] - cur["level"]) < thr_l \
                        and _shape_dist(o["shape"], cur["shape"]) < thr_s:
                    w0, w1 = cur["n"], o["n"]
                    cur["level"] = (cur["level"] * w0 + o["level"] * w1) / (w0 + w1)
                    cur["shape"] = (cur["shape"] * w0 + o["shape"] * w1) / (w0 + w1)
                    cur["n"] = w0 + w1
                    cur["t1"] = o["t1"]
                    cur["pauses"].add(o["pause"])
                    cur["members"].append(o)
                    continue
            out.append({"t0": o["t0"], "t1": o["t1"], "level": o["level"],
                        "shape": np.array(o["shape"], dtype=np.float64), "n": o["n"],
                        "pauses": {o["pause"]}, "members": [o]})
        return out

    items = list(obs)
    for _ in range(8):
        states = group(items)
        weak = [s for s in states if s["n"] < min_frames]
        if not weak or len(states) == len(weak):
            break
        items = [o for s in states if s["n"] >= min_frames for o in s["members"]]
    return [s for s in states if s["n"] >= min_frames]


def _background_changes(t, sdb, bdb, first, dead):
    """-> (events [(t_a, t_b, strength, label)], raw strength, info dict)."""
    obs, loud = _pause_observations(t, sdb, bdb, first, dead)
    n_pauses = len({o["pause"] for o in obs})
    info: Dict[str, object] = {"observations": obs, "n_pauses": n_pauses, "loud": loud,
                               "thr_level": FLOOR_SHIFT_DB, "thr_shape": SHAPE_SHIFT_DB,
                               "states": []}
    if n_pauses < MIN_PAUSES:
        return [], 0.0, info
    # This file's typical difference between consecutive pauses (robust: median).
    firsts = [o for i, o in enumerate(obs) if i == 0 or obs[i - 1]["pause"] != o["pause"]]
    d_l = [abs(b["level"] - a["level"]) for a, b in zip(firsts[:-1], firsts[1:])]
    d_s = [_shape_dist(a["shape"], b["shape"]) for a, b in zip(firsts[:-1], firsts[1:])]
    thr_l = max(FLOOR_SHIFT_DB, LEVEL_REL * float(np.median(d_l))) if len(d_l) >= 4 \
        else FLOOR_SHIFT_DB
    thr_s = max(SHAPE_SHIFT_DB, SHAPE_REL * float(np.median(d_s))) if len(d_s) >= 4 \
        else SHAPE_SHIFT_DB
    hop_s = float(t[1] - t[0])
    states = _states(obs, thr_l, thr_s, max(4, int(round(STATE_MIN_S / hop_s))))
    info.update(thr_level=thr_l, thr_shape=thr_s, states=states)
    events: List[Tuple[float, float, float, str]] = []
    raw = 0.0
    for a, b in zip(states[:-1], states[1:]):
        d_level = b["level"] - a["level"]
        d_shape = _shape_dist(a["shape"], b["shape"])
        s_shape = d_shape / thr_s
        level_ok = min(len(a["pauses"]), len(b["pauses"])) >= LEVEL_MIN_PAUSES
        s_level = abs(d_level) / thr_l if level_ok else 0.0
        # A level change supports a spectral change but cannot replace it.
        strength = max(s_shape, min(s_level, s_shape / LEVEL_NEEDS_SHAPE))
        raw = max(raw, strength)
        if strength >= 1.0:
            parts = [f"background spectrum {d_shape:.1f} dB RMS"]
            if s_level >= 1.0:
                parts.insert(0, f"background level {d_level:+.1f} dB")
            events.append((a["t1"], b["t0"], strength,
                           "background changes (" + ", ".join(parts) + ")"))
    return events, raw, info


def _clicks(x: np.ndarray, sr: int, t, ldb, loud: float
            ) -> Tuple[List[Tuple[float, float]], float, int]:
    """Isolated LPC-residual spikes in quiet context.

    Returns ([(time, ratio), ...] for candidate spikes, the largest isolated
    ratio over all analysed blocks, number of blocks analysed).
    """
    from scipy.linalg import solve_toeplitz
    from scipy.signal import lfilter

    blk = int(CLICK_BLOCK_S * sr)
    if blk <= 4 * LPC_ORDER:
        return [], 0.0, 0
    quiet_frame = (ldb < loud - CLICK_QUIET_BELOW_DB) & (ldb > SILENCE_DB + 1)
    n_blocks = (len(x) - LPC_ORDER) // blk
    if n_blocks <= 0:
        return [], 0.0, 0
    bt = (np.arange(n_blocks) * blk + LPC_ORDER + blk / 2) / sr
    # a block is analysed when the frames around it (+-60 ms) are all quiet
    fq = np.interp(bt, t, quiet_frame.astype(float))
    fq_lo = np.interp(bt - 0.06, t, quiet_frame.astype(float))
    fq_hi = np.interp(bt + 0.06, t, quiet_frame.astype(float))
    sel = np.flatnonzero((fq > 0.99) & (fq_lo > 0.99) & (fq_hi > 0.99))
    out = []
    best = 0.0
    for bi in sel:
        s0 = bi * blk + LPC_ORDER
        seg = x[s0 - LPC_ORDER:s0 + blk].astype(np.float64)
        body = seg[LPC_ORDER:]
        if not np.any(body):
            continue
        r = np.correlate(body, body, "full")[len(body) - 1:len(body) + LPC_ORDER]
        if r[0] <= 0:
            continue
        r[0] *= 1.0 + 1e-6
        try:
            a = solve_toeplitz(r[:LPC_ORDER], r[1:LPC_ORDER + 1])
        except Exception:
            continue
        e = lfilter(np.concatenate([[1.0], -a]), [1.0], seg)[LPC_ORDER:]
        sig = 1.4826 * np.median(np.abs(e)) + 1e-12
        ratio = np.abs(e) / sig
        j = int(np.argmax(ratio))
        if np.sum(ratio > 6.0) <= CLICK_MAX_OUTLIERS:
            best = max(best, float(ratio[j]))
            if ratio[j] >= CLICK_RATIO / 2:
                out.append(((s0 + j) / sr, float(ratio[j])))
    return out, best, int(len(sel))


def detect_splice(x: np.ndarray, sr: int) -> Evidence:
    dur = len(x) / sr
    if dur < MIN_SECONDS:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.MANIPULATION,
                                       f"too short ({dur:.1f} s < {MIN_SECONDS:.0f} s)")
    t, ldb, sdb, bdb, first = _band_frames(x, sr)
    # Digital silence belongs to audio.silence. Frames right next to it are also
    # excluded: codecs smear the silence edges into near-zero frames that would
    # otherwise read as a "new room".
    from scipy.ndimage import binary_dilation

    dead = ldb <= SILENCE_DB + 1
    if dead.any():
        k = int(round(SILENCE_GUARD_S / (t[1] - t[0]))) if len(t) > 1 else 1
        dead = binary_dilation(dead, structure=np.ones(2 * k + 1, bool))
        ldb = np.where(dead, SILENCE_DB, ldb)
    ok = ldb > SILENCE_DB + 1
    if ok.sum() < 50:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.MANIPULATION,
                                       "no non-silent audio")
    changes, raw_change, info = _background_changes(t, sdb, bdb, first, dead)
    obs = info["observations"]
    n_pauses = int(info["n_pauses"])
    measurable = n_pauses >= MIN_PAUSES

    events: List[Tuple[float, float, float, str]] = [
        (a_, b_, calibrate(s_, 1.0, CHANGE_SOFTNESS), lab) for a_, b_, s_, lab in changes]

    loud = float(np.percentile(ldb[ok], 95))
    clicks, raw_click, n_click_blocks = _clicks(x, sr, t, ldb, loud)
    for tc, rc in clicks:
        if rc >= CLICK_RATIO:
            events.append((tc - 0.02, tc + 0.02, calibrate(rc, CLICK_RATIO, CLICK_SOFTNESS),
                           f"waveform discontinuity (prediction error {rc:.0f} sigma)"))

    score = max(calibrate(raw_change, 1.0, CHANGE_SOFTNESS) if measurable else 0.0,
                calibrate(raw_click, CLICK_RATIO, CLICK_SOFTNESS) if n_click_blocks else 0.0)
    events.sort()
    segments = [Segment(max(0.0, float(a_) - 0.1), min(dur, float(b_) + 0.1), float(sc), lab)
                for a_, b_, sc, lab in events]

    # Trust grows with the number of pauses in which the room could be observed:
    # one usable pause every 4 s is as good as it gets.
    density = min(1.0, n_pauses / max(dur / 4.0, 1.0)) if measurable else 0.0
    confidence = float(np.clip(0.25 + (CONFIDENCE_MAX - 0.25) * density, 0.25, CONFIDENCE_MAX))
    markers = [0.5 * (e[0] + e[1]) for e in events]
    # per ~1 s window noise floor for display
    wt = np.arange(0.5, dur - 0.5 + 1e-9, 0.5)
    a, b = np.searchsorted(t, wt - 0.5), np.searchsorted(t, wt + 0.5)
    wfloor = np.array([np.percentile(ldb[i:j][ok[i:j]], FLOOR_PCT) if ok[i:j].sum() > 5
                       else np.nan for i, j in zip(a, b)])
    series = {
        "noise_floor": make_series(wt, np.nan_to_num(wfloor, nan=SILENCE_DB),
                                   "noise floor per 1 s window (dBFS)", markers),
    }
    if obs:
        series["pause_floor"] = make_series(
            [0.5 * (o["t0"] + o["t1"]) for o in obs], [o["level"] for o in obs],
            f"room level in settled pauses (dBFS above {FLOOR_BAND_MIN_HZ:.0f} Hz)", markers)
    blind = (" Only pauses of about 0.1 s or longer that are free of reverberation show the "
             "room; an insert without such a pause, or from a room with the same background, "
             "is invisible to this test.")
    if events:
        summary = (f"{len(events)} inconsistenc{'y' if len(events) == 1 else 'ies'} found: "
                   + "; ".join(f"{lab} between {a_:.2f} and {b_:.2f} s"
                               for a_, b_, _, lab in events[:4])
                   + f". Strongest background change {raw_change:.2f} (threshold 1.0, from "
                     f"{n_pauses} pauses), strongest click {raw_click:.1f} sigma (threshold "
                     f"{CLICK_RATIO:.0f}). A real change in the room (a window opened, a "
                     f"machine switched on) looks the same as spliced material.")
    elif measurable:
        summary = (f"Background is consistent across the {n_pauses} pauses in which the room "
                   f"could be observed: strongest change {raw_change:.2f} (threshold 1.0), "
                   f"strongest click {raw_click:.1f} sigma (threshold {CLICK_RATIO:.0f})."
                   + blind)
    else:
        summary = (f"Background not measurable: only {n_pauses} settled pause(s) found (needs "
                   f"{MIN_PAUSES}); continuous speech, music or noise leaves nothing to "
                   f"compare. Strongest click {raw_click:.1f} sigma (threshold "
                   f"{CLICK_RATIO:.0f}). This is not a finding of consistency.")
    details: Dict[str, object] = {
        "raw_change_strength": raw_change, "change_threshold": 1.0,
        "raw_click_ratio": raw_click, "click_threshold": CLICK_RATIO,
        "click_blocks_analysed": n_click_blocks,
        "pauses_observed": n_pauses, "pause_observations": len(obs),
        "background_measurable": bool(measurable),
        "level_threshold_db": float(info["thr_level"]),
        "shape_threshold_db": float(info["thr_shape"]),
        "states": [{"start": s["t0"], "end": s["t1"], "level_db": s["level"],
                    "settled_frames": s["n"], "pauses": len(s["pauses"])}
                   for s in info["states"]][:50],
        "loud_level_dbfs": loud,
        "median_floor_dbfs": float(np.nanmedian(wfloor)) if np.isfinite(wfloor).any() else None,
        "events": [{"start": e[0], "end": e[1], "score": e[2], "label": e[3]} for e in events],
    }
    return Evidence(detector=DETECTOR, title=TITLE, category=Category.MANIPULATION, score=score,
                    confidence=confidence, summary=summary, details=details,
                    segments=segments, series=series)
