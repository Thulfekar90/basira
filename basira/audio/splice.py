"""audio.splice -- background consistency and sample-level discontinuities.

Every recording carries the acoustic "room" it was made in: the level and the
spectral shape of its background noise. Material spliced in from another
recording usually brings a different background, and a hard cut often leaves a
click.

Background change points
------------------------
* 20 ms frames (50 % overlap): total level and 24 log-spaced band levels.
* The file is divided into 0.5 s blocks. At every block boundary, the
  ``SPAN_S`` seconds on each side are summarised by their *noise floor*
  (``FLOOR_PCT`` percentile of frame levels -- i.e. the pauses) and their
  *room fingerprint* (mean band-level shape of the quietest ``QUIET_PCT`` % of
  frames that lie within ``FP_MARGIN_DB`` of the floor, level removed).
* A boundary is compared only if both sides are "measurable": their floor is at
  least ``MEASURABLE_DB`` below the loud level of the file (continuous speech
  without pauses says nothing about the room).
* Change strength = max over {floor shift, fingerprint distance} of
  ``min(|delta| / absolute threshold, robust z / CHANGE_Z)``: a change must be
  both large in absolute terms *and* unusual for this file. ``raw >= 1`` is a
  change point; the location is refined at 10 ms resolution with a sliding
  low-percentile level.

Clicks (waveform discontinuities)
---------------------------------
In quiet passages (floor-level frames at least ``CLICK_QUIET_BELOW_DB`` below
the loud level) an order-``LPC_ORDER`` linear predictor is fitted per block; an
isolated prediction-error spike larger than ``CLICK_RATIO`` x the block's
robust residual sigma is a discontinuity no acoustic source produces (a cut
between unrelated samples).

Exactly-zero frames, and frames within ``SILENCE_GUARD_S`` of them, are
ignored here (``audio.silence`` handles them).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..core.types import Category, Evidence, Segment
from ..core.utils import calibrate
from ._dsp import SILENCE_DB, make_series, merge_close, power_frames

DETECTOR = "audio.splice"
TITLE = "Background / splice consistency"

MIN_SECONDS = 4.0
BLOCK_S = 0.5
SPAN_S = 2.0                 # context on each side of a candidate boundary
FLOOR_PCT = 5.0
QUIET_PCT = 20.0
FP_MARGIN_DB = 4.0           # fingerprint frames: at most this far above the floor
FP_MIN_FRAMES = 10           # (>= 100 ms of pause per side)
SILENCE_GUARD_S = 0.15       # frames this close to digital silence are ignored
FLOOR_MATCH_DB = 3.0         # a frame "shows" a floor when within this of its level
N_BANDS = 24
MEASURABLE_DB = 15.0
FLOOR_SHIFT_DB = 4.0         # absolute minimum floor change
SHAPE_SHIFT_DB = 3.0         # absolute minimum fingerprint (RMS dB) change
CHANGE_Z = 5.0               # robust z needed among all boundaries of this file
CHANGE_SOFTNESS = 0.25

LPC_ORDER = 16
CLICK_BLOCK_S = 0.04
CLICK_QUIET_BELOW_DB = 25.0  # quiet = this far below the file's loud level
CLICK_RATIO = 10.0           # residual spike / robust sigma (Gaussian max over 10 min ~ 5.6)
CLICK_SOFTNESS = 1.5
CLICK_MAX_OUTLIERS = 3       # an isolated spike, not a noisy / impulsive block


def _band_frames(x: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame times, frame level (dBFS) and band levels (dB) for ~20 ms frames, 50% hop."""
    nfft = int(2 ** np.round(np.log2(0.02 * sr)))
    hop = nfft // 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    nyq = sr / 2
    edges = np.geomspace(100.0, 0.95 * nyq, N_BANDS + 1)
    idx = np.searchsorted(freqs, edges)
    idx = np.maximum.accumulate(np.maximum(idx, 1))
    lv, bands = [], []
    for _, p in power_frames(x, nfft, hop):
        cs = np.cumsum(p, axis=1, dtype=np.float64)
        b = cs[:, idx[1:] - 1] - cs[:, idx[:-1] - 1]
        tot = cs[:, -1]
        lv.append(tot)
        bands.append(b)
    tot = np.concatenate(lv)
    B = np.concatenate(bands)
    t = (np.arange(len(tot)) * hop + nfft / 2) / sr
    # power_frames normalises bins to (2/sum(w))^2; Parseval turns the bin sum into the
    # frame's mean square (dBFS): offset = sum(w)^2 / (2 * nfft * sum(w^2)).
    w = np.hanning(nfft)
    off = 10 * np.log10(w.sum() ** 2 / (2 * nfft * np.sum(w * w)))
    with np.errstate(divide="ignore"):
        ldb = np.where(tot > 0, 10 * np.log10(tot + 1e-30) + off, SILENCE_DB)
        bdb = 10 * np.log10(np.maximum(B, 1e-14))
    return t, ldb, bdb


def _boundary_stats(t, ldb, bdb, dur):
    """Floor shift and fingerprint distance at every 0.5 s block boundary."""
    ok = ldb > SILENCE_DB + 1
    loud = float(np.percentile(ldb[ok], 95)) if ok.any() else 0.0
    bnds = np.arange(SPAN_S, dur - SPAN_S + 1e-9, BLOCK_S)
    out = {"t": bnds, "dfloor": np.full(len(bnds), np.nan), "dshape": np.full(len(bnds), np.nan),
           "floor_l": np.full(len(bnds), np.nan), "floor_r": np.full(len(bnds), np.nan)}
    starts = np.searchsorted(t, bnds - SPAN_S)
    mids = np.searchsorted(t, bnds)
    ends = np.searchsorted(t, bnds + SPAN_S)
    for k in range(len(bnds)):
        side = []
        for a, b in ((starts[k], mids[k]), (mids[k], ends[k])):
            sel = ok[a:b]
            if sel.sum() < 20:
                side.append(None)
                continue
            lv = ldb[a:b][sel]
            fl = float(np.percentile(lv, FLOOR_PCT))
            # room fingerprint from pause frames only: within FP_MARGIN_DB of the floor
            q = lv <= min(np.percentile(lv, QUIET_PCT), fl + FP_MARGIN_DB)
            fp = bdb[a:b][sel][q].mean(axis=0) if q.sum() >= FP_MIN_FRAMES else None
            side.append((fl, fp))
        if side[0] is None or side[1] is None:
            continue
        (fl, fpl), (fr, fpr) = side
        out["floor_l"][k], out["floor_r"][k] = fl, fr
        if fl > loud - MEASURABLE_DB or fr > loud - MEASURABLE_DB:
            continue
        out["dfloor"][k] = fr - fl
        if fpl is not None and fpr is not None:
            d = fpr - fpl                   # shape only: remove the overall level change
            out["dshape"][k] = float(np.sqrt(np.mean((d - d.mean()) ** 2)))
    out["loud"] = loud
    return out


def _robust_z_pos(v: np.ndarray) -> np.ndarray:
    f = np.isfinite(v)
    z = np.zeros_like(v)
    if f.sum() < 5:
        return z
    med = np.median(v[f])
    mad = 1.4826 * np.median(np.abs(v[f] - med)) + 1e-6
    z[f] = (v[f] - med) / mad
    return z


def _change_interval(t, ldb, t0: float, fl_left: float, fl_right: float
                     ) -> Tuple[float, float]:
    """Bracket a background-level change between the last pause frame of the old
    floor and the first pause frame of the new one.

    A room floor is only observable in pauses, so the honest localisation of a
    change is an interval: it happened after the last frame where the old floor
    was seen and before the first frame where the new floor is seen. Frames at
    the *quieter* floor's level cannot occur inside the louder room, which is
    what anchors the bracket.
    """
    ok = ldb > SILENCE_DB + 1
    q, lo_ = min(fl_left, fl_right), max(fl_left, fl_right)
    is_q = ok & (np.abs(ldb - q) <= FLOOR_MATCH_DB)
    is_l = ok & (np.abs(ldb - lo_) <= FLOOR_MATCH_DB)
    if fl_right > fl_left:      # rising: quiet room on the left
        cand = np.flatnonzero(is_q & (t >= t0 - 3.0) & (t <= t0 + 1.0))
        if cand.size == 0:
            return t0 - 0.25, t0 + 0.25
        a_ = t[cand[-1]]
        nxt = np.flatnonzero(is_l & (t > a_) & (t <= a_ + 3.0))
        b_ = t[nxt[0]] if nxt.size else a_ + 0.5
    else:                       # falling: quiet room on the right
        cand = np.flatnonzero(is_q & (t >= t0 - 1.0) & (t <= t0 + 3.0))
        if cand.size == 0:
            return t0 - 0.25, t0 + 0.25
        b_ = t[cand[0]]
        prv = np.flatnonzero(is_l & (t < b_) & (t >= b_ - 3.0))
        a_ = t[prv[-1]] if prv.size else b_ - 0.5
    return float(a_), float(b_)


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
    t, ldb, bdb = _band_frames(x, sr)
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
    st = _boundary_stats(t, ldb, bdb, dur)
    df, ds = st["dfloor"], st["dshape"]
    measurable = np.isfinite(df)
    z_f = _robust_z_pos(np.abs(df))
    z_s = _robust_z_pos(ds)
    with np.errstate(invalid="ignore"):
        s_floor = np.minimum(np.abs(df) / FLOOR_SHIFT_DB, z_f / CHANGE_Z)
        s_shape = np.minimum(ds / SHAPE_SHIFT_DB, z_s / CHANGE_Z)
    strength = np.nan_to_num(np.maximum(s_floor, s_shape), nan=0.0)
    if measurable.sum() < 5:
        strength[:] = 0.0
    raw_change = float(strength.max()) if strength.size else 0.0

    events: List[Tuple[float, float, str]] = []
    cand = np.flatnonzero(strength >= 1.0)
    if cand.size:
        for i in merge_close(st["t"][cand], strength[cand], SPAN_S):
            k = cand[i]
            # Low-percentile statistics lock onto the quieter side, so a change shows
            # as a plateau of boundaries; a rise starts at the plateau's first
            # boundary, a fall at its last.
            rising = np.nan_to_num(df[k]) > 0
            near = np.flatnonzero((np.abs(st["t"] - st["t"][k]) <= SPAN_S)
                                  & (strength >= 0.8 * strength[k]))
            tb = float(st["t"][near[0] if rising else near[-1]])
            if abs(np.nan_to_num(df[k])) >= FLOOR_SHIFT_DB:
                lo_t, hi_t = _change_interval(t, ldb, tb, st["floor_l"][k], st["floor_r"][k])
            else:
                lo_t, hi_t = tb - 0.25, tb + 0.25
            parts = []
            if abs(np.nan_to_num(df[k])) >= FLOOR_SHIFT_DB:
                parts.append(f"background level {df[k]:+.1f} dB")
            if np.nan_to_num(ds[k]) >= SHAPE_SHIFT_DB:
                parts.append(f"background spectrum {ds[k]:.1f} dB RMS")
            lab = "background changes (" + ", ".join(parts or ["unusual for this file"]) + ")"
            events.append((lo_t, hi_t, calibrate(strength[k], 1.0, CHANGE_SOFTNESS), lab))

    loud = st["loud"]
    clicks, raw_click, n_click_blocks = _clicks(x, sr, t, ldb, loud)
    for tc, rc in clicks:
        if rc >= CLICK_RATIO:
            events.append((tc - 0.02, tc + 0.02, calibrate(rc, CLICK_RATIO, CLICK_SOFTNESS),
                           f"waveform discontinuity (prediction error {rc:.0f} sigma)"))

    score = max(calibrate(raw_change, 1.0, CHANGE_SOFTNESS) if measurable.sum() >= 5 else 0.0,
                calibrate(raw_click, CLICK_RATIO, CLICK_SOFTNESS) if n_click_blocks else 0.0)
    events.sort()
    segments = [Segment(max(0.0, float(a_) - 0.1), min(dur, float(b_) + 0.1), float(sc), lab)
                for a_, b_, sc, lab in events]

    frac_meas = float(measurable.mean()) if measurable.size else 0.0
    confidence = float(np.clip(0.25 + 0.6 * frac_meas, 0.25, 0.8))
    markers = [0.5 * (e[0] + e[1]) for e in events]
    # per ~1 s window noise floor for display
    wt = np.arange(0.5, dur - 0.5 + 1e-9, 0.5)
    a, b = np.searchsorted(t, wt - 0.5), np.searchsorted(t, wt + 0.5)
    wfloor = np.array([np.percentile(ldb[i:j][ok[i:j]], FLOOR_PCT) if ok[i:j].sum() > 5
                       else np.nan for i, j in zip(a, b)])
    series = {
        "noise_floor": make_series(wt, np.nan_to_num(wfloor, nan=SILENCE_DB),
                                   "noise floor per 1 s window (dBFS)", markers),
        "change_strength": make_series(st["t"], strength, "change strength (1 = threshold)",
                                       markers),
    }
    if events:
        summary = (f"{len(events)} inconsistenc{'y' if len(events) == 1 else 'ies'} found: "
                   + "; ".join(f"{lab} between {a_:.2f} and {b_:.2f} s"
                               for a_, b_, _, lab in events[:4])
                   + f". Strongest background change {raw_change:.2f} (threshold 1.0), "
                     f"strongest click {raw_click:.1f} sigma (threshold {CLICK_RATIO:.0f}).")
    else:
        summary = (f"Background noise floor and room spectrum are consistent: strongest change "
                   f"{raw_change:.2f} (threshold 1.0), strongest click {raw_click:.1f} sigma "
                   f"(threshold {CLICK_RATIO:.0f}).")
    if frac_meas < 0.3:
        summary += " Few pauses to measure the background, so confidence is low."
    details: Dict[str, object] = {
        "raw_change_strength": raw_change, "change_threshold": 1.0,
        "raw_click_ratio": raw_click, "click_threshold": CLICK_RATIO,
        "click_blocks_analysed": n_click_blocks,
        "measurable_fraction": frac_meas, "loud_level_dbfs": loud,
        "median_floor_dbfs": float(np.nanmedian(wfloor)) if np.isfinite(wfloor).any() else None,
        "events": [{"start": e[0], "end": e[1], "score": e[2], "label": e[3]} for e in events],
    }
    return Evidence(detector=DETECTOR, title=TITLE, category=Category.MANIPULATION, score=score,
                    confidence=confidence, summary=summary, details=details,
                    segments=segments, series=series)
