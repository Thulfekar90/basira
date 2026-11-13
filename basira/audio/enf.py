"""audio.enf -- Electrical Network Frequency (mains hum) continuity.

Recordings made near mains-powered equipment pick up a faint 50 / 60 Hz hum
whose frequency wanders slowly (grid load) but *continuously*: the hum phase
cannot jump. Deleting or inserting audio breaks that continuity:

* a cut of ``d`` seconds shifts the hum phase by ``2*pi*f*d`` (mod 2*pi), e.g. a
  0.73 s cut at 50 Hz is a pi jump;
* a splice of material recorded at another time / place brings a different
  ENF frequency or no hum at all.

Method
------
1. Decimate to 1 kHz; measure narrow-band SNR of the 50, 100, 60 and 120 Hz
   lines (peak within +-0.15 Hz * k versus the median of 1..4 Hz * k away).
   Below ``HUM_SNR_MIN_DB`` -> not applicable.
2. Narrow Butterworth band-pass (``BW_HZ * k`` wide) around the measured hum
   line, Hilbert analytic signal, instantaneous phase minus the nominal
   carrier -> slowly varying phase residual.
3. Hum dropouts: the hum envelope falling ``DROPOUT_DB`` below its median for
   ``DROPOUT_MIN_S`` or longer (material recorded without hum was spliced in,
   or the hum was filtered locally). Phase / frequency statistics are masked
   inside dropouts, where they are meaningless.
4. Phase discontinuity statistic: at every instant, fit a line to the phase
   residual on each side (``FIT_S`` long, ``GUARD_S`` away to skip the filter
   transient) and compare both extrapolations at that instant. Smooth ENF
   drift is linear on that scale; a cut leaves a step. The statistic is
   wrapped to (-pi, pi]. Threshold: ``max(PHASE_JUMP_MIN_RAD, PHASE_Z *
   robust sigma)``.
5. ENF frequency track: zoom-DFT peak (parabolic interpolation) of 2 s
   frames every 0.25 s; a step in the median frequency before / after an
   instant beyond ``max(FREQ_STEP_MIN_HZ, FREQ_Z * sigma)`` is also flagged.

Limitations: a cut of an exact whole number of hum cycles leaves no phase
step; hum that is too weak, absent or added digitally after editing defeats the
test; compressed codecs may remove hum below their high-pass.
"""
from __future__ import annotations

from math import gcd
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt, welch

from ..core.types import Category, Evidence, Segment
from ..core.utils import calibrate
from ._dsp import make_series, merge_close, runs

DETECTOR = "audio.enf"
TITLE = "ENF (mains hum) continuity"

FS = 1000                    # analysis sample rate (Hz)
MIN_SECONDS = 6.0
HUM_SNR_MIN_DB = 10.0        # narrow-band peak over local background
BW_HZ = 1.0                  # band-pass width at the fundamental (x k for harmonic k)
PHASE_RATE = 100             # phase residual decimated to 100 Hz
FIT_S = 1.5                  # linear-fit length on each side
GUARD_S = 0.6                # gap between the two fits (filter transient)
PHASE_JUMP_MIN_RAD = 0.5     # absolute floor for a phase discontinuity
PHASE_Z = 6.0                # robust z for a phase discontinuity
FREQ_STEP_MIN_HZ = 0.015     # absolute floor for a frequency step (fundamental Hz)
FREQ_Z = 7.0
EVENT_MIN_GAP_S = 1.0
RESID_FACTOR = 4.0           # a fit window whose residual exceeds this x median
RESID_MIN_RAD = 0.15         # ...(and this absolute value) straddles a discontinuity
SIDELOBE_S = FIT_S + GUARD_S + 0.9   # a strong jump contaminates fits this far away
SIDELOBE_RATIO = 0.6         # ...so weaker events closer than that are suppressed
EDGE_S = 2.5                 # ignore file edges (filter transients)
DROPOUT_DB = 10.0            # hum level drop (vs file median) that counts as a dropout
DROPOUT_MIN_S = 0.8          # minimum dropout duration
DROPOUT_SOFTNESS_DB = 3.0


def _to_fs(x: np.ndarray, sr: int) -> np.ndarray:
    g = gcd(int(sr), FS)
    up, down = FS // g, int(sr) // g
    y = x.astype(np.float32)
    # Cheap integer pre-decimation for very high rates keeps the polyphase filter short.
    if up == 1 and down > 64 and down % 8 == 0:
        y = resample_poly(y, 1, 8)
        down //= 8
    return resample_poly(y, up, down).astype(np.float64)


def hum_snr(y: np.ndarray, fs: int = FS) -> List[Dict[str, float]]:
    """Narrow-band SNR (dB) of mains lines; sorted best first."""
    nper = int(min(len(y), 16 * fs))
    f, p = welch(y, fs, nperseg=nper, noverlap=nper // 2, detrend="constant")
    res = f[1] - f[0]
    out = []
    for nominal in (50.0, 60.0):
        for k in (1, 2):
            fc = nominal * k
            pk = np.abs(f - fc) <= 0.15 * k + res
            bg = (np.abs(f - fc) >= 1.0 * k) & (np.abs(f - fc) <= 4.0 * k)
            if not pk.any() or bg.sum() < 4:
                continue
            i = np.flatnonzero(pk)[np.argmax(p[pk])]
            # parabolic interpolation of the peak frequency
            if 0 < i < len(p) - 1:
                a, b, c = np.log(p[i - 1:i + 2] + 1e-30)
                den = a - 2 * b + c
                off = 0.5 * (a - c) / den if den != 0 else 0.0
            else:
                off = 0.0
            snr = 10 * np.log10(p[i] / (np.median(p[bg]) + 1e-30))
            out.append({"nominal": nominal, "harmonic": k, "freq": float(f[i] + off * res),
                        "snr_db": float(snr)})
    return sorted(out, key=lambda d: -d["snr_db"])


def _line_fits(p: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every window start s: LS line through p[s:s+w] -> (value at u=0, slope, rms resid)."""
    u = np.arange(w, dtype=np.float64)
    su, suu = u.sum(), (u * u).sum()
    ones = np.ones(w)
    sy = np.convolve(p, ones, "valid")
    suy = np.convolve(p, u[::-1], "valid")
    syy = np.convolve(p * p, ones, "valid")
    slope = (w * suy - su * sy) / (w * suu - su * su)
    icpt = (sy - slope * su) / w
    rss = syy - icpt * sy - slope * suy
    return icpt, slope, np.sqrt(np.maximum(rss, 0.0) / w)


def phase_discontinuity(ph: np.ndarray, rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapped jump (rad) at each sample, and the worse of the two fit residuals.

    nan where the windows do not fit. A window that straddles a discontinuity
    has a large residual, which is how side lobes of a real jump are rejected.
    """
    w, g = int(FIT_S * rate), int(GUARD_S * rate)
    # remove the global linear trend first (keeps the sums well conditioned)
    tt = np.arange(len(ph), dtype=np.float64)
    ph = ph - np.polyval(np.polyfit(tt, ph, 1), tt) if len(ph) > 2 else ph
    icpt, slope, rms = _line_fits(ph, w)
    n = len(ph)
    jump = np.full(n, np.nan)
    resid = np.full(n, np.nan)
    i = np.arange(w + g, n - w - g)
    if i.size == 0:
        return jump, resid
    sl, sr_ = i - g - w, i + g
    left = icpt[sl] + slope[sl] * (w + g)         # left fit extrapolated forward to i
    right = icpt[sr_] + slope[sr_] * (-g)         # right fit extrapolated back to i
    d = right - left
    jump[i] = (d + np.pi) % (2 * np.pi) - np.pi
    resid[i] = np.maximum(rms[sl], rms[sr_])
    return jump, resid


def freq_track(yb: np.ndarray, fs: int, fh: float, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """ENF frequency track (scaled to the fundamental) by zoom-DFT + parabolic peak."""
    from numpy.lib.stride_tricks import sliding_window_view

    nper, hop = 2 * fs, fs // 4
    if len(yb) < nper + hop:
        return np.zeros(0), np.zeros(0)
    frames = sliding_window_view(yb, nper)[::hop]
    win = np.hanning(nper)
    df = 0.02 * k
    fz = np.arange(fh - 0.6 * k, fh + 0.6 * k + df / 2, df)
    n = np.arange(nper)
    basis = np.exp(-2j * np.pi * np.outer(n, fz) / fs) * win[:, None]
    mag = np.abs(frames @ basis) ** 2
    i = np.clip(np.argmax(mag, axis=1), 1, len(fz) - 2)
    r = np.arange(len(i))
    a, b, c = (np.log(mag[r, i + o] + 1e-30) for o in (-1, 0, 1))
    den = a - 2 * b + c
    off = np.where(den != 0, 0.5 * (a - c) / np.where(den != 0, den, 1), 0.0)
    f = (fz[i] + np.clip(off, -1, 1) * df) / k
    t = (np.arange(len(i)) * hop + nper / 2) / fs
    return t, f


def _step_stat(t: np.ndarray, f: np.ndarray, gap: float = 1.25, span: float = 2.0) -> np.ndarray:
    rate = 1.0 / (t[1] - t[0]) if len(t) > 1 else 4.0
    g, s = int(round(gap * rate)), int(round(span * rate))
    out = np.full(len(f), np.nan)
    if len(f) < 2 * (g + s) + 1:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    med = np.median(sliding_window_view(f, s), axis=1)  # med[j] = median f[j:j+s]
    i = np.arange(g + s, len(f) - g - s)
    out[i] = med[i + g] - med[i - g - s]
    return out


def detect_enf(x: np.ndarray, sr: int) -> Evidence:
    dur = len(x) / sr
    if dur < MIN_SECONDS:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.MANIPULATION,
                                       f"too short ({dur:.1f} s < {MIN_SECONDS:.0f} s)")
    if sr < 2 * 130:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.MANIPULATION,
                                       "sample rate too low")
    y = _to_fs(x, sr)
    lines = hum_snr(y)
    best = lines[0] if lines else None
    if best is None or best["snr_db"] < HUM_SNR_MIN_DB:
        return Evidence.not_applicable(
            DETECTOR, TITLE, Category.MANIPULATION,
            f"no mains hum detected (best narrow-band SNR "
            f"{best['snr_db'] if best else float('nan'):.1f} dB < {HUM_SNR_MIN_DB:.0f} dB)",
            hum_lines=lines)
    # Prefer the fundamental if it is nearly as clean as the harmonic.
    fund = [d for d in lines if d["nominal"] == best["nominal"] and d["harmonic"] == 1]
    if best["harmonic"] == 2 and fund and fund[0]["snr_db"] >= best["snr_db"] - 3:
        best = fund[0]
    k, fh, nominal = int(best["harmonic"]), float(best["freq"]), best["nominal"]

    bw = BW_HZ * k
    sos = butter(2, [fh - bw / 2, fh + bw / 2], btype="band", fs=FS, output="sos")
    yb = sosfiltfilt(sos, y)
    t = np.arange(len(yb)) / FS
    an = hilbert(yb) * np.exp(-2j * np.pi * fh * t)
    step = FS // PHASE_RATE
    an = an[::step]
    td = t[::step]
    ph = np.unwrap(np.angle(an))
    amp = np.abs(an)

    # --- hum dropouts --------------------------------------------------------
    from scipy.ndimage import uniform_filter1d

    amp_db = 20 * np.log10(uniform_filter1d(amp, size=int(0.3 * PHASE_RATE), mode="nearest") + 1e-12)
    inner = (td >= 1.0) & (td <= dur - 1.0)
    ref_db = float(np.median(amp_db[inner])) if inner.any() else float(np.median(amp_db))
    low = (amp_db < ref_db - DROPOUT_DB) & inner
    st, en = runs(low)
    keep = (en - st) >= int(DROPOUT_MIN_S * PHASE_RATE)
    st, en = st[keep], en[keep]
    # Localise edges at the -6 dB crossing: the zero-phase band-pass smears an abrupt
    # stop symmetrically, so the half-amplitude point is the best edge estimate.
    half = amp_db < ref_db - 6.0
    for r in range(len(st)):
        a_, b_ = st[r], en[r]
        while a_ > 0 and half[a_ - 1]:
            a_ -= 1
        while b_ < len(half) and half[b_]:
            b_ += 1
        st[r], en[r] = a_, b_
    drop_depth = [float(ref_db - np.median(amp_db[a_:b_])) for a_, b_ in zip(st, en)]
    raw_d = max(drop_depth) if drop_depth else 0.0
    # mask phase / frequency statistics where any fit window touches a dropout
    reach = FIT_S + GUARD_S + 0.5
    bad = np.zeros(len(td), bool)
    for a_, b_ in zip(st, en):
        bad |= (td >= td[a_] - reach) & (td <= td[min(b_, len(td) - 1)] + reach)

    jump, fit_rms = phase_discontinuity(ph, PHASE_RATE)
    valid = (td >= EDGE_S) & (td <= dur - EDGE_S) & np.isfinite(jump)
    if valid.sum() < PHASE_RATE:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.MANIPULATION,
                                       "recording too short after edge trimming")
    sigma = 1.4826 * np.median(np.abs(jump[valid] - np.median(jump[valid]))) + 1e-6
    rms_ref = float(np.median(fit_rms[valid]))
    straddle = fit_rms > max(RESID_FACTOR * rms_ref, RESID_MIN_RAD)
    valid &= ~bad & ~straddle
    aj = np.abs(np.where(valid, jump, 0.0))
    thr_p = max(PHASE_JUMP_MIN_RAD, PHASE_Z * sigma)
    raw_p = float(aj.max())

    tf, f = freq_track(yb, FS, fh, k)
    fstep = _step_stat(tf, f) if len(f) else np.zeros(0)
    if len(f):
        fbad = np.interp(tf, td, bad.astype(float)) > 0
        fvalid = np.isfinite(fstep) & (tf >= EDGE_S) & (tf <= dur - EDGE_S)
    else:
        fbad = fvalid = np.zeros(0, bool)
    if fvalid.sum() > 8:
        fsig = 1.4826 * np.median(np.abs(fstep[fvalid] - np.median(fstep[fvalid]))) + 1e-6
        fvalid &= ~fbad
        fs_abs = np.abs(np.where(fvalid, fstep, 0.0))
        thr_f = max(FREQ_STEP_MIN_HZ, FREQ_Z * fsig)
        raw_f = float(fs_abs.max())
    else:
        fs_abs, thr_f, raw_f, fsig = np.zeros(len(f)), FREQ_STEP_MIN_HZ, 0.0, float("nan")

    score_p = calibrate(raw_p, thr_p, 0.25 * thr_p)
    score_f = calibrate(raw_f, thr_f, 0.25 * thr_f)
    score_d = calibrate(raw_d, DROPOUT_DB, DROPOUT_SOFTNESS_DB) if drop_depth else 0.0
    score = max(score_p, score_f, score_d)

    # --- events -----------------------------------------------------------------
    events: List[Tuple[float, float, str]] = []
    for a_, b_, dd in zip(st, en, drop_depth):
        sc = calibrate(dd, DROPOUT_DB, DROPOUT_SOFTNESS_DB)
        events.append((float(td[a_]), sc, f"hum drops out ({dd:.0f} dB)"))
        events.append((float(td[min(b_, len(td) - 1)]), sc, f"hum returns ({dd:.0f} dB)"))
    cand = np.flatnonzero(aj >= thr_p)
    if cand.size:
        kept = merge_close(td[cand], aj[cand], EVENT_MIN_GAP_S)
        kept = [cand[i] for i in kept]
        # side-lobe suppression: a strong jump biases the line fits up to SIDELOBE_S away
        kept = [j for j in kept if not any(
            abs(td[j] - td[m]) < SIDELOBE_S and aj[j] < SIDELOBE_RATIO * aj[m] for m in kept)]
        for j in kept:
            events.append((float(td[j]), calibrate(aj[j], thr_p, 0.25 * thr_p),
                           f"ENF phase jump {aj[j]:.2f} rad"))
    if fs_abs.size:
        fc = np.flatnonzero(fs_abs >= thr_f)
        if fc.size:
            for i in merge_close(tf[fc], fs_abs[fc], 2 * SIDELOBE_S):
                j = fc[i]
                if all(abs(tf[j] - e[0]) > SIDELOBE_S for e in events):
                    events.append((float(tf[j]), calibrate(fs_abs[j], thr_f, 0.25 * thr_f),
                                   f"ENF frequency step {fs_abs[j] * 1000:.0f} mHz"))
    events.sort()
    segments = [Segment(max(0.0, te - 0.5), min(dur, te + 0.5), sc, lab) for te, sc, lab in events]

    snr = best["snr_db"]
    confidence = float(np.clip(0.3 + (snr - HUM_SNR_MIN_DB) / 25.0, 0.3, 0.9))
    markers = [e[0] for e in events]
    # Phase residual for display: remove a slow (10 s) trend so drift does not dominate.
    resid = ph - uniform_filter1d(ph, size=int(10 * PHASE_RATE), mode="nearest")
    series = {
        "enf_frequency": make_series(tf, f, f"ENF frequency (Hz, from {fh:.2f} Hz line)", markers),
        "phase_residual": make_series(td, resid, "hum phase residual (rad)", markers),
        "phase_discontinuity": make_series(td[valid], jump[valid], "phase jump statistic (rad)",
                                           markers),
        "hum_level": make_series(td, amp_db - ref_db, "hum level vs median (dB)", markers),
    }
    if events:
        summary = (f"Mains hum at {nominal:.0f} Hz (harmonic {k}, SNR {snr:.1f} dB) shows "
                   f"{len(events)} discontinuit{'y' if len(events) == 1 else 'ies'}: largest phase "
                   f"jump {raw_p:.2f} rad (threshold {thr_p:.2f}), largest frequency step "
                   f"{raw_f * 1000:.0f} mHz (threshold {thr_f * 1000:.0f}), "
                   f"{len(drop_depth)} hum dropout(s) (deepest {raw_d:.0f} dB, threshold "
                   f"{DROPOUT_DB:.0f}). The hum of a "
                   f"continuous recording cannot jump; this indicates a cut or splice at "
                   + ", ".join(f"{e[0]:.2f} s" for e in events[:5]) + ".")
    else:
        summary = (f"Mains hum at {nominal:.0f} Hz (harmonic {k}, SNR {snr:.1f} dB) is "
                   f"continuous: largest phase jump {raw_p:.2f} rad (threshold {thr_p:.2f}), "
                   f"largest frequency step {raw_f * 1000:.0f} mHz (threshold "
                   f"{thr_f * 1000:.0f}). Note: a cut of a whole number of hum cycles would "
                   f"not show.")
    details = {
        "raw_phase_jump_rad": raw_p, "phase_threshold_rad": thr_p, "phase_sigma_rad": sigma,
        "raw_freq_step_hz": raw_f, "freq_threshold_hz": thr_f, "freq_sigma_hz": fsig,
        "hum_nominal_hz": nominal, "hum_harmonic": k, "hum_freq_hz": fh, "hum_snr_db": snr,
        "hum_lines": lines, "enf_mean_hz": float(np.mean(f)) if len(f) else None,
        "enf_std_hz": float(np.std(f)) if len(f) else None,
        "hum_amplitude_cv": float(np.std(amp) / (np.mean(amp) + 1e-12)),
        "raw_dropout_db": raw_d, "dropout_threshold_db": DROPOUT_DB,
        "dropouts": [{"start": float(td[a_]), "end": float(td[min(b_, len(td) - 1)]),
                      "depth_db": dd} for a_, b_, dd in zip(st, en, drop_depth)],
        "events": [{"time": e[0], "score": e[1], "label": e[2]} for e in events],
    }
    return Evidence(detector=DETECTOR, title=TITLE, category=Category.MANIPULATION, score=score,
                    confidence=confidence, summary=summary, details=details,
                    segments=segments, series=series)
