"""audio.bandwidth / audio.bandwidth_shift -- effective bandwidth and its consistency.

A file's sample rate promises a bandwidth (Nyquist = sr / 2). Content that was
produced at a lower rate and upsampled -- typical of neural TTS / voice-cloning
pipelines (16, 22.05 or 24 kHz models), but also of telephone and VoIP audio
and of many transcodes -- has nothing above the original Nyquist: the spectrum
ends in a brick wall far below the file's Nyquist.

audio.bandwidth (AI_GENERATION, moderate confidence)
    Long-term spectrum (Hann frames). The effective bandwidth is the highest
    frequency still within ``BW_DROP_DB`` of the spectral peak; a *brick-wall*
    edge is located with :func:`._dsp.lowpass_cutoff` (>= ``CUTOFF_DROP_DB``
    drop across a few hundred Hz, occupied band below). ``raw`` = missing
    fraction of the band, ``1 - cutoff / Nyquist`` (0 when there is no
    brick wall); ``score = calibrate(raw, MISSING_THRESHOLD, MISSING_SOFTNESS)``.
    Lossy codecs low-pass at 15-20 kHz on purpose, so for lossy files edges
    above ``LOSSY_LOWPASS_MIN_HZ`` are reported but not scored.

audio.bandwidth_shift (MANIPULATION)
    The same edge search on every ~1 s window (0.5 s hop). Windows whose edge
    differs from the file's typical edge by more than ``SHIFT_THRESHOLD_OCT``
    octaves, in runs of at least ``SHIFT_MIN_WINDOWS``, indicate material from
    a different source / chain spliced in. ``raw`` = largest such deviation
    (octaves); full-band windows count as Nyquist. For lossy files the
    threshold is ``LOSSY_SHIFT_THRESHOLD_OCT`` and confidence is low, because
    MP3 / AAC / Vorbis encoders drop high bands frame by frame. A window where no edge can
    be seen (e.g. a quiet passage) is assumed to share the file's dominant
    state unless it has real content in the file's stop band.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import Category, Evidence, Segment
from ..core.utils import calibrate
from ._dsp import block_spectra, lowpass_cutoff, make_series, runs, to_db

DETECTOR = "audio.bandwidth"
TITLE = "Effective bandwidth (upsampling)"
DETECTOR_SHIFT = "audio.bandwidth_shift"
TITLE_SHIFT = "Bandwidth consistency over time"

BLOCK_S = 0.5
MIN_SECONDS = 1.0
BW_DROP_DB = 60.0            # effective bandwidth: highest freq within this of the peak
CUTOFF_DROP_DB = 25.0        # brick-wall: drop across the edge
MISSING_THRESHOLD = 0.2      # cutoff below 80 % of Nyquist
MISSING_SOFTNESS = 0.06
LOSSY_LOWPASS_MIN_HZ = 15000.0
BANDWIDTH_CONFIDENCE = 0.4
SILENT_BLOCK_DB = -110.0     # blocks quieter than this (mean power) are skipped

SHIFT_THRESHOLD_OCT = 0.35
SHIFT_SOFTNESS_OCT = 0.1
SHIFT_MIN_WINDOWS = 2
SHIFT_MAX_FRACTION = 0.45    # deviating windows must be a minority
SHIFT_CONFIDENCE = 0.6
LOSSY_SHIFT_THRESHOLD_OCT = 0.8   # lossy codecs drop HF bands frame by frame
LOSSY_SHIFT_CONFIDENCE = 0.25
WIDER_DB = 20.0              # stop-band content needed to call a window "wider"


def _nfft(sr: int) -> int:
    return int(2 ** np.round(np.log2(sr / 46.875)))  # ~47 Hz bins at any rate (1024 @ 48k)


def spectra(x: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-0.5 s mean power and mean log-power spectra, shared by both detectors."""
    return block_spectra(x, sr, _nfft(sr), BLOCK_S)


def _effective_bw(freqs: np.ndarray, s_db: np.ndarray) -> float:
    from scipy.ndimage import median_filter

    k = max(3, int(round(200.0 / (freqs[1] - freqs[0]))) | 1)
    sm = median_filter(s_db, size=k, mode="nearest")
    lo = np.searchsorted(freqs, 50.0)
    peak = sm[lo:].max()
    above = np.flatnonzero(sm[lo:] >= peak - BW_DROP_DB)
    return float(freqs[lo + above[-1]]) if above.size else float("nan")


def detect_bandwidth(x: np.ndarray, sr: int, *, lossy: bool = False,
                     spec: Optional[Tuple[np.ndarray, ...]] = None) -> Evidence:
    dur = len(x) / sr
    if dur < MIN_SECONDS:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.AI_GENERATION, "too short")
    if sr < 16000:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.AI_GENERATION,
                                       f"sample rate {sr} Hz is already narrowband")
    freqs, P, cnt, _ = spec if spec is not None else spectra(x, sr)
    active = (to_db(P.mean(axis=1)) > SILENT_BLOCK_DB) & (cnt > 0)
    if active.sum() < 1:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.AI_GENERATION, "no signal")
    S = to_db(P[active].mean(axis=0))
    nyq = sr / 2
    eff = _effective_bw(freqs, S)
    cut = lowpass_cutoff(freqs, S, drop_db=CUTOFF_DROP_DB)
    fc = float(cut["cutoff_hz"][0])
    has_edge = np.isfinite(fc)
    raw = (1.0 - fc / nyq) if has_edge else 0.0
    codec_lowpass = bool(lossy and has_edge and fc >= LOSSY_LOWPASS_MIN_HZ)
    scored_raw = 0.0 if codec_lowpass else raw
    score = calibrate(scored_raw, MISSING_THRESHOLD, MISSING_SOFTNESS)

    if has_edge and not codec_lowpass:
        typical = {8000: "8 kHz (telephone / 16 kHz models)", 11025: "11.025 kHz",
                   12000: "12 kHz (24 kHz TTS / vocoders)", 16000: "16 kHz",
                   22050: "22.05 kHz"}
        near = min(typical, key=lambda f: abs(f - fc))
        hint = f", close to {typical[near]}" if abs(near - fc) / near < 0.08 else ""
        summary = (f"Spectrum ends in a brick wall at {fc / 1000:.1f} kHz{hint} "
                   f"(drop {cut['drop_db'][0]:.0f} dB) although the file could hold "
                   f"{nyq / 1000:.1f} kHz: {raw:.0%} of the band is empty (threshold "
                   f"{MISSING_THRESHOLD:.0%}). The audio was produced at a lower sample rate and "
                   f"upsampled -- common in TTS / voice-cloning output, but equally in phone, "
                   f"VoIP or transcoded recordings, so this is a weak cue on its own.")
    elif codec_lowpass:
        summary = (f"Spectrum is cut at {fc / 1000:.1f} kHz, consistent with the low-pass of "
                   f"the lossy codec itself; not scored.")
    else:
        summary = (f"No brick-wall low-pass: content extends to {eff / 1000:.1f} kHz (within "
                   f"{BW_DROP_DB:.0f} dB of the peak) of the {nyq / 1000:.1f} kHz available; "
                   f"missing band fraction 0 (threshold {MISSING_THRESHOLD:.0%}).")
    vis = make_series(freqs / 1000.0, S - S.max(), "long-term spectrum (dB re peak)",
                      [fc / 1000.0] if has_edge else [], xlabel="frequency (kHz)")
    details = {"raw_missing_fraction": raw, "threshold": MISSING_THRESHOLD,
               "cutoff_hz": fc if has_edge else None, "edge_drop_db": float(cut["drop_db"][0]),
               "effective_bandwidth_hz": eff, "nyquist_hz": nyq, "lossy": lossy,
               "codec_lowpass": codec_lowpass}
    conf = BANDWIDTH_CONFIDENCE if has_edge and not codec_lowpass else 0.25
    return Evidence(detector=DETECTOR, title=TITLE, category=Category.AI_GENERATION, score=score,
                    confidence=conf, summary=summary, details=details,
                    series={"spectrum": vis})


def detect_bandwidth_shift(x: np.ndarray, sr: int, *, lossy: bool = False,
                           spec: Optional[Tuple[np.ndarray, ...]] = None
                           ) -> Evidence:
    dur = len(x) / sr
    if dur < 4.0:
        return Evidence.not_applicable(DETECTOR_SHIFT, TITLE_SHIFT, Category.MANIPULATION,
                                       "too short")
    if sr < 16000:
        return Evidence.not_applicable(DETECTOR_SHIFT, TITLE_SHIFT, Category.MANIPULATION,
                                       f"sample rate {sr} Hz is already narrowband")
    freqs, P, cnt, L = spec if spec is not None else spectra(x, sr)
    nyq = sr / 2
    # ~1 s windows, 0.5 s hop = pairs of adjacent 0.5 s blocks. Per-window spectra
    # are geometric means (mean dB), so one splice click cannot fill a stop band.
    den = np.maximum(cnt[:-1] + cnt[1:], 1)[:, None]
    W = (P[:-1] * cnt[:-1, None] + P[1:] * cnt[1:, None]) / den
    wdb = (L[:-1] * cnt[:-1, None] + L[1:] * cnt[1:, None]) / den
    wt = (np.arange(len(W)) + 1) * BLOCK_S          # window centres
    active = to_db(W.mean(axis=1)) > SILENT_BLOCK_DB
    if active.sum() < 6:
        return Evidence.not_applicable(DETECTOR_SHIFT, TITLE_SHIFT, Category.MANIPULATION,
                                       "not enough non-silent audio")
    # The file's dominant state: its long-term edge (or full band).
    Sg = to_db(P[(to_db(P.mean(axis=1)) > SILENT_BLOCK_DB) & (cnt > 0)].mean(axis=0))
    g_fc = float(lowpass_cutoff(freqs, Sg, drop_db=CUTOFF_DROP_DB)["cutoff_hz"][0])
    cut = lowpass_cutoff(freqs, wdb, drop_db=CUTOFF_DROP_DB, occ_db=to_db(W))
    fc = np.where(np.isfinite(cut["cutoff_hz"]), cut["cutoff_hz"], nyq)
    if np.isfinite(g_fc):
        # Band-limited file: a window without its own edge is consistent unless it
        # carries real content in the file's stop band (louder than the file's
        # typical stop-band level by WIDER_DB) -- then it is wider than the rest.
        a_, b_ = np.searchsorted(freqs, [g_fc * 1.2, 0.97 * nyq])
        if b_ - a_ >= 3:
            stop = np.median(wdb[:, a_:b_], axis=1)
            typical = float(np.median(stop[active]))
            wider = ~np.isfinite(cut["cutoff_hz"]) & (stop >= typical + WIDER_DB)
        else:
            wider = np.zeros(len(fc), bool)
        fc = np.where(np.isfinite(cut["cutoff_hz"]), cut["cutoff_hz"],
                      np.where(wider, nyq, g_fc))
    if lossy:  # codec low-pass region: do not compare edges inside it
        fc = np.minimum(fc, LOSSY_LOWPASS_MIN_HZ)
    ref = float(np.median(fc[active]))
    dev = np.abs(np.log2(fc / ref))
    dev[~active] = 0.0
    thr = LOSSY_SHIFT_THRESHOLD_OCT if lossy else SHIFT_THRESHOLD_OCT
    bad = dev >= thr
    st, en = runs(bad)
    keep = (en - st) >= SHIFT_MIN_WINDOWS
    st, en = st[keep], en[keep]
    frac_bad = float(bad[active].mean())
    raw = max((float(np.median(dev[a:b])) for a, b in zip(st, en)), default=0.0)
    if frac_bad > SHIFT_MAX_FRACTION:   # no clear majority: inconsistent throughout
        raw_eff = 0.0
    else:
        raw_eff = raw
    score = calibrate(raw_eff, thr, SHIFT_SOFTNESS_OCT)
    segments: List[Segment] = []
    if raw_eff > 0:
        for a, b in zip(st, en):
            d = float(np.median(dev[a:b]))
            fcr = float(np.median(fc[a:b]))
            # windows are 1 s long: the run spans from the first window's start to the last's end
            segments.append(Segment(max(0.0, float(wt[a]) - 0.5), min(dur, float(wt[b - 1]) + 0.5),
                                    calibrate(d, thr, SHIFT_SOFTNESS_OCT),
                                    f"bandwidth {fcr / 1000:.1f} kHz vs {ref / 1000:.1f} kHz "
                                    f"elsewhere"))
    if segments:
        summary = (f"Bandwidth is not constant: {len(segments)} passage(s) with a different "
                   f"low-pass edge (largest deviation {raw:.2f} octaves, threshold "
                   f"{thr:.2f}); typical edge {ref / 1000:.1f} kHz. Material "
                   f"from a different recording / processing chain appears to be spliced in at "
                   + ", ".join(f"{s.start:.1f}-{s.end:.1f} s" for s in segments[:4]) + ".")
    elif frac_bad > SHIFT_MAX_FRACTION:
        summary = (f"Bandwidth varies throughout the file ({frac_bad:.0%} of windows deviate), "
                   f"so no single passage stands out; not scored.")
    else:
        summary = (f"Bandwidth is consistent over time (typical edge {ref / 1000:.1f} kHz, "
                   f"largest deviation {raw:.2f} octaves, threshold {thr:.2f})."
                   + (" Lossy codecs vary their high-frequency bands frame by frame, so this "
                      "check is less reliable here." if lossy else ""))
    conf = LOSSY_SHIFT_CONFIDENCE if lossy else SHIFT_CONFIDENCE
    details: Dict[str, object] = {"raw_deviation_oct": raw, "threshold_oct": thr,
                                  "reference_edge_hz": ref, "deviating_fraction": frac_bad,
                                  "lossy": lossy}
    series = {"window_bandwidth": make_series(wt, np.where(active, fc, np.nan) / 1000.0,
                                              "low-pass edge per 1 s window (kHz)",
                                              [s.start for s in segments])}
    return Evidence(detector=DETECTOR_SHIFT, title=TITLE_SHIFT, category=Category.MANIPULATION,
                    score=score, confidence=conf, summary=summary, details=details,
                    segments=segments, series=series)
