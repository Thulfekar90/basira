"""Small, vectorised DSP helpers shared by the audio detectors.

Everything here works on float32 mono arrays and processes long inputs in
chunks so a 10-minute 48 kHz file never materialises a full STFT in memory.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SILENCE_DB = -200.0          # level assigned to frames that are exactly zero
_CHUNK = 1 << 21             # samples per chunk for energy computations


def to_mono(samples: np.ndarray) -> np.ndarray:
    """(n,) or (n, ch) -> float32 (n,), NaN/inf replaced by 0."""
    a = np.asarray(samples, dtype=np.float32)
    if a.size == 0:
        return np.empty(0, dtype=np.float32)
    if a.ndim == 2:
        if 0 < a.shape[0] < a.shape[1] and a.shape[0] <= 8:  # (ch, n) given by mistake
            a = a.T
        if a.shape[1] == 1:
            a = a[:, 0]
        else:
            mix = a.mean(axis=1, dtype=np.float32)
            # Out-of-phase channels cancel in a mono mix ([x, -x] -> silence), possibly
            # only in part of the file. Compare the mix with the loudest channel block by
            # block; if any audible stretch is cancelled, analyse one channel for the whole
            # file (switching mid-file would itself create a discontinuity).
            ch_rms = np.sqrt(np.mean(np.square(a, dtype=np.float64), axis=0))
            lead = a[:, int(np.argmax(ch_rms))]
            blk = 2048
            n = len(mix)
            if n:
                nb = -(-n // blk)                              # ceil: last block may be partial
                pad = nb * blk - n
                sq_mix = np.pad(np.square(mix.astype(np.float64)), (0, pad)).reshape(nb, blk)
                sq_ch = np.pad(np.square(lead.astype(np.float64)), (0, pad)).reshape(nb, blk)
                count = np.full(nb, blk, dtype=np.float64)
                count[-1] = blk - pad
                e_mix, e_ch = sq_mix.sum(axis=1) / count, sq_ch.sum(axis=1) / count
                audible = e_ch > max(1e-9, 1e-4 * float(e_ch.max()))
                cancelled = audible & (e_mix < 0.09 * e_ch)
                a = lead if cancelled.any() else mix   # one channel is always a valid choice
            else:
                a = mix
    a = np.nan_to_num(a.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(a, dtype=np.float32)


def hop_energies(x: np.ndarray, hop: int) -> np.ndarray:
    """Sum of squares over consecutive non-overlapping hops (float64)."""
    m = len(x) // hop
    out = np.empty(m, dtype=np.float64)
    step = max(1, _CHUNK // hop)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        seg = x[i0 * hop:i1 * hop].astype(np.float64)
        out[i0:i1] = np.einsum("ij,ij->i", seg.reshape(-1, hop), seg.reshape(-1, hop))
    return out


def frame_levels(x: np.ndarray, sr: int, hop_s: float = 0.01, frame_hops: int = 2
                 ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Frame RMS level in dBFS (frame = ``frame_hops`` hops) -> (t_center, db, hop)."""
    hop = max(1, int(round(sr * hop_s)))
    e = hop_energies(x, hop)
    if len(e) < frame_hops:
        return np.zeros(0), np.zeros(0), hop
    fe = np.convolve(e, np.ones(frame_hops), "valid") / (hop * frame_hops)
    with np.errstate(divide="ignore"):
        db = np.maximum(10 * np.log10(fe + 1e-30), SILENCE_DB)
    db[fe <= 0] = SILENCE_DB
    t = (np.arange(len(fe)) * hop + hop * frame_hops / 2) / sr
    return t, db, hop


def power_frames(x: np.ndarray, nfft: int, hop: int, chunk_frames: int = 1024
                 ) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (first_frame_index, |rfft|^2 of Hann-windowed frames) in chunks (float32).

    Power is normalised so a full-scale sine reads ~0 dB at its peak bin.
    """
    import scipy.fft as sfft

    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    view = sliding_window_view(x, nfft)[::hop]
    win = np.hanning(nfft).astype(np.float32)
    norm = (2.0 / win.sum()) ** 2
    for i0 in range(0, len(view), chunk_frames):
        fr = view[i0:i0 + chunk_frames] * win
        spec = sfft.rfft(fr, axis=1, workers=-1)
        p = (spec.real * spec.real + spec.imag * spec.imag) * norm
        yield i0, p.astype(np.float32, copy=False)


def block_spectra(x: np.ndarray, sr: int, nfft: int, block_s: float
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per time block: mean power spectrum and mean *log* power spectrum.

    Returns (freqs, P[n_blocks, n_bins], counts, L[n_blocks, n_bins]) where L is
    the mean of 10*log10(power) over frames that are not digital silence (a
    geometric mean, robust to a single click).
    """
    hop = nfft
    blk = max(1, int(round(block_s * sr)))
    n_blocks = max(1, int(np.ceil(len(x) / blk)))
    nb = nfft // 2 + 1
    acc = np.zeros((n_blocks, nb), dtype=np.float64)
    lacc = np.zeros((n_blocks, nb), dtype=np.float64)
    cnt = np.zeros(n_blocks, dtype=np.int64)
    lcnt = np.zeros(n_blocks, dtype=np.int64)
    for i0, p in power_frames(x, nfft, hop):
        centers = (np.arange(i0, i0 + len(p)) * hop + nfft // 2)
        b = np.minimum(centers // blk, n_blocks - 1)
        np.add.at(cnt, b, 1)
        # frames of a chunk are sorted by block -> reduce contiguous groups
        ub, first = np.unique(b, return_index=True)
        sums = np.add.reduceat(p.astype(np.float64), first, axis=0)
        acc[ub] += sums
        live = p.sum(axis=1) > 0
        lp = np.where(live[:, None], 10 * np.log10(p.astype(np.float64) + 1e-20), 0.0)
        lacc[ub] += np.add.reduceat(lp, first, axis=0)
        np.add.at(lcnt, b[live], 1)
    P = acc / np.maximum(cnt, 1)[:, None]
    L = lacc / np.maximum(lcnt, 1)[:, None]
    L[lcnt == 0] = -200.0
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    return freqs, P, cnt, L


def to_db(p: np.ndarray, floor: float = 1e-20) -> np.ndarray:
    return 10 * np.log10(np.asarray(p, dtype=np.float64) + floor)


def runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) indices of True runs."""
    m = np.asarray(mask, dtype=np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def downsample_xy(x: Sequence[float], y: Sequence[float], max_points: int = 1500
                  ) -> Tuple[List[float], List[float]]:
    """Reduce a curve to <= max_points, keeping the most deviant point of each bucket."""
    x = np.asarray(x, dtype=np.float64)
    y = np.nan_to_num(np.asarray(y, dtype=np.float64))
    if len(x) <= max_points:
        return [round(float(v), 4) for v in x], [round(float(v), 5) for v in y]
    edges = np.linspace(0, len(x), max_points + 1).astype(int)
    med = np.median(y)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            idx.append(a + int(np.argmax(np.abs(y[a:b] - med))))
    idx = np.asarray(idx)
    return [round(float(v), 4) for v in x[idx]], [round(float(v), 5) for v in y[idx]]


def make_series(x, y, ylabel: str, markers: Optional[Sequence[float]] = None,
                xlabel: str = "time (s)") -> Dict[str, object]:
    xs, ys = downsample_xy(x, y)
    return {"x": xs, "y": ys, "xlabel": xlabel, "ylabel": ylabel,
            "markers": [round(float(m), 3) for m in (markers or [])]}


def merge_close(times: np.ndarray, values: np.ndarray, min_gap: float) -> List[int]:
    """Non-maximum suppression: indices of the strongest events at least min_gap apart."""
    order = np.argsort(-np.asarray(values))
    kept: List[int] = []
    for i in order:
        if all(abs(times[i] - times[j]) >= min_gap for j in kept):
            kept.append(int(i))
    return sorted(kept, key=lambda i: times[i])


OCC_LOGMEAN_SLACK_DB = 6.0   # a log-mean level sits a few dB below the power mean
TRANSITIONS_HZ = (200.0, 400.0, 800.0, 1600.0, 2400.0, 3200.0, 4800.0)


def lowpass_cutoff(freqs: np.ndarray, s_db: np.ndarray, *, min_hz: float = 2000.0,
                   drop_db: float = 25.0, occupancy_min: float = 0.6,
                   occupancy_rel_db: float = 50.0, max_transition_frac: float = 0.35,
                   below_hz: float = 1200.0, occ_db: Optional[np.ndarray] = None
                   ) -> Dict[str, np.ndarray]:
    """Locate a brick-wall low-pass edge in dB spectra (vectorised over rows).

    ``s_db`` is (n_bins,) or (n_rows, n_bins). For each candidate edge ``f``:

    * ``below(f)`` = median level over ``[f - below_hz, f]``;
    * ``above(f, T)`` = loudest (300 Hz-median-smoothed) level anywhere in
      ``[f + T, 0.995 * Nyquist]`` -- a real low-pass leaves *nothing* above its
      stop band, so ``drop = below - above`` is large;
    * the transition width ``T`` may be at most ``max_transition_frac * f``
      (0.35 -> at least ~65 dB/octave for a 25 dB drop). Natural roll-offs
      (microphones, air absorption, speech tilt) are far gentler;
    * isolated tones / sweeps (and the skirts of filters) are rejected by
      requiring the band from 300 Hz to the edge to be broadly occupied:
      ``occupancy`` = fraction of *unsmoothed* bins within ``occupancy_rel_db``
      of the band's strong bins (90th percentile), and the smoothed level just
      below the edge must itself be within ``occupancy_rel_db`` of them. (A
      tone's leakage skirt survives median smoothing and would otherwise look
      like broadband content.) ``occ_db`` optionally supplies a different
      spectrum for this test (e.g. power-mean when ``s_db`` is a log-mean).

    Returns arrays (one value per row): ``cutoff_hz`` (nan = no edge found),
    ``drop_db`` (largest drop seen) and ``occupancy``.
    """
    from scipy.ndimage import median_filter

    S = np.atleast_2d(np.asarray(s_db, dtype=np.float64))
    O = S if occ_db is None else np.atleast_2d(np.asarray(occ_db, dtype=np.float64))
    n_rows, nb = S.shape
    out = {"cutoff_hz": np.full(n_rows, np.nan), "drop_db": np.zeros(n_rows),
           "occupancy": np.zeros(n_rows)}
    nyq = float(freqs[-1])
    df = float(freqs[1] - freqs[0])
    top = int(np.searchsorted(freqs, 0.995 * nyq))
    kb = max(3, int(round(below_hz / df)) | 1)
    ks = max(3, int(round(300.0 / df)) | 1)
    i_min = max(int(np.searchsorted(freqs, min_hz)), kb)
    g0 = max(1, int(round(TRANSITIONS_HZ[0] / df)))
    if top - i_min < 4 * g0:
        return out
    below_med = median_filter(S, size=(1, kb), mode="nearest")
    smooth = median_filter(S, size=(1, ks), mode="nearest")
    smooth[:, top:] = -np.inf
    rev_max = np.maximum.accumulate(smooth[:, ::-1], axis=1)[:, ::-1]
    rev_max = np.concatenate([rev_max, np.full((n_rows, 1), -np.inf)], axis=1)
    i = np.arange(i_min, top - g0)
    below = below_med[:, i - kb // 2]
    drop = np.full((n_rows, len(i)), -np.inf)
    trans = np.zeros((n_rows, len(i)))
    for T in TRANSITIONS_HZ:
        g = int(round(T / df))
        allowed = (freqs[i] * max_transition_frac >= T) & (i + g < top)
        if not allowed.any():
            continue
        d = below - rev_max[:, np.minimum(i + g, top)]
        d[:, ~allowed] = -np.inf
        better = d > drop
        drop = np.where(better, d, drop)
        trans = np.where(better, T, trans)
    out["drop_db"] = np.max(np.where(np.isfinite(drop), drop, 0.0), axis=1)
    lo = int(np.searchsorted(freqs, 300.0))
    for r in range(n_rows):
        cand = np.flatnonzero(drop[r] >= drop_db)
        if cand.size == 0:
            continue
        # highest edge first (the true edge is the top of the occupied band)
        for c in cand[::-1][:64]:
            fi = i[c]
            if fi <= lo + 2:
                continue
            band = O[r, lo:fi]
            ref = float(np.percentile(band, 90))
            slack = 0.0 if occ_db is None else OCC_LOGMEAN_SLACK_DB
            if below[r, c] < ref - occupancy_rel_db - slack:
                continue
            occ = float(np.mean(band >= ref - occupancy_rel_db))
            if occ >= occupancy_min:
                out["cutoff_hz"][r] = float(freqs[fi])
                out["occupancy"][r] = occ
                break
    return out
