"""Deterministic synthetic audio + forgeries with known ground truth.

Nothing here is meant to sound like real speech; it reproduces the *statistical*
properties the audio detectors rely on:

* a "speech-like" source: glottal pulse train with a wandering f0, three formant
  resonators, syllable-rate amplitude modulation, fricative noise bursts and
  word pauses (so the spectrum extends naturally up to Nyquist),
* a room: stationary coloured noise (low-frequency rumble + white component),
* mains hum: a weak 50 Hz component (+ 2nd harmonic) whose frequency drifts
  slowly at random, like real electrical-network frequency (ENF).

All generators are seeded and return float32 mono arrays quantised to 16 bit.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.signal import butter, lfilter, resample_poly, sosfilt

SR = 48000


def quantize16(x: np.ndarray) -> np.ndarray:
    return (np.round(np.clip(x, -1, 1 - 1 / 32768) * 32768) / 32768).astype(np.float32)


def _ctrl(rng: np.random.Generator, duration: float, step: float, n: int, sr: int) -> np.ndarray:
    """Smooth random control curve (unit-variance values every ``step`` s, interpolated)."""
    ct = np.arange(0.0, duration + 2 * step, step)
    cv = rng.standard_normal(len(ct))
    return np.interp(np.arange(n) / sr, ct, cv)


def _resonator(x: np.ndarray, freq: float, bw: float, sr: int) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    th = 2 * np.pi * freq / sr
    return lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)


def speech_like(duration: float, sr: int = SR, seed: int = 0, level_rms: float = 0.08
                ) -> np.ndarray:
    """Speech-like source with exact-zero pauses (no noise)."""
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr

    # f0 contour 90..240 Hz -> glottal pulse train
    f0 = np.clip(150 * 2 ** (0.35 * _ctrl(rng, duration, 0.25, n, sr)), 90, 240)
    cyc = np.cumsum(f0) / sr
    pulses = np.zeros(n)
    pulses[1:][np.diff(np.floor(cyc)) > 0] = 1.0
    src = lfilter([1.0], [1, -0.96], lfilter([1.0], [1, -0.9], pulses))
    src -= lfilter([0.01], [1, -0.99], src)  # crude DC removal

    out = np.zeros(n)
    pos = rng.uniform(0.3, 0.6)
    ramp = int(0.02 * sr)
    while True:
        dur_w = rng.uniform(0.5, 1.8)
        if pos + dur_w > duration - 0.3:
            break
        i0, i1 = int(pos * sr), int((pos + dur_w) * sr)
        seg = src[i0:i1]
        f1, f2, f3 = rng.uniform(350, 800), rng.uniform(1000, 2000), rng.uniform(2300, 3200)
        v = (_resonator(seg, f1, 90, sr) + 0.6 * _resonator(seg, f2, 120, sr)
             + 0.3 * _resonator(seg, f3, 180, sr) + 0.05 * seg)
        rate = rng.uniform(3.5, 5.5)
        tt = t[i0:i1] - t[i0]
        env = 0.4 + 0.6 * (0.5 - 0.5 * np.cos(2 * np.pi * rate * tt + rng.uniform(0, 6.3))) ** 1.5
        v = v * env
        # fricative bursts (broadband, high-passed noise)
        fr = np.zeros(i1 - i0)
        for _ in range(rng.integers(1, 3)):
            L = int(rng.uniform(0.06, 0.15) * sr)
            if L >= len(fr):
                continue
            s0 = rng.integers(0, len(fr) - L)
            fr[s0:s0 + L] = rng.standard_normal(L) * np.hanning(L)
        sos = butter(4, min(3000.0, 0.4 * sr), btype="high", fs=sr, output="sos")
        fr = sosfilt(sos, fr)
        v = v / (np.std(v) + 1e-12) + 0.35 * fr
        g = np.ones(i1 - i0)
        g[:ramp] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
        g[-ramp:] = g[:ramp][::-1]
        out[i0:i1] = v * g
        pos += dur_w + rng.uniform(0.25, 0.7)
    active = out != 0
    out *= level_rms / (np.sqrt(np.mean(out[active] ** 2)) + 1e-12)
    return out


def room_noise(n: int, sr: int = SR, seed: int = 1, level_rms: float = 0.002) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    rumble = lfilter([1.0], [1, -0.97], rng.standard_normal(n))
    x = 0.5 * w / w.std() + rumble / rumble.std()
    return x * level_rms / x.std()


def mains_hum(n: int, sr: int = SR, seed: int = 2, freq: float = 50.0, amp: float = 0.003,
              drift_hz: float = 0.02) -> np.ndarray:
    """Hum whose frequency wanders slowly (random, smooth) around ``freq``."""
    rng = np.random.default_rng(seed)
    dur = n / sr
    f = freq + drift_hz * _ctrl(rng, dur, 4.0, n, sr)
    ph = 2 * np.pi * np.cumsum(f) / sr + rng.uniform(0, 2 * np.pi)
    return amp * np.sin(ph) + 0.4 * amp * np.sin(2 * ph + 0.7)


def recording(duration: float = 30.0, sr: int = SR, seed: int = 0, hum: bool = True,
              noise_gain: float = 1.0) -> np.ndarray:
    """Clean 'original' recording: speech + room noise (+ ENF hum)."""
    n = int(duration * sr)
    x = speech_like(duration, sr, seed) + noise_gain * room_noise(n, sr, seed + 100)
    if hum:
        x = x + mains_hum(n, sr, seed + 200)
    return x


# --------------------------------------------------------------------------- #
# Forgeries (each returns samples and the ground-truth edit time(s) in seconds)
# --------------------------------------------------------------------------- #
def forge_cut(x: np.ndarray, sr: int = SR, at: float = 14.0, remove: float = 0.73
              ) -> Tuple[np.ndarray, float]:
    """Delete ``remove`` seconds starting at ``at`` (ENF phase jump of 2*pi*50*0.73)."""
    i0, i1 = int(at * sr), int((at + remove) * sr)
    return np.concatenate([x[:i0], x[i1:]]), at


def foreign_segment(duration: float = 3.0, sr: int = SR, seed: int = 7) -> np.ndarray:
    """Speech from a different 'room' and device: 4x noise, no hum, recorded at
    16 kHz (8 kHz bandwidth) and upsampled to ``sr``."""
    n = int(duration * sr)
    y = speech_like(duration, sr, seed) + 4.0 * room_noise(n, sr, seed + 100)
    lo = resample_poly(y, 16000, sr)
    return resample_poly(lo, sr, 16000)[:n]


def forge_insert(x: np.ndarray, sr: int = SR, at: float = 12.0, seg: np.ndarray = None
                 ) -> Tuple[np.ndarray, Tuple[float, float]]:
    seg = foreign_segment(sr=sr) if seg is None else seg
    i0 = int(at * sr)
    return np.concatenate([x[:i0], seg, x[i0:]]), (at, at + len(seg) / sr)


def forge_zeros(x: np.ndarray, sr: int = SR, at: float = 17.3, dur: float = 0.3
                ) -> Tuple[np.ndarray, float]:
    i0 = int(at * sr)
    return np.concatenate([x[:i0], np.zeros(int(dur * sr)), x[i0:]]), at


def tts_like(duration: float = 30.0, sr: int = SR, seed: int = 3) -> np.ndarray:
    """'TTS-like': 24 kHz synthetic speech upsampled to 48 kHz, digital-silence pauses."""
    y = speech_like(duration, 24000, seed)
    return resample_poly(y, sr // 24000 if sr % 24000 == 0 else sr, 1 if sr % 24000 == 0 else 24000)


def make_all(duration: float = 30.0, sr: int = SR) -> Dict[str, Tuple[np.ndarray, object]]:
    clean = recording(duration, sr, seed=0)
    cut, t_cut = forge_cut(clean, sr)
    ins, t_ins = forge_insert(clean, sr)
    zer, t_zer = forge_zeros(clean, sr)
    return {
        "clean": (quantize16(clean), None),
        "cut": (quantize16(cut), t_cut),
        "insert": (quantize16(ins), t_ins),
        "zeros": (quantize16(zer), t_zer),
        "tts": (quantize16(tts_like(duration, sr)), None),
    }


# --------------------------------------------------------------------------- #
# A more realistic recording chain (benchmarks/audio_robustness.py)
#
# ``simulate_recording`` turns any dry speech signal (the benchmark uses offline
# text-to-speech output; the tests use ``speech_like``) into something closer to a
# microphone recording than ``recording`` above: reverberation, coloured room noise
# whose level wanders, optional mains hum with harmonics, and low-level background
# events. It is still a simulation -- no real microphone, pre-amp, AGC or codec chain.
# --------------------------------------------------------------------------- #
def room_ir(sr: int = SR, rt60: float = 0.35, seed: int = 0, drr_db: float = 8.0) -> np.ndarray:
    """Synthetic room impulse response: direct sound, three early reflections and an
    exponentially decaying noise tail (60 dB down after ``rt60`` s) whose high
    frequencies die faster. ``drr_db`` = direct-to-reverberant energy ratio."""
    rng = np.random.default_rng(seed)
    n = int(min(1.5, 1.3 * rt60) * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * 10 ** (-3.0 * t / rt60)
    dark = sosfilt(butter(1, 3500.0, btype="low", fs=sr, output="sos"), tail)
    mix = np.clip(t / (0.6 * rt60), 0, 1)            # tail gets duller with time
    tail = (1 - mix) * tail + mix * dark * 2.0
    tail[: int(0.004 * sr)] = 0.0
    for _ in range(3):
        tail[int(rng.uniform(0.004, 0.03) * sr)] += rng.uniform(2.0, 5.0) * rng.choice([-1, 1])
    tail *= 10 ** (-drr_db / 20) / (np.sqrt(np.sum(tail ** 2)) + 1e-12)
    ir = tail
    ir[0] = 1.0
    return ir


def coloured_noise(n: int, sr: int = SR, seed: int = 0, level_db: float = -55.0,
                   tilt: float = 0.5, wander_db: float = 1.0) -> np.ndarray:
    """Room noise: 1/f^tilt-shaped Gaussian noise + low-frequency rumble + a white
    (microphone self-noise) component; RMS ``level_db`` dBFS; its level wanders slowly
    by about ``wander_db`` (ventilation, traffic)."""
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    shape = 1.0 / np.maximum(f, 40.0) ** tilt
    shape = shape + 4.0 * shape[0] * np.exp(-f / 120.0)       # rumble below ~150 Hz
    x = np.fft.irfft(spec * shape, n)
    x = x / x.std() + 0.15 * rng.standard_normal(n)
    x *= 10 ** (wander_db * _ctrl(rng, n / sr, 3.0, n, sr) / 20.0)
    return x * 10 ** (level_db / 20) / x.std()


def hum_harmonics(n: int, sr: int = SR, seed: int = 0, freq: float = 50.0,
                  level_db: float = -58.0, drift_hz: float = 0.02) -> np.ndarray:
    """Mains hum with odd-heavy harmonics up to the 5th; frequency wanders like ENF."""
    rng = np.random.default_rng(seed)
    f = freq + drift_hz * _ctrl(rng, n / sr, 4.0, n, sr)
    ph = 2 * np.pi * np.cumsum(f) / sr + rng.uniform(0, 2 * np.pi)
    rel = (1.0, rng.uniform(0.2, 0.6), rng.uniform(0.3, 0.8), rng.uniform(0.05, 0.2),
           rng.uniform(0.1, 0.4))
    x = sum(a * np.sin(k * ph + rng.uniform(0, 2 * np.pi)) for k, a in enumerate(rel, start=1))
    return x * 10 ** (level_db / 20) * np.sqrt(2.0)


def background_events(n: int, sr: int = SR, seed: int = 0, count: int = 3,
                      level_db: float = -45.0) -> np.ndarray:
    """A few short low-level sounds a real room produces (a creak, paper, a tap):
    band-passed noise bursts of 30-300 ms with a fast attack and slow decay."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for _ in range(count):
        L = int(rng.uniform(0.03, 0.3) * sr)
        if L >= n:
            continue
        lo = rng.uniform(150.0, 2500.0)
        hi = min(lo * rng.uniform(1.5, 6.0), 0.45 * sr)
        b = sosfilt(butter(2, [lo, hi], btype="band", fs=sr, output="sos"),
                    rng.standard_normal(L))
        env = np.exp(-np.arange(L) / (rng.uniform(0.1, 0.4) * L))
        env[: max(1, int(0.003 * sr))] *= np.linspace(0, 1, max(1, int(0.003 * sr)))
        b = b * env
        s0 = rng.integers(0, n - L)
        out[s0:s0 + L] += b * 10 ** ((level_db + rng.uniform(-6, 6)) / 20) / (b.std() + 1e-12)
    return out


def slow_agc(x: np.ndarray, sr: int = SR, target_db: float = -23.0, max_gain_db: float = 10.0,
             max_cut_db: float = 6.0, attack_s: float = 0.15, release_s: float = 1.5
             ) -> np.ndarray:
    """Simple automatic gain control as found in phones and dictaphones: the gain follows
    the 10 ms block level (fast when the level rises, slowly when it falls), so the noise
    floor is pumped up during pauses. Limited to ``-max_cut_db`` .. ``+max_gain_db``."""
    blk = max(1, int(0.01 * sr))
    m = len(x) // blk
    if m < 2:
        return x
    lev = 10 * np.log10(np.mean(x[: m * blk].reshape(m, blk) ** 2, axis=1) + 1e-12)
    a_up, a_dn = np.exp(-0.01 / attack_s), np.exp(-0.01 / release_s)
    sm = np.empty(m)
    cur = lev[0]
    for i in range(m):
        a = a_up if lev[i] > cur else a_dn
        cur = a * cur + (1 - a) * lev[i]
        sm[i] = cur
    gain_db = np.clip(target_db - 6.0 - sm, -max_cut_db, max_gain_db)
    g = np.interp(np.arange(len(x)), (np.arange(m) + 0.5) * blk, gain_db)
    return x * 10 ** (g / 20)


def simulate_recording(speech: np.ndarray, sr: int = SR, seed: int = 0, *,
                       speech_db: float = -23.0, rt60: float = 0.35, noise_db: float = -55.0,
                       noise_tilt: float = 0.5, hum_db: float = None, hum_freq: float = 50.0,
                       events: int = 2, agc: bool = False) -> np.ndarray:
    """Dry speech -> simulated room recording (float64, not yet quantised).

    ``speech_db`` is the RMS level of the active speech; ``hum_db=None`` means a
    recording without mains hum (battery-powered device away from the grid);
    ``agc=True`` passes the result through :func:`slow_agc`.
    """
    from scipy.signal import fftconvolve

    x = np.asarray(speech, dtype=np.float64)
    n = len(x)
    wet = fftconvolve(x, room_ir(sr, rt60, seed + 11), mode="full")[:n]
    act = np.abs(wet) > 0.02 * np.max(np.abs(wet) + 1e-12)
    wet *= 10 ** (speech_db / 20) / (np.sqrt(np.mean(wet[act] ** 2)) + 1e-12)
    y = wet + coloured_noise(n, sr, seed + 12, noise_db, noise_tilt)
    if hum_db is not None:
        y = y + hum_harmonics(n, sr, seed + 13, hum_freq, hum_db)
    if events:
        y = y + background_events(n, sr, seed + 14, events, noise_db + 12.0)
    if agc:
        y = slow_agc(y, sr, target_db=speech_db)
    return np.clip(y, -0.999, 0.999)
