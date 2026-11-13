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
