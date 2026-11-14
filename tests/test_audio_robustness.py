"""What the audio engine does on more realistic material than tests/test_audio.py:
reverberant simulated recordings with coloured room noise, automatic gain control and
codecs (tests/synth_audio.py: simulate_recording). Every test here pins a behaviour
measured by benchmarks/audio_robustness.py -- including the blind spots, so that nobody
"fixes" them by accident into false alarms.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import synth_audio as S  # noqa: E402

from basira.audio import analyze_audio  # noqa: E402
from basira.audio.bandwidth import SHIFT_CONFIDENCE  # noqa: E402
from basira.audio.enf import detect_enf  # noqa: E402
from basira.audio.splice import CONFIDENCE_MAX, detect_splice  # noqa: E402
from basira.core.io import ffmpeg_exe  # noqa: E402

SR = 44100
DUR = 24.0
SUSPICIOUS = 0.45      # category score at which the verdict becomes "Suspicious"


def _rec(seed: int, **room) -> np.ndarray:
    return S.simulate_recording(S.speech_like(DUR, SR, seed), SR, seed, **room)


@pytest.fixture(scope="module")
def plain():
    return _rec(1)


@pytest.mark.parametrize("room", [
    {},                                             # ordinary room, no hum
    {"agc": True},                                  # gain control pumps the noise floor
    {"rt60": 0.6},                                  # reverberant: few pauses reach the floor
    {"agc": True, "hum_db": -55.0, "hum_freq": 60.0, "noise_db": -48.0},
])
def test_clean_simulated_recordings_stay_unflagged(room):
    r = analyze_audio("", samples=S.quantize16(_rec(2, **room)), sr=SR)
    assert all(e.error is None for e in r.evidences)
    assert r.scores["manipulation"] < 0.2, (room, r.scores, r.explanation)
    assert not r.fused_segments, [s.to_dict() for s in r.fused_segments]


def test_switch_to_a_room_with_another_noise_spectrum_is_located(plain):
    a = _rec(1, noise_db=-56.0, noise_tilt=0.2)
    b = _rec(1, noise_db=-50.0, noise_tilt=1.0)
    at = 12.3
    i0 = int(at * SR)
    e = detect_splice(S.quantize16(np.concatenate([a[:i0], b[i0:]])), SR)
    assert e.score > 0.5, e.summary
    # the edit is bracketed by the last pause of the old room and the first of the new one
    assert any(s.start - 0.5 <= at <= s.end + 0.5 for s in e.segments), \
        [s.to_dict() for s in e.segments]
    assert "background spectrum" in e.segments[0].label
    # a background change alone is "Suspicious" at most
    assert e.confidence <= CONFIDENCE_MAX < 0.75


def test_floor_level_change_alone_is_not_evidence():
    """Blind spot on purpose: gain control, or a speaker moving, shifts the floor of an
    untouched recording by many dB. Same noise spectrum, +10 dB from 12 s on."""
    dry = S.speech_like(DUR, SR, 3)
    n = len(dry)
    gain = np.ones(n)
    gain[int(12.0 * SR):] = 10 ** (10 / 20)
    x = 0.5 * dry + S.coloured_noise(n, SR, 3, -58.0, 0.5, wander_db=0.5) * gain
    e = detect_splice(S.quantize16(x), SR)
    assert e.details["background_measurable"]
    assert len(e.details["states"]) == 2, e.details["states"]      # the change is seen ...
    assert e.score < 0.5 and not e.segments, e.summary             # ... and not reported


def test_no_pauses_means_not_measurable_not_consistent():
    rng = np.random.default_rng(0)
    t = np.arange(int(10 * SR)) / SR
    x = (0.1 * rng.standard_normal(len(t)) * (1.0 + 0.9 * np.sin(2 * np.pi * 3.0 * t)))
    e = detect_splice(x.astype(np.float32), SR)
    assert not e.details["background_measurable"]
    assert e.score < 0.5 and "not measurable" in e.summary
    assert e.confidence <= 0.25


def _with_noise(sig: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return S.quantize16(sig + 0.003 * rng.standard_normal(len(sig)))


def test_enf_declines_tones_that_are_not_mains_hum():
    n = int(20 * SR)
    t = np.arange(n) / SR
    # a held bass note 0.2 Hz away from 60 Hz
    e = detect_enf(_with_noise(0.05 * np.sin(2 * np.pi * 59.8 * t)), SR)
    assert e.confidence == 0.0 and "away from the mains frequency" in e.summary
    # a bass line at 60 Hz: decaying notes of random length and level, each starting at its
    # own phase -- every note onset would be a "phase jump"
    for seed in range(4):
        rng = np.random.default_rng(seed)
        x = np.zeros(n)
        pos = 0.3
        while pos < 19.0:
            dur = rng.uniform(0.6, 2.5)
            i0, i1 = int(pos * SR), int(min(pos + dur, 20.0) * SR)
            tt = t[i0:i1] - t[i0]
            x[i0:i1] += (0.05 * 10 ** (rng.uniform(-15, 0) / 20) * np.exp(-tt / rng.uniform(0.5, 2.0))
                         * np.sin(2 * np.pi * 60.0 * tt + rng.uniform(0, 2 * np.pi)))
            pos += dur + rng.uniform(0.0, 0.8)
        e = detect_enf(_with_noise(x, seed), SR)
        assert e.score < 0.5 and e.confidence <= 0.3 and not e.segments, (seed, e.summary)
    # hum-like tone that is loud in a fifth of the file only
    loud = (t % 5.0) < 1.0
    e = detect_enf(_with_noise(np.where(loud, 0.05, 0.003) * np.sin(2 * np.pi * 60.0 * t)), SR)
    assert e.confidence == 0.0 and "not steady" in e.summary, e.summary
    # real hum is accepted, and the report says how much the test could have seen
    e = detect_enf(_with_noise(0.01 * np.sin(2 * np.pi * 50.0 * t)), SR)
    assert e.confidence > 0.5 and e.score < 0.5
    assert 0.5 < e.details["visible_cut_fraction"] <= 1.0


def test_enf_cut_in_a_reverberant_recording_with_strong_hum():
    x = _rec(4, hum_db=-45.0, noise_db=-60.0)
    y, at = S.forge_cut(x, SR, at=11.0, remove=0.73)       # 0.73 s at 50 Hz: half a cycle
    clean = detect_enf(S.quantize16(x), SR)
    cut = detect_enf(S.quantize16(y), SR)
    assert clean.score < 0.5, clean.summary
    assert cut.score > 0.5, cut.summary
    assert any(s.start - 0.5 <= at <= s.end + 0.5 for s in cut.segments)


def test_bandwidth_shift_cannot_flag_on_its_own():
    # produced audio changes bandwidth by design (ring tones); needs a second detector
    assert SHIFT_CONFIDENCE < SUSPICIOUS


@pytest.mark.parametrize("ext,args", [
    ("mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
    ("opus", ["-c:a", "libopus", "-b:a", "24k"]),
])
def test_clean_recording_survives_a_messenger_codec(tmp_path, plain, ext, args):
    exe = ffmpeg_exe()
    if not exe:
        pytest.skip("no ffmpeg")
    enc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if args[1] not in enc:
        pytest.skip(f"ffmpeg without {args[1]}")
    src, dst = str(tmp_path / "clean.wav"), str(tmp_path / f"clean.{ext}")
    sf.write(src, S.quantize16(plain), SR, subtype="PCM_16")
    subprocess.run([exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", src,
                    *args, dst], check=True, capture_output=True)
    r = analyze_audio(dst)
    assert all(e.error is None for e in r.evidences), [e.error for e in r.evidences]
    assert r.info["lossy"]
    assert r.scores["manipulation"] < SUSPICIOUS, (r.scores, r.explanation)
