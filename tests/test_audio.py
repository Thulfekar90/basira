"""Audio engine tests on deterministic synthetic recordings with known ground truth."""
from __future__ import annotations

import os
import struct
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import synth_audio as S  # noqa: E402

from basira.audio import analyze_audio  # noqa: E402
from basira.core.types import Category  # noqa: E402

TOL = 0.5  # seconds


def near(segments, t, tol=TOL, max_len=6.0):
    """At least one (reasonably short) segment within ``tol`` s of time ``t``."""
    return any(s.start - tol <= t <= s.end + tol and (s.end - s.start) <= max_len
               for s in segments)


@pytest.fixture(scope="module")
def signals():
    return S.make_all()


@pytest.fixture(scope="module")
def reports(signals, tmp_path_factory):
    d = tmp_path_factory.mktemp("audio")
    out = {}
    for name, (x, _) in signals.items():
        p = str(d / f"{name}.wav")
        sf.write(p, x, S.SR, subtype="PCM_16")
        out[name] = analyze_audio(p)
    return out


def _manip(report):
    return [e for e in report.evidences if e.category == Category.MANIPULATION]


def test_report_contract(reports):
    r = reports["clean"]
    assert r.media_type == "audio"
    assert r.info["sample_rate"] == S.SR and abs(r.info["duration"] - 30.0) < 0.01
    assert r.info["subtype"] == "PCM_16"
    ids = [e.detector for e in r.evidences]
    for det in ("audio.metadata", "audio.enf", "audio.splice", "audio.silence",
                "audio.bandwidth", "audio.bandwidth_shift", "audio.synthetic"):
        assert det in ids
    assert all(e.error is None for e in r.evidences), [e.error for e in r.evidences]
    assert r.preview is not None and r.preview.dtype == np.uint8 and r.preview.ndim == 3
    for e in r.evidences:
        for s in e.series.values():
            assert set(s) >= {"x", "y", "xlabel", "ylabel", "markers"}
            assert len(s["x"]) == len(s["y"]) <= 1500
    r.to_dict()  # JSON-serialisable


def test_clean_is_clean(reports):
    r = reports["clean"]
    bad = [(e.detector, e.score) for e in _manip(r) if e.score > 0.5]
    assert not bad, bad
    assert r.verdict_level <= 1, (r.verdict, r.scores)
    enf = r.get("audio.enf")
    assert enf.confidence > 0.5, "the synthetic recording carries 50 Hz hum"
    assert enf.details["hum_nominal_hz"] == 50.0


def test_cut_detected_by_enf(reports, signals):
    t = signals["cut"][1]
    e = reports["cut"].get("audio.enf")
    assert e.score > 0.5, e.summary
    assert near(e.segments, t), [s.to_dict() for s in e.segments]
    assert reports["cut"].verdict_level >= 2


def test_insert_from_other_room(reports, signals):
    t0, t1 = signals["insert"][1]
    r = reports["insert"]
    sp = r.get("audio.splice")
    assert sp.score > 0.5, sp.summary
    assert near(sp.segments, t0) and near(sp.segments, t1), [s.to_dict() for s in sp.segments]
    bw = r.get("audio.bandwidth_shift")
    assert bw.score > 0.5, bw.summary
    assert near(bw.segments, t0, max_len=4.0), [s.to_dict() for s in bw.segments]
    # the long-term bandwidth is still full: the insert is a minority
    assert r.get("audio.bandwidth").score < 0.5


def test_inserted_digital_silence(reports, signals):
    t = signals["zeros"][1]
    e = reports["zeros"].get("audio.silence")
    assert e.score > 0.5 and e.confidence >= 0.8, e.summary
    assert near(e.segments, t)
    assert e.details["n_runs"] == 1


def test_tts_like(reports):
    tts, clean = reports["tts"], reports["clean"]
    assert tts.scores["ai_generation"] > clean.scores["ai_generation"] + 0.3, \
        (tts.scores, clean.scores)
    bw = tts.get("audio.bandwidth")
    assert bw.score > 0.5 and 11000 < bw.details["cutoff_hz"] < 13500, bw.summary
    assert tts.get("audio.synthetic").score > 0.5
    # systematic digital silence is reported as a property of the file, not an edit
    assert tts.get("audio.silence").score < 0.5


def test_false_positives_noise_and_sweep(tmp_path):
    rng = np.random.default_rng(5)
    n = 20 * S.SR
    noise = (0.1 * rng.standard_normal(n)).astype(np.float32)
    t = np.arange(n) / S.SR
    sweep = 0.5 * np.sin(2 * np.pi * 20 * 20 / np.log(1000) * (np.exp(t / 20 * np.log(1000)) - 1))
    for name, x in (("noise", noise), ("sweep", S.quantize16(sweep))):
        p = str(tmp_path / f"{name}.wav")
        sf.write(p, x, S.SR, subtype="PCM_16")
        r = analyze_audio(p)
        bad = [(e.detector, round(e.score, 3)) for e in _manip(r) if e.score > 0.5]
        assert not bad, (name, bad)
        assert r.scores["manipulation"] < 0.2, (name, r.scores)


def _wav_with_info(path, x, sr, software):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    val = software.encode() + b"\x00"
    if len(val) % 2:
        val += b"\x00"
    info = b"INFO" + b"ISFT" + struct.pack("<I", len(val)) + val
    lst = b"LIST" + struct.pack("<I", len(info)) + info
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    data = b"data" + struct.pack("<I", len(pcm)) + pcm
    body = b"WAVE" + fmt + lst + data
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)


def test_metadata_elevenlabs(tmp_path, signals):
    x = signals["tts"][0][: 5 * S.SR]
    p = str(tmp_path / "gen.wav")
    _wav_with_info(p, x, S.SR, "ElevenLabs Text to Speech")
    r = analyze_audio(p)
    m = r.get("audio.metadata")
    assert m.category == Category.AI_GENERATION and m.decisive and m.score >= 0.9, m.summary
    assert m.details["tags"].get("software", "").startswith("ElevenLabs")
    assert r.verdict_level == 4


def test_metadata_editor_and_plain(tmp_path, signals):
    x = signals["clean"][0][: 5 * S.SR]
    p = str(tmp_path / "edited.wav")
    _wav_with_info(p, x, S.SR, "Audacity 3.4")
    m = analyze_audio(p).get("audio.metadata")
    assert m.category == Category.PROVENANCE and not m.decisive and 0.6 < m.score < 0.9
    q = str(tmp_path / "plain.wav")
    sf.write(q, x, S.SR, subtype="PCM_16")
    assert analyze_audio(q).get("audio.metadata").score < 0.5


@pytest.mark.parametrize("fmt,ext", [("FLAC", "flac"), ("MP3", "mp3")])
def test_compressed_formats(tmp_path, signals, fmt, ext):
    x, t = signals["zeros"]
    p = str(tmp_path / f"zeros.{ext}")
    sf.write(p, x, S.SR, format=fmt)
    r = analyze_audio(p)
    assert all(e.error is None for e in r.evidences), [e.error for e in r.evidences]
    assert abs(r.info["duration"] - len(x) / S.SR) < 0.2
    assert r.info["lossy"] == (fmt == "MP3")
    if fmt == "FLAC":  # lossless: identical decision
        e = r.get("audio.silence")
        assert e.score > 0.5 and near(e.segments, t)
    else:  # mp3: no manipulation flags away from the edit (codec smearing at the edit is ok)
        bad = [s for e in _manip(r) if e.detector != "audio.silence" and e.score > 0.5
               for s in e.segments if not near([s], t, tol=1.0)]
        assert not bad, [s.to_dict() for s in bad]


def test_samples_input(signals):
    x, t = signals["cut"]
    stereo = np.stack([x, x], axis=1)
    r = analyze_audio("track_from_video.mp4", samples=stereo, sr=S.SR)
    assert r.info["source"] == "samples" and r.info["channels"] == 2
    assert r.get("audio.metadata").confidence == 0.0  # no file on disk -> not applicable
    e = r.get("audio.enf")
    assert e.score > 0.5 and near(e.segments, t)


def test_short_and_silent_inputs():
    r = analyze_audio("short", samples=np.zeros(4000, np.float32), sr=16000)
    assert all(e.error is None for e in r.evidences)
    assert r.verdict_level == 0
    rng = np.random.default_rng(0)
    r = analyze_audio("tiny", samples=0.01 * rng.standard_normal(800).astype(np.float32), sr=8000)
    assert all(e.error is None for e in r.evidences), [e.error for e in r.evidences]
