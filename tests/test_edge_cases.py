"""Unusual and broken inputs must never crash the engines or raise false alarms."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

import basira
from basira.core.report_html import render_html

import synth_image as S


@pytest.fixture(scope="module")
def d(tmp_path_factory):
    return str(tmp_path_factory.mktemp("edge"))


@pytest.fixture(scope="module")
def base():
    return S.camera_photo(1, 480, 640)


def _check(path, max_level=1):
    r = basira.analyze(path)
    assert [e.error for e in r.evidences if e.error] == []
    assert r.verdict_level <= max_level, (r.verdict, r.scores)
    json.dumps(r.to_dict())
    assert "<html" in render_html(r)
    return r


@pytest.mark.parametrize("name,mode,kw", [
    ("gray.jpg", "L", {"quality": 85}), ("gray.png", "L", {}), ("pal.gif", "P", {}),
    ("cmyk.jpg", "CMYK", {"quality": 85}), ("img.webp", "RGB", {"quality": 80}),
    ("img.tif", "RGB", {}), ("prog.jpg", "RGB", {"quality": 85, "progressive": True}),
])
def test_image_modes_and_formats(d, base, name, mode, kw):
    p = os.path.join(d, name)
    Image.fromarray(base).convert(mode).save(p, **kw)
    _check(p)


def test_alpha_tiny_flat_and_noise_images(d, base):
    cases = {
        "rgba.png": np.dstack([base, np.full(base.shape[:2], 200, np.uint8)]),
        "tiny.png": base[:20, :30], "one.png": base[:1, :1],
        "black.png": np.zeros((300, 400, 3), np.uint8),
        "white.png": np.full((300, 400, 3), 255, np.uint8),
        "noise.png": (np.random.RandomState(0).rand(300, 400, 3) * 255).astype(np.uint8),
    }
    for name, arr in cases.items():
        p = os.path.join(d, name)
        Image.fromarray(arr).save(p)
        _check(p, max_level=0)


def test_truncated_jpeg_is_decoded_and_noted(d, base):
    full = os.path.join(d, "full.jpg")
    Image.fromarray(base).save(full, quality=85)
    raw = open(full, "rb").read()
    p = os.path.join(d, "trunc.jpg")
    open(p, "wb").write(raw[: len(raw) // 2])
    r = basira.analyze(p)
    assert "truncated" in r.info.get("notes", "")


def test_garbage_and_hostile_mp4(d):
    for name, body in (("garbage.mp4", os.urandom(4000)),
                       ("badbox.mp4", b"\0" * 12 + b"\xff\xff\xff\xffmoov" + b"\x00\x00\x00\x01trak"
                        + os.urandom(200))):
        p = os.path.join(d, name)
        open(p, "wb").write(b"\x00\x00\x00\x18ftypmp42" + body)
        r = basira.analyze(p)                 # must not raise
        # ...and must not present an undecodable file as a clean result.
        assert r.verdict_level == 1 and "could not be analysed" in r.verdict
        json.dumps(r.to_dict())


def test_empty_short_stereo_and_clipped_audio(d):
    rs = np.random.RandomState(1)
    cases = {"empty.wav": np.zeros((0,), np.float32), "short.wav": np.zeros((100,), np.float32),
             "stereo.wav": (rs.randn(48000 * 3, 2) * 0.05).astype(np.float32),
             "clip.wav": np.ones((48000,), np.float32)}
    for name, x in cases.items():
        p = os.path.join(d, name)
        sf.write(p, x, 48000)
        _check(p, max_level=0)


def test_unknown_file_type_is_a_clean_error(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    with pytest.raises(ValueError, match="Unrecognised media type: notes.txt"):
        basira.analyze(str(p))
