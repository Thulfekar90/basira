"""Regression tests for issues found in independent code review."""
from __future__ import annotations

import os
import struct
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import basira
from basira.core.fusion import finalize
from basira.core.types import Category, Evidence, Report

import synth_image as S


def test_generator_name_in_a_title_is_not_proof(tmp_path):
    meta = PngInfo()
    meta.add_text("Title", "Firefly at night")
    p = str(tmp_path / "firefly.png")
    Image.fromarray(S.camera_photo(2, 384, 512)).save(p, pnginfo=meta)
    r = basira.analyze(p)
    assert r.verdict_level < 4 and not r.get("image.generator_metadata").decisive


def test_generator_in_software_field_is_proof(tmp_path):
    meta = PngInfo()
    meta.add_text("Software", "NovelAI")
    p = str(tmp_path / "nai.png")
    Image.fromarray(S.camera_photo(2, 384, 512)).save(p, pnginfo=meta)
    assert basira.analyze(p).verdict_level == 4


def test_non_finite_score_becomes_a_failed_detector():
    e = Evidence("x.nan", "NaN detector", Category.MANIPULATION, float("nan"), 0.9, "s")
    assert e.error and e.score == 0.0 and e.confidence == 0.0
    r = finalize(Report(path="x", media_type="image", info={}, evidences=[e]))
    assert all(np.isfinite(v) for v in r.scores.values())
    assert r.verdict_level == 1 and "could not be analysed" in r.verdict


@pytest.mark.parametrize("size", [(2000, 1), (1, 2000), (3000, 2)])
def test_extremely_thin_images(tmp_path, size):
    p = str(tmp_path / "thin.png")
    Image.fromarray(np.zeros((size[1], size[0], 3), np.uint8)).save(p)
    basira.analyze(p)


def test_pixel_limit_is_enforced_before_decoding(tmp_path):
    from basira.image.load import load_image

    p = str(tmp_path / "big.png")
    Image.fromarray(np.zeros((400, 500, 3), np.uint8)).save(p)
    with pytest.raises(ValueError, match="limit"):
        load_image(p, max_pixels=100_000)


@pytest.mark.parametrize("body", [b"#EXTM3U\nhttp://example.invalid/a.ts\n",
                                  b"ffconcat version 1.0\nfile '/etc/hosts'\n"])
def test_playlists_and_reference_files_are_refused(tmp_path, body):
    p = tmp_path / "clip.mp4"
    p.write_bytes(body)
    with pytest.raises(ValueError, match="playlist"):
        basira.analyze(str(p))


def test_out_of_phase_stereo_is_not_mixed_to_silence():
    from basira.audio._dsp import to_mono

    x = (np.random.RandomState(0).randn(48000) * 0.1).astype(np.float32)
    mono = to_mono(np.stack([x, -x], axis=1))
    assert float(np.abs(mono).mean()) > 0.01


def test_edit_list_entries_never_read_past_their_box():
    from basira.video.container import _parse_moov

    def box(t, payload):
        return struct.pack(">I4s", 8 + len(payload), t) + payload

    elst = box(b"elst", struct.pack(">II", 0, 3))        # claims 3 entries, holds none
    free = box(b"free", struct.pack(">IiI", 9999, 1234, 0x00010000) * 4)
    hdlr = box(b"hdlr", b"\0" * 8 + b"vide" + b"\0" * 12)
    trak = box(b"trak", box(b"edts", elst + free) + box(b"mdia", hdlr))
    out = _parse_moov(box(b"mvhd", b"") + trak)          # empty mvhd must not raise either
    assert out["edit_entries"] == 0


def test_local_server_requires_token_and_local_host(tmp_path):
    from http.server import ThreadingHTTPServer

    from basira.server import Handler

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    srv.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    srv.token = "secret-token"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    path, _ = S.make_authentic(str(tmp_path), 0)
    data = open(path, "rb").read()
    url = f"http://127.0.0.1:{port}/analyze?name=a.jpg"

    def post(headers):
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        try:
            return urllib.request.urlopen(req, timeout=60).status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        assert post({}) == 403                                             # no token
        assert post({"X-Basira-Token": "wrong"}) == 403
        assert post({"X-Basira-Token": "secret-token", "Host": "evil.example"}) == 403
        assert post({"X-Basira-Token": "secret-token",
                     "Origin": "http://evil.example"}) == 403
        assert post({"X-Basira-Token": "secret-token"}) == 200
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).read().decode()
        assert "secret-token" in page and "Drop an image" in page
    finally:
        srv.shutdown()
        srv.server_close()
