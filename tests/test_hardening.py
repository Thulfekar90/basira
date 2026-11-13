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


# ---- second review round -------------------------------------------------------------
def test_chunk_named_parameters_without_generator_content_is_not_proof(tmp_path):
    meta = PngInfo()
    meta.add_text("parameters", "ordinary camera parameters")
    meta.add_text("prompt", '{"note": "not a graph"}')
    p = str(tmp_path / "plain.png")
    Image.fromarray(S.camera_photo(2, 384, 512)).save(p, pnginfo=meta)
    r = basira.analyze(p)
    assert r.verdict_level < 4 and not r.get("image.generator_metadata").decisive


def test_real_comfyui_graph_is_proof(tmp_path):
    meta = PngInfo()
    meta.add_text("prompt", '{"3": {"class_type": "KSampler", "inputs": {"seed": 1}}}')
    p = str(tmp_path / "comfy.png")
    Image.fromarray(S.camera_photo(2, 384, 512)).save(p, pnginfo=meta)
    assert basira.analyze(p).verdict_level == 4


def test_partial_stereo_cancellation_is_not_turned_into_silence():
    from basira.audio._dsp import to_mono

    x = (np.random.RandomState(0).randn(48000 * 4) * 0.1).astype(np.float32)
    right = x.copy()
    right[48000:52800] *= -1                      # 0.1 s out of phase
    mono = to_mono(np.stack([x, right], axis=1))
    assert float(np.abs(mono[48000:52800]).mean()) > 0.01


def test_comb_detector_returns_the_fundamental_period():
    from basira.image.jpeg import _period

    assert _period(np.resize([900, 100, 9000], 97), 48)[0] == 3
    assert _period(np.resize([50, 900, 60, 40, 5000], 97), 48)[0] == 5
    rs = np.random.RandomState(0)
    lap = np.clip(np.round(rs.laplace(0, 6, 200000)).astype(int), -48, 48)
    assert _period(np.bincount(lap + 48, minlength=97), 48)[0] == 1


def test_dq_block_score_is_calibrated_under_both_hypotheses():
    """Untouched coefficients must average ~0 and tampered ones ~1 on the scored support."""
    from basira.image.jpeg import _llr_tables

    rs = np.random.RandomState(1)
    h = np.resize([900, 100, 9000, 300, 50], 97)
    llr, eu, et, live = _llr_tables(h, 5, 3.0, 48)
    b = np.flatnonzero(live)
    norm = lambda sm: (llr[sm].sum() - eu[sm].sum()) / (et[sm].sum() - eu[sm].sum())  # noqa: E731
    assert abs(norm(rs.choice(b, 20000, p=h[b] / h[b].sum()))) < 0.05
    assert abs(norm(rs.choice(b, 20000)) - 1) < 0.05
    # A window where only one bin survives the noise floor cannot discriminate: not scored.
    h1 = np.zeros(97, int)
    h1[50:53] = [90, 9, 1]
    assert not _llr_tables(h1, 3, 1.0, 48)[3][52]


def test_failed_child_does_not_make_a_failed_parent_look_clean():
    bad = Evidence.failed("video.decode", "Frame decoding", Category.MANIPULATION, RuntimeError("x"))
    child = finalize(Report(path="x", media_type="audio", info={}, evidences=[
        Evidence.failed("audio.enf", "ENF", Category.MANIPULATION, RuntimeError("y"))]))
    r = finalize(Report(path="x", media_type="video", info={}, evidences=[bad], children=[child]))
    assert r.verdict_level == 1 and "could not be analysed" in r.verdict


def test_opencv_protocol_whitelist_survives_existing_options(monkeypatch):
    from basira.core.io import _enforce_opencv_whitelist

    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|protocol_whitelist;http")
    _enforce_opencv_whitelist()
    v = os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
    assert "rtsp_transport;tcp" in v and v.endswith("protocol_whitelist;file,pipe")
    assert "http" not in v
