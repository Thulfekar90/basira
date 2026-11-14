"""Regression tests for issues found in independent code review."""
from __future__ import annotations

import os
import struct
import sys
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
        req = urllib.request.Request(url + "&lang=ar", data=data, method="POST",
                                     headers={"X-Basira-Token": "secret-token"})
        ar = urllib.request.urlopen(req, timeout=60).read().decode()
        assert 'lang="ar"' in ar and 'dir="rtl"' in ar                   # report follows the UI language
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


# ---- third review round --------------------------------------------------------------
def test_empty_workflow_chunk_is_not_proof(tmp_path):
    meta = PngInfo()
    meta.add_text("workflow", '{"nodes": []}')
    meta.add_text("invokeai_metadata", '{"model": null}')
    p = str(tmp_path / "empty_wf.png")
    Image.fromarray(S.camera_photo(2, 384, 512)).save(p, pnginfo=meta)
    assert basira.analyze(p).verdict_level < 4


def test_partial_child_failure_keeps_parent_from_looking_clean():
    quiet = Evidence("audio.silence", "Silence", Category.MANIPULATION, 0.02, 0.9, "ok")
    broken = Evidence.failed("audio.enf", "ENF", Category.MANIPULATION, RuntimeError("y"))
    child = finalize(Report(path="x", media_type="audio", info={}, evidences=[quiet, broken]))
    parent_ev = Evidence("video.jump", "Jumps", Category.MANIPULATION, 0.03, 0.6, "ok")
    r = finalize(Report(path="x", media_type="video", info={}, evidences=[parent_ev],
                        children=[child]))
    assert child.verdict_level == 1 and r.verdict_level == 1
    assert "failed" in r.verdict


@pytest.mark.parametrize("n,start", [(17408, 16384), (1024, 0)])
def test_stereo_cancellation_in_tail_and_short_inputs(n, start):
    from basira.audio._dsp import to_mono

    left = np.full(n, 0.1, np.float32)
    right = left.copy()
    right[start:] *= -1
    assert float(np.abs(to_mono(np.stack([left, right], axis=1))[start:]).mean()) > 0.05


def test_lookalike_pair_does_not_hide_a_real_clone(tmp_path):
    """Two independently noisy copies of one texture (many matches, different fine detail)
    next to an exact clone: the clone must still be reported."""
    import cv2

    from basira.image.copymove import analyze_copymove
    from basira.image.load import from_array

    rs = np.random.RandomState(3)
    img = S.scene(768, 640, 4).astype(np.float32)
    tex = cv2.GaussianBlur(rs.rand(256, 256).astype(np.float32) * 255, (0, 0), 3)
    tex = (tex - tex.mean()) * 6 + 128
    img[20:276, 20:276] = tex[..., None] + rs.randn(256, 256, 3) * 6
    img[20:276, 340:596] = tex[..., None] + rs.randn(256, 256, 3) * 6
    img = np.clip(img + rs.randn(*img.shape) * 2, 0, 255).astype(np.uint8)
    img[600:728, 400:528] = img[330:458, 60:188]            # the exact clone
    e = analyze_copymove(from_array(img))
    assert e.score > 0.5, e.details["transforms"]
    assert e.heatmap[620:700, 420:500].mean() > 0.5


def test_analysed_seconds_covers_key_frames():
    from basira.video.temporal import FrameScan

    fs = FrameScan(fps=25, n_total=6, width=4, height=4, stride=3, idx=np.array([0, 3]),
                   thumbs=np.zeros((2, 2, 2), np.float32), diff=np.zeros(2), hist=np.zeros(2),
                   noise=np.zeros(2), pts=np.array([0.0, 0.12]), key_pts={5: 4.0})
    assert fs.time_of(5) == 4.0 and fs.analysed_seconds == 4.0


def test_local_server_rejects_malformed_requests_cleanly(tmp_path):
    """Found by fuzzing the server: malformed lengths, odd paths, non-ASCII tokens and short
    bodies must get a reply, never a dropped connection or a traceback."""
    import socket
    from http.server import ThreadingHTTPServer

    from basira.server import Handler

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    srv.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    srv.token = "secret-token"
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def raw(path, headers, body=b""):
        with socket.create_connection(("127.0.0.1", port), timeout=30) as s:
            head = f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            head += "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
            s.sendall(head.encode("utf-8") + body)
            s.shutdown(socket.SHUT_WR)
            reply = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                reply += chunk
        return int(reply.split(b" ", 2)[1])

    ok = {"X-Basira-Token": "secret-token"}
    try:
        for bad in ("abc", "0x10", "1e3", "١٢٣", "-5"):
            assert raw("/analyze?name=a.jpg", {**ok, "Content-Length": bad}) == 400, bad
        assert raw("/analyzeXYZ", {**ok, "Content-Length": "3"}, b"abc") == 404
        assert raw("/analyze", {"X-Basira-Token": "s\u00e9cret", "Content-Length": "3"},
                   b"abc") == 403
        assert raw("/analyze?name=a.jpg", {**ok, "Content-Length": "1000"}, b"x" * 10) == 400
        for name in ("..", ".", "...%2F.."):
            assert raw(f"/analyze?name={name}", {**ok, "Content-Length": "3"}, b"abc") == 422
    finally:
        srv.shutdown()
        srv.server_close()


def test_ela_is_not_applicable_beyond_the_jpeg_size_limit():
    """PIL cannot write a JPEG wider than 65,500 px; ELA must step aside, not error."""
    import numpy as np

    from basira.image.ela import analyze_ela
    from basira.image.load import ImageData

    d = ImageData(path="wide.png", raw=b"", format="PNG",
                  rgb=np.zeros((64, 65501, 3), np.uint8), pil_info={})
    ev = analyze_ela(d)
    assert ev.confidence == 0 and not ev.error


def test_signature_prefilter_finds_exactly_what_the_plain_regexes_find():
    """scan_signatures skips a pattern when none of its required literals occur. That must
    never change the result: compare it with running every regex over the same bytes."""
    import random
    import re

    from basira.core.utils import SIGNATURES, _anchors, scan_signatures

    assert all(_anchors(p) for p, _, _ in SIGNATURES), "every pattern should get a prefilter"
    samples = [b"Kling AI", b"KlingAI", b"DALL\xb7E 3", b"DALL-E", b"Grok 2", b"Grok-Imagine",
               b"Sampler: Euler a, CFG scale: 7", b"jumb\x00\x01 c2pa", b"LAME3.100", b"LAME3",
               b"Lavf58.76.100", b"GIMP 2.10", b'"class_type" :', b"x.ai", b"Sora", b"Veo 3",
               b"urn:uuid:" + b"0" * 36 + b" c2pa", b"Resemble.ai", b"resemble.ai"]
    rng = random.Random(7)
    for p, _, _ in SIGNATURES:
        samples += _anchors(p)
    for trial in range(40):
        picks = rng.sample(samples, rng.randint(0, 6))
        data = bytearray(rng.getrandbits(8) for _ in range(4096))
        for s in picks:
            at = rng.randrange(len(data))
            data[at:at] = s
        data = bytes(data)
        plain = {label for p, label, _ in SIGNATURES if re.search(p, data)}
        fast = {f["label"] for f in scan_signatures(data)}
        assert fast == plain, (trial, picks)


# ---- fourth review round (Codex, on the specialists' changes) --------------------------
def test_report_axis_has_few_ticks_for_absurd_declared_durations():
    """A 1.5 kB Matroska file declared 10^12 s; the axis builder then tried ~278M ticks."""
    from basira.core.report_html import _ticks

    for d in (0.5, 5, 4000, 30000, 1e6, 1e12, 1e300):
        assert 1 <= len(_ticks(d)) <= 10, d
    assert _ticks(float("inf")) == [] and _ticks(float("nan")) == [] and _ticks(0) == []


def test_csv_output_never_overwrites_a_file_being_scanned(tmp_path, monkeypatch):
    from basira import cli

    path, _ = S.make_authentic(str(tmp_path), 0)
    before = open(path, "rb").read()
    link = tmp_path / "alias.csv"
    os.link(path, link)                                         # hard link to the evidence
    for target in (path, str(link)):
        code = cli.main(["scan", path, "--csv", target, "--no-html", "-q"])
        assert code == 1
        assert open(path, "rb").read() == before


def test_csv_output_survives_undecodable_file_names(tmp_path):
    if sys.platform in ("darwin", "win32"):
        pytest.skip("APFS / NTFS refuse non-UTF-8 file names")
    from basira import cli

    path, _ = S.make_authentic(str(tmp_path), 0)
    odd = os.path.join(os.fsencode(str(tmp_path)), b"bad\xffname.jpg")
    os.rename(path, odd)
    out = tmp_path / "out.csv"
    cli.main(["scan", os.fsdecode(odd), "--csv", str(out), "--no-html", "-q"])
    assert "bad" in out.read_text(encoding="utf-8-sig")


def test_refused_audio_track_is_not_a_clean_video(tmp_path, monkeypatch):
    """A video whose declared sound track is refused by a limit must not come back as
    'No significant traces found' with the track silently left out. (FFmpeg cannot write
    65 channels, so the limit is lowered to 1 and the track is stereo.)"""
    import subprocess

    import synth_video as V
    from basira.core import io as bio
    from basira.core.io import ffmpeg_exe

    silent = V.encode(V.make_frames(60, 0), str(tmp_path / "v.mp4"))
    out = str(tmp_path / "v2.mkv")
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", silent,
                        "-f", "lavfi", "-i", "anoisesrc=d=2:r=8000:a=0.01",
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "pcm_s16le",
                        "-ac", "2", "-shortest", out], capture_output=True)
    assert p.returncode == 0, p.stderr.decode()[:300]
    monkeypatch.setattr(bio, "MAX_AUDIO_CHANNELS", 1)
    r = basira.analyze(out)
    assert r.verdict_level != 0 or any(e.error for e in r.evidences), r.verdict
    assert "could not be analysed" in r.verdict or "failed" in r.verdict, r.verdict


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are POSIX only")
@pytest.mark.parametrize("worker", ["start_child_and_hang", "start_child_and_crash"])
def test_worker_pool_leaves_no_decoder_behind(worker):
    """A worker that times out or dies must take the processes it started (FFmpeg) with it."""
    import subprocess
    import time

    import pool_helpers
    from basira import cli

    items = [{"index": 0, "path": "x", "display": "x", "rel": "x"}]
    events = list(cli.run_pool(items, 1, 3, getattr(pool_helpers, worker), {}))
    assert events[-1][1]["error"]["kind"] in ("timeout", "crashed")
    pattern = f"time.sleep\\({pool_helpers.MARK}\\)"
    for _ in range(20):
        left = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout
        if not left.strip():
            break
        time.sleep(0.25)
    subprocess.run(["pkill", "-f", pattern])
    assert not left.strip(), f"orphaned child process(es): {left.split()}"
