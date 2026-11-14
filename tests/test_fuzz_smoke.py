"""A bounded, deterministic fuzz run plus one regression case per fuzz finding.

The full campaign is ``python tests/fuzz_harness.py --out DIR --cases N``; this file keeps
the suite honest between campaigns. Every test below either reproduces a failure the fuzzer
or the containment experiments found (see SECURITY.md) or proves the harness itself works.
"""
from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
import threading
import time
import zlib

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

import basira
import fuzz_harness as F
from basira.core import io as bio
from basira.image import load as iload
from basira.image import metadata as imeta

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the harness needs POSIX "
                                "process groups and pipes")


# ---- helpers ---------------------------------------------------------------------------
def _png_chunk(typ: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)


def _png(width: int, height: int, idat: bytes) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + \
        _png_chunk(b"IEND", b"")


def _jpeg(arr=None, **kw) -> bytes:
    b = io.BytesIO()
    Image.fromarray(F._texture(1) if arr is None else arr).save(b, format="JPEG", **kw)
    return b.getvalue()


def _refused(path, match: str) -> str:
    """analyze() must raise ValueError whose message names no directory."""
    with pytest.raises(ValueError, match=match) as ei:
        basira.analyze(str(path))
    msg = str(ei.value)
    assert os.path.dirname(str(path)) not in msg and os.sep + "tmp" not in msg, msg
    return msg


class _FifoCanary:
    """A FIFO that records whether anything opened it for reading."""

    def __init__(self, path: str):
        self.path, self.opened, self._stop = path, False, False
        os.mkfifo(path)
        self._t = threading.Thread(target=self._watch, daemon=True)
        self._t.start()

    def _watch(self) -> None:
        while not self._stop:
            try:        # opening a FIFO for writing without blocking succeeds only if a reader exists
                os.close(os.open(self.path, os.O_WRONLY | os.O_NONBLOCK))
                self.opened = True
                time.sleep(0.05)
            except OSError:
                time.sleep(0.01)

    def close(self) -> bool:
        time.sleep(0.15)
        self._stop = True
        self._t.join(timeout=2)
        return self.opened


# ---- the harness itself ----------------------------------------------------------------
def test_cases_are_reproducible_from_seed_and_index():
    corpus = F._image_seeds(full=False)
    a = [F.make_case(corpus, 3, i) for i in range(40)]
    b = [F.make_case(corpus, 3, i) for i in range(40)]
    assert [(n, d) for n, d, _ in a] == [(n, d) for n, d, _ in b]
    assert [d for _, d, _ in a] != [d for _, d, _ in (F.make_case(corpus, 4, i) for i in range(40))]
    assert len({m for _, _, meta in a for m in meta["mutators"]}) >= 8
    assert all(len(d) <= F.MAX_CASE_BYTES for _, d, _ in a)


def test_oracle_flags_what_it_is_meant_to_flag(tmp_path, monkeypatch):
    from basira.core.types import Report

    p = tmp_path / "ok.png"
    Image.fromarray(F._texture(1)).save(p)
    assert F.check_case(str(p))["findings"] == []

    def boom(path, **_):
        raise KeyError("escaped")
    monkeypatch.setattr(basira, "analyze", boom)
    assert [f["kind"] for f in F.check_case(str(p))["findings"]] == ["exception"]

    def leaky(path, **_):
        raise ValueError(f"cannot read {path}")
    monkeypatch.setattr(basira, "analyze", leaky)
    assert [f["kind"] for f in F.check_case(str(p))["findings"]] == ["path_leak"]

    class NanReport(Report):
        def to_dict(self):
            return {"path": self.path, "scores": {"overall": float("nan")}, "evidences": []}

    def nan(path, **_):
        return NanReport(path=path, media_type="image", info={}, verdict="v", verdict_level=1)
    monkeypatch.setattr(basira, "analyze", nan)
    assert "non_finite_json" in [f["kind"] for f in F.check_case(str(p))["findings"]]

    def shells_out(path, **_):
        subprocess.run([sys.executable, "-c", "pass"])
        return Report(path=path, media_type="image", info={}, verdict="v", verdict_level=1)
    monkeypatch.setattr(basira, "analyze", shells_out)
    assert "process" in [f["kind"] for f in F.check_case(str(p))["findings"]]

    def clean_but_unreadable(path, **_):
        return Report(path=path, media_type="image", info={}, verdict_level=0,
                      verdict="No significant traces found")
    bad = tmp_path / "bad.png"
    bad.write_bytes(_png(64, 64, b"not a zlib stream"))
    monkeypatch.setattr(basira, "analyze", clean_but_unreadable)
    assert "clean_verdict_for_undecodable_file" in \
        [f["kind"] for f in F.check_case(str(bad))["findings"]]


def test_bounded_campaign_is_clean_and_the_watchdog_works(tmp_path):
    """120 mutated inputs through worker processes, then a deliberate hang."""
    s = F.run_campaign(str(tmp_path / "fuzz"), n_cases=120, seed=7, jobs=2, timeout=60,
                       fast_corpus=True)
    assert s["cases"] == 120 and set(s["outcomes"]) <= {"report", "rejected"}, s["outcomes"]
    assert s["findings"] == {}, {k: (v["count"], v["detail"], v["example"])
                                 for k, v in s["findings"].items()}
    assert min(len(s["by_kind"].get(k, {})) for k in ("image", "audio", "video")) >= 1
    w = F.Worker(timeout=2.0)
    try:
        rec = w.run(str(tmp_path / "unused"), selftest="hang")
        assert rec["outcome"] == "hang" and w.proc is None      # worker was killed
    finally:
        w.close()


# ---- image loader: every decode failure is a ValueError, never a blank "clean" report ----
@pytest.mark.parametrize("name,body,match", [
    ("magic_only.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 200, "not a readable image"),
    ("garbage.png", b"\x89PNG\r\n\x1a\n" + bytes(range(256)), "not a readable image|no pixel"),
    ("cut_after_header.png", _png(64, 64, zlib.compress(b"\0" * (64 * 193)))[:60], "pixel|readable"),
    ("bad_idat.png", _png(64, 64, b"this is not a zlib stream at all"), "no pixel data"),
    ("declares_65535x65535.png", _png(65535, 65535, zlib.compress(b"\0" * 64)), "limit"),
    ("declares_13000x13000.png", _png(13000, 13000, zlib.compress(b"\0" * 64)), "limit"),
    ("bomb_7700x7700.png", _png(7700, 7700, zlib.compress(b"\0" * 64)), "pixels per byte"),
    ("riff_webp_stub.webp", b"RIFF\x10\x00\x00\x00WEBPVP8 \x04\x00\x00\x00abcd", "readable|decoded"),
    ("tiff_stub.tif", b"II*\x00\x08\x00\x00\x00\x01\x00\x00\x01\x03\x00", "readable|decoded"),
    ("not_media.heic", b"\x00\x00\x00\x18ftypheic" + b"\x00" * 40, "readable"),
])
def test_undecodable_images_are_refused_with_a_clean_error(tmp_path, name, body, match):
    p = tmp_path / name
    p.write_bytes(body)
    _refused(p, match)


def test_failed_metadata_read_does_not_mask_a_failed_decode(tmp_path):
    """Found by fuzzing: getexif() swallowed the decoder error, the second load() then
    returned a blank frame and the report said 'No significant traces found'."""
    good = io.BytesIO()
    Image.fromarray(np.dstack([F._texture(1), F._texture(1)[..., 0]])).save(good, format="PNG")
    raw = bytearray(good.getvalue())
    i = raw.find(b"IDAT")
    raw[i + 40:i + 300] = bytes(260)                       # wreck the zlib stream ...
    n = struct.unpack(">I", raw[i - 4:i])[0]
    raw[i + 4 + n:i + 8 + n] = struct.pack(">I", zlib.crc32(bytes(raw[i:i + 4 + n])) & 0xFFFFFFFF)
    p = tmp_path / "rgba_broken.png"
    p.write_bytes(bytes(raw))
    try:
        r = basira.analyze(str(p))
    except ValueError as e:
        assert "rgba_broken.png" in str(e) and str(tmp_path) not in str(e)
    else:                                                  # partly decoded: must say so
        assert "truncated or damaged" in r.info.get("notes", "")


def test_large_flat_canvas_is_refused_before_decoding(tmp_path, monkeypatch):
    p = tmp_path / "flat.png"
    Image.new("RGB", (5000, 4000), (90, 90, 90)).save(p)          # 20 MP in ~60 kB
    opens = []
    real_open = iload.open_untrusted
    monkeypatch.setattr(iload, "open_untrusted", lambda *a, **k: opens.append(1) or real_open(*a, **k))
    _refused(p, "pixels per byte")
    assert len(opens) == 1                 # the header was read; the pixels never were
    monkeypatch.setattr(iload, "open_untrusted", real_open)
    monkeypatch.setenv("BASIRA_MAX_IMAGE_PIXELS", "1000000")
    small = tmp_path / "small.png"
    Image.fromarray(F._texture(1, 1100, 1000)).save(small)
    _refused(small, "limit is 1 MP")
    assert iload.load_image(str(small), max_pixels=2_000_000).rgb.shape == (1100, 1000, 3)


def test_only_allowlisted_image_decoders_run(tmp_path, monkeypatch):
    """Found by experiment: an EPS file named .jpg made Pillow start Ghostscript."""
    started = []
    real = subprocess.Popen

    def spy(args, *a, **k):
        started.append(args)
        return real(args, *a, **k)
    monkeypatch.setattr(subprocess, "Popen", spy)
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 100 100\n%%EndComments\nshowpage\n%%EOF\n"
    for name, body in (("drawing.jpg", eps), ("layers.png", b"8BPS\x00\x01" + b"\x00" * 60),
                       ("icon.gif", b"\x00\x00\x01\x00\x01\x00" + b"\x10" * 60)):
        p = tmp_path / name
        p.write_bytes(body)
        _refused(p, "not a readable image|text file")
    assert started == []
    assert "EPS" not in iload.ALLOWED_FORMATS and "PSD" not in iload.ALLOWED_FORMATS


def test_xmp_scan_is_linear_in_the_number_of_openers():
    """Found by review of the fuzz corpus: the old lazy regex needed 15 s for 4,000
    unterminated '<x:xmpmeta' openers and minutes for a 1 MB file."""
    raw = _jpeg()[:-2] + b"<x:xmpmeta " * 80_000 + b"\xff\xd9"
    d = iload.ImageData(path="x.jpg", raw=raw, format="JPEG", rgb=np.zeros((8, 8, 3), np.uint8),
                        pil_info={})
    t0 = time.time()
    assert imeta._xmp_text(d) == ""
    assert time.time() - t0 < 5.0
    d.raw = b"....<x:xmpmeta ><x:xmpmeta x='1'>hello</x:xmpmeta>"
    assert imeta._xmp_text(d).endswith("hello</x:xmpmeta>")


def test_oversized_exif_thumbnail_is_not_decoded():
    thumb = bytearray(_jpeg(np.zeros((32, 40, 3), np.uint8), quality=50))
    assert imeta.extract_exif_thumbnail(_jpeg(exif=F._exif_blob(bytes(thumb)))).shape == (32, 40, 3)
    sof = bytes(thumb).find(b"\xff\xc0")
    thumb[sof + 5:sof + 9] = struct.pack(">HH", 3000, 4000)       # header now declares 12 MP
    assert imeta.extract_exif_thumbnail(_jpeg(exif=F._exif_blob(bytes(thumb)))) is None
    assert iload._table_natural([1, 2, 3]) is None                # malformed DQT: no crash


def test_truncated_jpeg_is_not_accused_by_its_own_thumbnail(tmp_path):
    """Measured on truncated seeds: the blank, undecoded part of the frame differed from
    the EXIF thumbnail and the verdict became 'Suspicious' (3 of 72 truncated images)."""
    thumb = _jpeg(np.asarray(Image.fromarray(F._texture(1)).resize((40, 32))), quality=70)
    full = _jpeg(quality=88, exif=F._exif_blob(thumb))
    whole = tmp_path / "whole.jpg"
    whole.write_bytes(full)
    assert basira.analyze(str(whole)).get("image.thumbnail").confidence > 0
    for frac in (0.4, 0.6, 0.8):
        p = tmp_path / f"cut{int(frac * 100)}.jpg"
        p.write_bytes(full[:int(len(full) * frac)])
        d = iload.load_image(str(p))
        assert d.truncated and "truncated" in d.notes[0]
        ev = [e for e in imeta.analyze_metadata(d) if e.detector == "image.thumbnail"]
        assert len(ev) == 1 and ev[0].confidence == 0 and ev[0].score == 0, ev[0].summary


# ---- audio loader ---------------------------------------------------------------------
def _wav(tmp_path, name="a.wav", sr=16000, seconds=0.5) -> bytearray:
    p = tmp_path / name
    sf.write(str(p), (np.random.RandomState(0).randn(int(sr * seconds)) * 0.1).astype(np.float32),
             sr, subtype="PCM_16")
    return bytearray(p.read_bytes())


def test_undecodable_audio_raises_valueerror_without_a_path(tmp_path):
    """Found by fuzzing: RuntimeError escaped, carrying FFmpeg's stderr with the full path."""
    for name, body in (("broken.flac", b"fLaC" + bytes(range(200))),
                       ("broken.ogg", b"OggS" + b"\x00" * 300),
                       ("broken.m4a", b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 64),
                       ("id3_only.mp3", b"ID3\x03\x00\x00\x00\x00\x00\x0a" + b"\x00" * 64)):
        p = tmp_path / name
        p.write_bytes(body)
        assert "0x" not in _refused(p, name)                   # no decoder object addresses


def test_implausible_audio_headers_are_refused(tmp_path):
    """Found by fuzzing: a 41 kB WAV declaring 2,147,483,647 Hz cost 2.1 GB of memory."""
    raw = _wav(tmp_path)
    i = raw.find(b"fmt ")
    for field, value, match in ((slice(i + 12, i + 16), struct.pack("<I", 0x7FFFFFFF), "sample rate"),
                                (slice(i + 12, i + 16), struct.pack("<I", 7), "sample rate"),
                                (slice(i + 10, i + 12), struct.pack("<H", 1024), "channels|decode")):
        bad = bytearray(raw)
        bad[field] = value
        p = tmp_path / "bad.wav"
        p.write_bytes(bytes(bad))
        _refused(p, match)


def test_decoded_audio_is_capped_whatever_the_file_declares(tmp_path, monkeypatch):
    assert bio._frame_budget(192_000, 8, 600.0) * 8 <= bio.MAX_AUDIO_VALUES
    assert bio._frame_budget(48_000, 2, 600.0) == 600 * 48_000          # ordinary files: no change
    assert bio._frame_budget(655_350, 8, None) * 8 <= bio.MAX_AUDIO_VALUES
    monkeypatch.setattr(bio, "MAX_AUDIO_VALUES", 4000)
    p = tmp_path / "long.flac"
    sf.write(str(p), np.zeros((48_000, 2), np.int16), 48_000, format="FLAC")   # silence: tiny file
    data, sr = bio.load_audio(str(p), max_seconds=600.0)
    assert data.shape == (2000, 2) and sr == 48_000
    cmd = []
    monkeypatch.setattr(subprocess, "run", lambda c, **k: cmd.extend(c) or (_ for _ in ()).throw(
        subprocess.TimeoutExpired(c, 1)))
    with pytest.raises(ValueError, match="longer than"):
        bio._load_audio_ffmpeg(str(p), max_seconds=600.0)
    assert cmd[cmd.index("-fs") + 1] == str(4000 * 4) and "-t" in cmd


def test_file_with_no_decodable_samples_is_not_a_clean_report(tmp_path, monkeypatch):
    """Found by fuzzing: a damaged M4A decoded to zero samples and was reported as
    'No significant traces found'. A well-formed empty WAV is still accepted."""
    p = tmp_path / "damaged.m4a"
    p.write_bytes(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 64)
    monkeypatch.setattr(bio, "_load_audio_ffmpeg",
                        lambda path, **k: (np.zeros((0, 1), np.float32), 44100))
    with pytest.raises(bio.MediaDecodeError, match="no audio samples"):
        bio.load_audio(str(p))
    empty = tmp_path / "empty.wav"
    sf.write(str(empty), np.zeros((0,), np.float32), 48000)
    assert bio.load_audio(str(empty))[0].shape[0] == 0


# ---- sniffing and decoder containment --------------------------------------------------
def test_sniff_refuses_what_must_not_reach_a_decoder(tmp_path):
    sdp = ("o=- 0 0 IN IP4 127.0.0.1\ns=x\nc=IN IP4 127.0.0.1\nt=0 0\n"
           "m=video 5004 RTP/AVP 96\na=rtpmap:96 H264/90000\n")
    cases = {"session.mp4": (sdp.encode(), "text file"),          # SDP without its "v=0" line
             "list.ts": (b"#EXTM3U\n#EXT-X-TARGETDURATION:10\n", "playlist"),
             "manifest.mp4": (b"<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\">", "playlist"),
             "script.avi": (b"file 'a.mp4'\nfile 'b.mp4'\n", "playlist"),
             "empty.jpg": (b"", "empty")}
    for name, (body, match) in cases.items():
        p = tmp_path / name
        p.write_bytes(body)
        with pytest.raises(ValueError, match=match):
            bio.sniff(str(p))
    with pytest.raises(ValueError, match="not a regular file"):
        bio.sniff(str(tmp_path))
    fifo = str(tmp_path / "pipe.mp4")
    os.mkfifo(fifo)
    t0 = time.time()
    with pytest.raises(ValueError, match="not a regular file"):   # used to block forever
        basira.analyze(fifo)
    assert time.time() - t0 < 10.0
    amr = tmp_path / "voice.amr"                                  # real media that starts "#!"
    amr.write_bytes(b"#!AMR\n" + bytes([0x3C]) + os.urandom(31))
    assert bio.sniff(str(amr)) == "audio"


def test_ffmpeg_is_invoked_with_whitelists_and_a_file_url(tmp_path, monkeypatch):
    import imageio_ffmpeg

    assert bio.ffmpeg_exe() == imageio_ffmpeg.get_ffmpeg_exe()    # the pinned build, not $PATH
    formats = set(bio._FF_FORMATS.split(","))
    assert not formats & {"hls", "applehttp", "concat", "sdp", "rtsp", "rtp", "sap", "image2",
                          "lavfi", "dash", "tty", "vobsub", "concatf"}
    assert bio._FF_SAFE[:2] == ["-protocol_whitelist", "file"]
    assert "-nostdin" in bio._FF_GLOBAL and \
        int(bio._FF_GLOBAL[bio._FF_GLOBAL.index("-max_alloc") + 1]) <= 1 << 30
    assert bio.decoder_path("pipe:0.mp4") == "file:" + os.path.abspath("pipe:0.mp4")
    opts = dict(o.split(";", 1) for o in os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"].split("|"))
    assert opts["format_whitelist"] == bio._FF_FORMATS and "http" not in opts["protocol_whitelist"]
    # A file whose *name* looks like another protocol is still read as that file.
    raw = _wav(tmp_path, "plain.wav")
    monkeypatch.chdir(tmp_path)
    for name in ("pipe:0.wav", "concat:plain.wav|plain.wav", "-i.wav", "a%03d.wav"):
        with open(name, "wb") as f:
            f.write(bytes(raw))
        data, sr = bio._load_audio_ffmpeg(name)
        assert sr == 16000 and data.shape[0] == 8000, name
    assert "plain.wav" in bio.ffmpeg_probe_text("plain.wav")
    assert str(tmp_path) not in bio.ffmpeg_probe_text(str(tmp_path / "plain.wav"))


@pytest.mark.parametrize("name,body", [
    ("list.m3u8", "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:10.0,\n"
                  "{target}\n#EXT-X-ENDLIST\n"),
    ("script.ffconcat", "ffconcat version 1.0\nfile 'canary.ts'\n"),
])
def test_reference_formats_cannot_make_a_decoder_open_another_file(tmp_path, name, body):
    """Found by experiment: with only the protocol whitelist, an HLS playlist handed to the
    guarded FFmpeg or to OpenCV made it open a second local file. The demuxer whitelist
    now stops both, independently of sniff()."""
    import cv2

    canary = _FifoCanary(str(tmp_path / "canary.ts"))
    p = tmp_path / name
    p.write_text(body.format(target=canary.path))
    try:
        assert "Video:" not in bio.ffmpeg_probe_text(str(p))
        # FFmpeg refuses the playlist: either "no audio" or a decode error is fine; what
        # matters is the canary check below (nothing was opened through it).
        try:
            assert bio.extract_audio_track(str(p), max_seconds=1) is None
        except bio.MediaDecodeError:
            pass
        bio._enforce_opencv_whitelist()
        cap = cv2.VideoCapture(str(p), cv2.CAP_FFMPEG)
        assert not cap.isOpened()
        cap.release()
    finally:
        assert canary.close() is False, "a decoder opened the file the playlist pointed to"
    with pytest.raises(ValueError, match="playlist"):
        basira.analyze(str(p))
