"""Deterministic mutation fuzzer for Basira's untrusted-file entry point.

    python tests/fuzz_harness.py --out /some/scratch/dir --cases 4000 --seed 1 --jobs 4

Nothing here needs a dependency beyond what Basira already installs. The seed corpus
(small valid files of every accepted type) is generated locally with Pillow, soundfile
and the bundled FFmpeg; nothing is downloaded. Every mutated input is derived from
``(seed, case index)`` alone, so a case can be regenerated from its number.

The oracle (see :func:`check_case`). For every input ``basira.analyze(path)`` must either

* return a ``Report`` that survives ``to_dict()`` -> strict ``json.dumps`` and
  ``render_html``, or
* raise ``ValueError`` whose message does not contain an absolute path.

Anything else is a finding: another exception type, a hang (each case runs in a worker
process with a watchdog), a peak resident size above the limit, non-finite numbers in
the JSON, a level-0 "No significant traces found" verdict for a file that no decoder
could read, a child process other than FFmpeg with the local-files protocol whitelist,
or any socket activity.

Layout of the output directory::

    corpus/            the generated seed files
    cases/             mutated inputs (deleted after a clean run of the case)
    findings/<sig>/    one reproducer per distinct finding signature + its record
    results.jsonl      one line per case
    summary.json       counts, timings, findings grouped by signature
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import select
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback
import zlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CASE_TIMEOUT_S = 20.0
RSS_LIMIT_BYTES = int(1.5 * (1 << 30))
MAX_CASE_BYTES = 1 << 20            # mutated inputs are kept under 1 MB
INTERESTING_BYTES = (0x00, 0xFF, 0x7F, 0x80)
INTERESTING_U32 = (0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE, 0x00FFFFFF,
                   0x7FFFFFF0, 0x10000, 0xFFFF)
ALL_EXT = (".jpg", ".png", ".gif", ".tif", ".webp", ".bmp", ".wav", ".flac", ".mp3", ".ogg",
           ".m4a", ".mp4", ".mov", ".mkv", ".avi", ".webm", ".ts", ".heic", ".avif", ".opus",
           ".aiff", "", ".txt")

# ============================================================================== #
# Seed corpus
# ============================================================================== #
IMG_W, IMG_H = 176, 144
VID_W, VID_H, VID_FPS, VID_FRAMES = 64, 48, 12, 20
AUDIO_SR, AUDIO_SECONDS = 16000, 1.3


def _texture(seed: int, h: int = IMG_H, w: int = IMG_W):
    """A small photo-like RGB array: smooth structure plus sensor-like noise."""
    import numpy as np
    from PIL import Image

    rs = np.random.RandomState(seed)
    low = (rs.rand(h // 12 + 2, w // 12 + 2, 3) * 255).astype(np.uint8)
    base = np.asarray(Image.fromarray(low).resize((w, h), Image.BICUBIC), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    base += 18 * np.sin(xx / 7.0)[..., None] + 12 * np.cos(yy / 5.0)[..., None]
    base += rs.randn(h, w, 3) * 4
    return np.clip(base, 0, 255).astype(np.uint8)


def _exif_blob(thumb_jpeg: bytes) -> bytes:
    """EXIF (little-endian TIFF) with camera tags, an Exif sub-IFD and an IFD1 thumbnail."""
    strings = {271: b"FuzzCam\x00", 272: b"Model 1\x00", 305: b"firmware 1.0\x00",
               306: b"2024:01:02 03:04:05\x00"}
    sub = {36867: b"2024:01:02 03:04:05\x00"}
    n0, nsub, n1 = len(strings) + 1, len(sub), 3
    off0 = 8
    off_sub = off0 + 2 + n0 * 12 + 4
    off1 = off_sub + 2 + nsub * 12 + 4
    off_data = off1 + 2 + n1 * 12 + 4
    data = b""

    def put(b: bytes) -> int:
        nonlocal data
        o = off_data + len(data)
        data += b + (b"\x00" if len(b) & 1 else b"")
        return o

    ifd0 = struct.pack("<H", n0)
    for tag in sorted(strings):
        ifd0 += struct.pack("<HHII", tag, 2, len(strings[tag]), put(strings[tag]))
    ifd0 += struct.pack("<HHII", 0x8769, 4, 1, off_sub) + struct.pack("<I", off1)
    ifds = struct.pack("<H", nsub)
    for tag in sorted(sub):
        ifds += struct.pack("<HHII", tag, 2, len(sub[tag]), put(sub[tag]))
    ifds += struct.pack("<I", 0)
    toff = put(thumb_jpeg)
    ifd1 = struct.pack("<H", n1) + struct.pack("<HHIHH", 0x103, 3, 1, 6, 0)
    ifd1 += struct.pack("<HHII", 0x201, 4, 1, toff) + struct.pack("<HHII", 0x202, 4, 1, len(thumb_jpeg))
    ifd1 += struct.pack("<I", 0)
    return b"Exif\x00\x00" + b"II*\x00" + struct.pack("<I", off0) + ifd0 + ifds + ifd1 + data


def _save(im, **kw) -> bytes:
    b = io.BytesIO()
    im.save(b, **kw)
    return b.getvalue()


def _image_seeds(full: bool = True) -> Dict[str, bytes]:
    import numpy as np
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    a = _texture(1)
    im = Image.fromarray(a)
    out: Dict[str, bytes] = {}
    thumb = _save(im.resize((40, 32)), format="JPEG", quality=70)
    out["exif_thumb.jpg"] = _save(im, format="JPEG", quality=88, exif=_exif_blob(thumb))
    out["progressive.jpg"] = _save(im, format="JPEG", quality=80, progressive=True)
    out["cmyk.jpg"] = _save(im.convert("CMYK"), format="JPEG", quality=85)
    out["gray_411.jpg"] = _save(im.convert("L"), format="JPEG", quality=60)
    meta = PngInfo()
    meta.add_text("Software", "fuzz corpus")
    meta.add_text("Comment", "compressed text " * 8, zip=True)
    meta.add_itxt("Description", "ünïcode description", lang="en", tkey="Description")
    meta.add_text("parameters", "a cat\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1")
    out["text.png"] = _save(im, format="PNG", pnginfo=meta)
    pal = im.convert("P", palette=Image.ADAPTIVE, colors=32)
    out["palette_trns.png"] = _save(pal, format="PNG", transparency=3)
    out["rgba.png"] = _save(Image.fromarray(np.dstack([a, a[..., 0]])), format="PNG")
    out["gray16.png"] = _save(Image.fromarray((a[..., 0].astype(np.uint16) * 257)), format="PNG")
    frames = [Image.fromarray(np.roll(a, 9 * i, axis=1)).convert("P", palette=Image.ADAPTIVE)
              for i in range(3)]
    out["anim.gif"] = _save(frames[0], format="GIF", save_all=True, append_images=frames[1:],
                            duration=80, loop=0)
    out["plain.tif"] = _save(im, format="TIFF")
    out["lzw.tif"] = _save(im, format="TIFF", compression="tiff_lzw")
    out["deflate_multi.tif"] = _save(im, format="TIFF", compression="tiff_adobe_deflate",
                                     save_all=True, append_images=[im.convert("L")])
    out["lossy.webp"] = _save(im, format="WEBP", quality=75)
    out["lossless.webp"] = _save(im, format="WEBP", lossless=True)
    out["anim.webp"] = _save(Image.fromarray(a), format="WEBP", save_all=True, duration=80,
                             append_images=[Image.fromarray(np.roll(a, 7, axis=0))])
    out["rgb.bmp"] = _save(im, format="BMP")
    if full:
        # Large enough for every pixel detector to run instead of declaring itself
        # "not applicable" (the small seeds mostly exercise the loaders).
        big = Image.fromarray(_texture(2, 480, 640))
        once = _save(big, format="JPEG", quality=92)
        out["photo_640_resaved.jpg"] = _save(Image.open(io.BytesIO(once)), format="JPEG", quality=80)
        out["photo_512.png"] = _save(Image.fromarray(_texture(3, 384, 512)), format="PNG")
    return out


def _ffmpeg(args: List[str], stdin: Optional[bytes] = None) -> None:
    from basira.core.io import ffmpeg_exe

    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("no ffmpeg binary available")
    r = subprocess.run([exe, "-hide_banner", "-nostdin" if stdin is None else "-hide_banner",
                        "-loglevel", "error", "-y", *args], input=stdin,
                       capture_output=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + r.stderr.decode("utf-8", "replace")[-400:])


def _audio_signal(seconds: float = AUDIO_SECONDS):
    import numpy as np

    rs = np.random.RandomState(5)
    n = int(AUDIO_SR * seconds)
    t = np.arange(n) / AUDIO_SR
    x = 0.25 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    x += 0.02 * rs.randn(n) + 0.01 * np.sin(2 * np.pi * 50 * t)
    return x.astype(np.float32)


def _audio_seeds(work: str, full: bool = True) -> Dict[str, bytes]:
    import numpy as np
    import soundfile as sf

    x = _audio_signal()
    out: Dict[str, bytes] = {}
    if full:                     # long enough (9 s) for the ENF / splice / silence detectors
        long_wav = os.path.join(work, "long.wav")
        sf.write(long_wav, _audio_signal(9.0), AUDIO_SR, subtype="PCM_16")
        out["long.wav"] = open(long_wav, "rb").read()
        p = os.path.join(work, "long.mp3")
        _ffmpeg(["-i", long_wav, "-c:a", "libmp3lame", "-b:a", "32k", p])
        out["long.mp3"] = open(p, "rb").read()
    wav = os.path.join(work, "pcm16.wav")
    sf.write(wav, x, AUDIO_SR, subtype="PCM_16")
    raw = open(wav, "rb").read()
    # Add a LIST/INFO chunk by hand so the RIFF walker has something to parse.
    info = b"INFO" + b"ISFT" + struct.pack("<I", 12) + b"fuzz corpus\x00" \
        + b"ICMT" + struct.pack("<I", 8) + b"comment\x00"
    lst = b"LIST" + struct.pack("<I", len(info)) + info
    body = raw[12:] + lst
    out["pcm16.wav"] = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body
    p = os.path.join(work, "float_stereo.wav")
    sf.write(p, np.stack([x, x[::-1]], axis=1), AUDIO_SR, subtype="FLOAT")
    out["float_stereo.wav"] = open(p, "rb").read()
    p = os.path.join(work, "a.flac")
    sf.write(p, x, AUDIO_SR, format="FLAC")
    out["a.flac"] = open(p, "rb").read()
    p = os.path.join(work, "a.aiff")
    sf.write(p, x, AUDIO_SR, format="AIFF")
    out["a.aiff"] = open(p, "rb").read()
    for name, args in (
            ("a.mp3", ["-c:a", "libmp3lame", "-b:a", "32k", "-id3v2_version", "3",
                       "-metadata", "title=fuzz"]),
            ("a.ogg", ["-c:a", "libvorbis", "-q:a", "1", "-metadata", "title=fuzz"]),
            ("a.opus", ["-c:a", "libopus", "-b:a", "16k"]),
            ("a.m4a", ["-c:a", "aac", "-b:a", "32k", "-metadata", "title=fuzz"])):
        p = os.path.join(work, name)
        _ffmpeg(["-i", wav, *args, p])
        out[name] = open(p, "rb").read()
    return out


def _video_frames(n: int = VID_FRAMES) -> bytes:
    import numpy as np

    big = _texture(9, VID_H + 24, VID_W + 2 * n + 8)
    rs = np.random.RandomState(3)
    frames = []
    for i in range(n):
        f = big[12:12 + VID_H, 2 * i:2 * i + VID_W].astype(np.int16)
        f = f + rs.randint(-3, 4, f.shape)
        frames.append(np.clip(f, 0, 255).astype(np.uint8).tobytes())
    return b"".join(frames)


def _video_seeds(work: str, audio_wav: str, fast_only: bool = False) -> Dict[str, bytes]:
    raw = _video_frames()
    vin = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VID_W}x{VID_H}", "-r", str(VID_FPS),
           "-i", "-"]
    ain = ["-i", audio_wav]
    x264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-g", "6", "-bf", "1"]
    recipes: List[Tuple[str, List[str], bool]] = [
        ("h264_aac.mp4", [*x264, "-c:a", "aac", "-b:a", "32k", "-shortest",
                          "-metadata", "title=fuzz"], True),
        ("h264_noaudio.mp4", [*x264, "-movflags", "+faststart"], False),
    ]
    if not fast_only:
        recipes += [
            ("mpeg4_pcm.mov", ["-c:v", "mpeg4", "-q:v", "5", "-c:a", "pcm_s16le", "-shortest"], True),
            ("h264_vorbis.mkv", [*x264, "-c:a", "libvorbis", "-q:a", "1", "-shortest"], True),
            ("mpeg4_mp3.avi", ["-c:v", "mpeg4", "-q:v", "5", "-c:a", "libmp3lame", "-b:a", "32k",
                               "-shortest"], True),
            ("mjpeg_noaudio.avi", ["-c:v", "mjpeg", "-q:v", "5"], False),
            ("vp8_vorbis.webm", ["-c:v", "libvpx", "-b:v", "100k", "-deadline", "realtime",
                                 "-cpu-used", "8", "-c:a", "libvorbis", "-q:a", "1", "-shortest"], True),
            ("vp9_noaudio.webm", ["-c:v", "libvpx-vp9", "-b:v", "100k", "-deadline", "realtime",
                                  "-cpu-used", "8"], False),
            ("frag.mp4", [*x264, "-movflags", "frag_keyframe+empty_moov"], False),
            ("h264.ts", [*x264, "-f", "mpegts"], False),
        ]
    out: Dict[str, bytes] = {}
    for name, args, with_audio in recipes:
        p = os.path.join(work, name)
        _ffmpeg([*vin, *(ain if with_audio else []), *args, p], stdin=raw)
        out[name] = open(p, "rb").read()
    if not fast_only:            # 110 frames + 9 s of sound: temporal and GOP detectors can run
        p = os.path.join(work, "long_h264_aac.mp4")
        _ffmpeg([*vin, "-i", os.path.join(work, "long.wav"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-preset", "veryfast", "-g", "12", "-bf", "0", "-c:a", "aac", "-b:a", "32k",
                 "-shortest", p], stdin=_video_frames(110))
        out["long_h264_aac.mp4"] = open(p, "rb").read()
    return out


def build_corpus(corpus_dir: str, fast_only: bool = False) -> Dict[str, bytes]:
    """Generate (or load) the seed corpus. ``fast_only`` skips the slower video encodes."""
    marker = os.path.join(corpus_dir, ".complete-fast" if fast_only else ".complete")
    full_marker = os.path.join(corpus_dir, ".complete")
    if os.path.exists(marker) or os.path.exists(full_marker):
        names = json.load(open(marker if os.path.exists(marker) else full_marker))
        return {n: open(os.path.join(corpus_dir, n), "rb").read() for n in names}
    os.makedirs(corpus_dir, exist_ok=True)
    work = os.path.join(corpus_dir, "_work")
    os.makedirs(work, exist_ok=True)
    seeds: Dict[str, bytes] = {}
    seeds.update(_image_seeds(full=not fast_only))
    seeds.update(_audio_seeds(work, full=not fast_only))
    seeds.update(_video_seeds(work, os.path.join(work, "pcm16.wav"), fast_only=fast_only))
    shutil.rmtree(work, ignore_errors=True)
    for n, b in seeds.items():
        with open(os.path.join(corpus_dir, n), "wb") as f:
            f.write(b)
    with open(marker, "w") as f:
        json.dump(sorted(seeds), f)
    return {n: seeds[n] for n in sorted(seeds)}


def seed_kind(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext in (".jpg", ".png", ".gif", ".tif", ".webp", ".bmp"):
        return "image"
    if ext in (".wav", ".flac", ".aiff", ".mp3", ".ogg", ".opus", ".m4a"):
        return "audio"
    return "video"


# ============================================================================== #
# Structure discovery: (offset of the length field, width, endianness, start, end)
# ============================================================================== #
Span = Tuple[int, int]                       # [start, end) of one structural unit
LenField = Tuple[int, int, str]              # offset, byte width, "<" or ">"

MP4_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta", b"dinf",
                  b"moof", b"traf", b"mvex"}


def _png_struct(d: bytes) -> Tuple[List[Span], List[LenField]]:
    spans, lens, p = [], [], 8
    while p + 12 <= len(d):
        n = struct.unpack(">I", d[p:p + 4])[0]
        end = min(len(d), p + 12 + n)
        spans.append((p, end))
        lens.append((p, 4, ">"))
        p = end
    return spans, lens


def _jpeg_struct(d: bytes) -> Tuple[List[Span], List[LenField]]:
    spans, lens, p = [], [], 2
    while p + 4 <= len(d) and d[p] == 0xFF:
        m = d[p + 1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            p += 2
            continue
        n = struct.unpack(">H", d[p + 2:p + 4])[0]
        end = min(len(d), p + 2 + n)
        spans.append((p, end))
        lens.append((p + 2, 2, ">"))
        if m == 0xDA:                      # entropy-coded data follows; treat rest as one span
            spans.append((end, len(d)))
            break
        p = end
    return spans, lens


def _riff_struct(d: bytes, start: int = 12, end: Optional[int] = None, depth: int = 0
                 ) -> Tuple[List[Span], List[LenField]]:
    spans: List[Span] = []
    lens: List[LenField] = [(4, 4, "<")] if depth == 0 else []
    end = len(d) if end is None else end
    p = start
    while p + 8 <= end:
        n = struct.unpack("<I", d[p + 4:p + 8])[0]
        e = min(end, p + 8 + n + (n & 1))
        spans.append((p, e))
        lens.append((p + 4, 4, "<"))
        if d[p:p + 4] == b"LIST" and depth < 3 and e - p > 12:
            s2, l2 = _riff_struct(d, p + 12, e, depth + 1)
            spans += s2
            lens += l2
        p = e
    return spans, lens


def _mp4_struct(d: bytes, start: int = 0, end: Optional[int] = None, depth: int = 0
                ) -> Tuple[List[Span], List[LenField]]:
    spans, lens = [], []
    end = len(d) if end is None else end
    p = start
    while p + 8 <= end:
        n, typ = struct.unpack(">I4s", d[p:p + 8])
        hs = 8
        if n == 1 and p + 16 <= end:
            n = struct.unpack(">Q", d[p + 8:p + 16])[0]
            hs = 16
        elif n == 0:
            n = end - p
        if n < hs:
            break
        e = min(end, p + n)
        spans.append((p, e))
        lens.append((p, 4, ">"))
        if typ in MP4_CONTAINERS and depth < 8:
            s2, l2 = _mp4_struct(d, p + hs, e, depth + 1)
            spans += s2
            lens += l2
        elif typ in (b"stsz", b"stco", b"stss", b"stts", b"ctts", b"stsc", b"elst", b"stsd",
                     b"dref") and p + 16 <= e:
            lens.append((p + 12, 4, ">"))          # entry count
            if typ == b"stsz" and p + 20 <= e:
                lens.append((p + 16, 4, ">"))
        p = e
    return spans, lens


def _marker_struct(d: bytes, marker: bytes) -> Tuple[List[Span], List[LenField]]:
    pos, spans = [], []
    i = d.find(marker)
    while i >= 0 and len(pos) < 400:
        pos.append(i)
        i = d.find(marker, i + 1)
    for a, b in zip(pos, pos[1:] + [len(d)]):
        spans.append((a, b))
    return spans, []


def _tiff_struct(d: bytes) -> Tuple[List[Span], List[LenField]]:
    spans, lens = [], []
    if len(d) < 8 or d[:2] not in (b"II", b"MM"):
        return spans, lens
    en = "<" if d[:2] == b"II" else ">"
    lens.append((4, 4, en))
    off, seen = struct.unpack(en + "I", d[4:8])[0], set()
    while 0 < off < len(d) - 2 and off not in seen and len(seen) < 16:
        seen.add(off)
        n = struct.unpack(en + "H", d[off:off + 2])[0]
        lens.append((off, 2, en))
        for k in range(min(n, 64)):
            e = off + 2 + 12 * k
            if e + 12 > len(d):
                break
            spans.append((e, e + 12))
            lens.append((e + 4, 4, en))      # value count
            lens.append((e + 8, 4, en))      # value / offset
        nxt = off + 2 + 12 * n
        if nxt + 4 > len(d):
            break
        lens.append((nxt, 4, en))
        off = struct.unpack(en + "I", d[nxt:nxt + 4])[0]
    return spans, lens


def _flac_struct(d: bytes) -> Tuple[List[Span], List[LenField]]:
    spans, lens, p = [], [], 4
    while p + 4 <= len(d):
        n = int.from_bytes(d[p + 1:p + 4], "big")
        e = min(len(d), p + 4 + n)
        spans.append((p, e))
        lens.append((p + 1, 3, ">"))
        last = d[p] & 0x80
        p = e
        if last:
            spans.append((p, len(d)))
            break
    return spans, lens


def structure(d: bytes) -> Tuple[List[Span], List[LenField]]:
    """Best-effort structural units and length fields of a file, by magic."""
    try:
        if d[:8] == b"\x89PNG\r\n\x1a\n":
            return _png_struct(d)
        if d[:2] == b"\xff\xd8":
            return _jpeg_struct(d)
        if d[:4] in (b"RIFF", b"FORM"):
            if d[:4] == b"FORM":                       # AIFF: big-endian chunks
                spans, lens, p = [], [(4, 4, ">")], 12
                while p + 8 <= len(d):
                    n = struct.unpack(">I", d[p + 4:p + 8])[0]
                    e = min(len(d), p + 8 + n + (n & 1))
                    spans.append((p, e))
                    lens.append((p + 4, 4, ">"))
                    p = e
                return spans, lens
            return _riff_struct(d)
        if d[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"styp"):
            return _mp4_struct(d)
        if d[:4] in (b"II*\x00", b"MM\x00*"):
            return _tiff_struct(d)
        if d[:4] == b"fLaC":
            return _flac_struct(d)
        if d[:4] == b"OggS":
            return _marker_struct(d, b"OggS")
        if d[:3] == b"GIF":
            s1, _ = _marker_struct(d, b"\x21\xf9")     # graphic control extensions
            s2, _ = _marker_struct(d, b"\x2c")          # image descriptors (approximate)
            return s1 + s2[:40], [(6, 2, "<"), (8, 2, "<")]
        if d[:4] == b"\x1aE\xdf\xa3":
            s1, _ = _marker_struct(d, b"\x1f\x43\xb6\x75")   # Matroska clusters
            s2, _ = _marker_struct(d, b"\xa3")                # SimpleBlock ids (approximate)
            return s1 + s2[:60], []
        if d[:1] == b"G" and len(d) > 376 and d[188:189] == b"G":
            return [(i, i + 188) for i in range(0, min(len(d), 188 * 300), 188)], []
        if d[:3] == b"ID3" or (len(d) > 2 and d[0] == 0xFF and d[1] & 0xE0 == 0xE0):
            return _marker_struct(d, b"\xff\xfb")
    except (struct.error, IndexError):
        pass
    return [], []


# ============================================================================== #
# Mutators: (rng, data, corpus) -> bytes
# ============================================================================== #
def _put(d: bytearray, off: int, width: int, endian: str, value: int) -> None:
    value &= (1 << (8 * width)) - 1
    d[off:off + width] = value.to_bytes(width, "little" if endian == "<" else "big")


def m_bitflip(rng, data, corpus):
    d = bytearray(data)
    for _ in range(rng.choice((1, 1, 2, 4, 8, 32))):
        i = rng.randrange(len(d))
        d[i] ^= 1 << rng.randrange(8)
    return bytes(d)


def m_bitflip_header(rng, data, corpus):
    d = bytearray(data)
    lim = min(len(d), rng.choice((16, 64, 256, 1024)))
    for _ in range(rng.choice((1, 2, 3, 6))):
        i = rng.randrange(lim)
        d[i] ^= 1 << rng.randrange(8)
    return bytes(d)


def m_byte_interesting(rng, data, corpus):
    d = bytearray(data)
    lim = len(d) if rng.random() < 0.5 else min(len(d), 2048)
    for _ in range(rng.choice((1, 2, 4, 16))):
        d[rng.randrange(lim)] = rng.choice(INTERESTING_BYTES)
    return bytes(d)


def m_int_interesting(rng, data, corpus):
    d = bytearray(data)
    lim = len(d) if rng.random() < 0.4 else min(len(d), 4096)
    for _ in range(rng.choice((1, 1, 2, 3))):
        w = rng.choice((2, 4, 4, 8))
        if lim <= w:
            break
        _put(d, rng.randrange(lim - w), w, rng.choice("<>"), rng.choice(INTERESTING_U32)
             if w <= 4 else rng.choice((0, 1, 2 ** 63 - 1, 2 ** 64 - 1, 2 ** 32)))
    return bytes(d)


def m_truncate_boundary(rng, data, corpus):
    spans, _ = structure(data)
    if not spans:
        return m_truncate_random(rng, data, corpus)
    a, b = rng.choice(spans)
    cut = rng.choice((a, b, a + 1, a + 4, a + 8, a + 12, max(a, b - 1), max(a, b - 4), (a + b) // 2))
    return data[:max(0, min(len(data), cut))]


def m_truncate_random(rng, data, corpus):
    if rng.random() < 0.3:
        return data[:rng.choice((0, 1, 2, 3, 4, 8, 11, 12, 16, 32, 64))]
    return data[:rng.randrange(len(data) + 1)]


def m_len_tamper(rng, data, corpus):
    _, lens = structure(data)
    if not lens:
        return m_int_interesting(rng, data, corpus)
    d = bytearray(data)
    for _ in range(rng.choice((1, 1, 1, 2))):
        off, w, en = rng.choice(lens)
        if off + w > len(d):
            continue
        cur = int.from_bytes(d[off:off + w], "little" if en == "<" else "big")
        top = (1 << (8 * w)) - 1
        val = rng.choice((0, 1, top, top - 1, top >> 1, (top >> 1) + 1, cur + 1, max(cur - 1, 0),
                          cur * 2, cur + 8, len(d), len(d) + 1, len(d) * 4 + 7, cur ^ 0x80,
                          0x7FFFFFF7 & top))
        _put(d, off, w, en, val)
    return bytes(d)


def m_dup_segment(rng, data, corpus):
    spans, _ = structure(data)
    if not spans:
        a = rng.randrange(len(data))
        spans = [(a, min(len(data), a + rng.choice((1, 16, 256, 4096))))]
    a, b = rng.choice(spans)
    reps = rng.choice((1, 1, 2, 8, 64))
    seg = data[a:b]
    if len(seg) * reps > MAX_CASE_BYTES // 2:
        reps = max(1, (MAX_CASE_BYTES // 2) // max(len(seg), 1))
    return data[:b] + seg * reps + data[b:]


def m_swap_segments(rng, data, corpus):
    spans, _ = structure(data)
    if len(spans) < 2:
        return m_shuffle_blocks(rng, data, corpus)
    (a, b), (c, e) = sorted(rng.sample(spans, 2))
    if b > c:                                   # nested spans: move the inner one to the front
        return data[:a] + data[c:e] + data[a:c] + data[e:]
    return data[:a] + data[c:e] + data[b:c] + data[a:b] + data[e:]


def m_drop_segment(rng, data, corpus):
    spans, _ = structure(data)
    if not spans:
        a = rng.randrange(len(data))
        return data[:a] + data[a + rng.choice((1, 4, 64, 1024)):]
    a, b = rng.choice(spans)
    return data[:a] + data[b:]


def m_shuffle_blocks(rng, data, corpus):
    n = rng.choice((64, 512, 4096))
    blocks = [data[i:i + n] for i in range(0, len(data), n)]
    if len(blocks) < 3:
        return data[::-1]
    i, j = rng.sample(range(1, len(blocks)), 2) if len(blocks) > 2 else (0, 1)
    blocks[i], blocks[j] = blocks[j], blocks[i]
    return b"".join(blocks)


def m_fill(rng, data, corpus):
    d = bytearray(data)
    a = rng.randrange(len(d))
    n = min(len(d) - a, rng.choice((4, 16, 64, 1024, 65536)))
    d[a:a + n] = bytes([rng.choice(INTERESTING_BYTES)]) * n
    return bytes(d)


def m_insert_random(rng, data, corpus):
    a = rng.randrange(len(data) + 1)
    n = rng.choice((1, 4, 64, 4096))
    return data[:a] + bytes(rng.getrandbits(8) for _ in range(n)) + data[a:]


def m_splice(rng, data, corpus):
    other = corpus[rng.choice(sorted(corpus))]
    a = rng.choice((0, 4, 8, 12, 16, 64, rng.randrange(len(data) + 1)))
    b = rng.choice((0, 4, 8, 12, rng.randrange(len(other) + 1)))
    return data[:a] + other[b:]


def m_append_other(rng, data, corpus):
    """Polyglot: a complete second file after (or inside) the first."""
    other = corpus[rng.choice(sorted(corpus))]
    return data + other if rng.random() < 0.6 else other[:64] + data


def m_magic_swap(rng, data, corpus):
    other = corpus[rng.choice(sorted(corpus))]
    n = rng.choice((2, 3, 4, 8, 12, 16))
    return other[:n] + data[n:]


def _png_fix_crc(d: bytearray) -> None:
    p = 8
    while p + 12 <= len(d):
        n = struct.unpack(">I", d[p:p + 4])[0]
        if p + 12 + n > len(d):
            break
        d[p + 8 + n:p + 12 + n] = struct.pack(">I", zlib.crc32(bytes(d[p + 4:p + 8 + n])) & 0xFFFFFFFF)
        p += 12 + n


HUGE = (0, 1, 0x7FFFFFFF, 0xFFFFFFFF, 65535, 65536, 100000, 30000, 0x40000000)


def m_huge_dims(rng, data, corpus):
    """Declare enormous / zero / inconsistent dimensions, sample counts or rates."""
    d = bytearray(data)
    v = rng.choice(HUGE)
    v2 = rng.choice(HUGE + (3, 64))
    if d[:8] == b"\x89PNG\r\n\x1a\n" and len(d) > 33:
        which = rng.randrange(3)
        if which in (0, 2):
            d[16:20] = struct.pack(">I", v)
        if which in (1, 2):
            d[20:24] = struct.pack(">I", v2)
        if rng.random() < 0.8:
            _png_fix_crc(d)
    elif d[:2] == b"\xff\xd8":
        for m in (0xC0, 0xC1, 0xC2):
            i = bytes(d).find(bytes([0xFF, m]))
            if i > 0 and i + 9 < len(d):
                d[i + 5:i + 7] = struct.pack(">H", v & 0xFFFF)
                if rng.random() < 0.5:
                    d[i + 7:i + 9] = struct.pack(">H", v2 & 0xFFFF)
    elif d[:3] == b"GIF" and len(d) > 10:
        d[6:8] = struct.pack("<H", v & 0xFFFF)
        d[8:10] = struct.pack("<H", v2 & 0xFFFF)
        i = bytes(d).find(b"\x2c", 13)
        if i > 0 and i + 9 < len(d) and rng.random() < 0.6:
            d[i + 5:i + 9] = struct.pack("<HH", v & 0xFFFF, v2 & 0xFFFF)
    elif d[:2] == b"BM" and len(d) > 26:
        d[18:22] = struct.pack("<I", v)
        if rng.random() < 0.5:
            d[22:26] = struct.pack("<I", v2)
    elif d[:4] == b"RIFF" and d[8:12] == b"WEBP" and len(d) > 30:
        if d[12:16] == b"VP8X":
            d[24:27] = (v & 0xFFFFFF).to_bytes(3, "little")
            d[27:30] = (v2 & 0xFFFFFF).to_bytes(3, "little")
        elif d[12:16] == b"VP8 ":
            d[26:30] = struct.pack("<HH", v & 0xFFFF, v2 & 0xFFFF)
        elif d[12:16] == b"VP8L":
            d[21:25] = struct.pack("<I", v)
    elif d[:4] in (b"II*\x00", b"MM\x00*"):
        en = "<" if d[:2] == b"II" else ">"
        spans, _ = _tiff_struct(bytes(d))
        for a, _b in spans:
            tag = struct.unpack(en + "H", d[a:a + 2])[0]
            if tag in (256, 257, 278, 279, 273, 258, 277) and rng.random() < 0.6:
                typ = struct.unpack(en + "H", d[a + 2:a + 4])[0]
                if typ == 3:
                    d[a + 8:a + 10] = struct.pack(en + "H", v & 0xFFFF)
                else:
                    d[a + 8:a + 12] = struct.pack(en + "I", v)
    elif d[:4] == b"RIFF" and d[8:12] == b"WAVE":
        i = bytes(d).find(b"fmt ")
        j = bytes(d).find(b"data")
        which = rng.randrange(4)
        if i > 0 and which == 0:
            d[i + 10:i + 12] = struct.pack("<H", v & 0xFFFF)          # channels
        elif i > 0 and which == 1:
            d[i + 12:i + 16] = struct.pack("<I", v)                    # sample rate
        elif i > 0 and which == 2:
            d[i + 20:i + 24] = struct.pack("<HH", v & 0xFFFF, v2 & 0xFFFF)  # align, bits
        elif j > 0:
            d[j + 4:j + 8] = struct.pack("<I", v)                      # data length
            d[4:8] = struct.pack("<I", v2)
    elif d[:4] == b"RIFF" and d[8:12] == b"AVI ":
        i = bytes(d).find(b"avih")
        j = bytes(d).find(b"strf")
        if i > 0 and i + 48 < len(d):
            d[i + 40:i + 48] = struct.pack("<II", v, v2)
            if rng.random() < 0.5:
                d[i + 24:i + 28] = struct.pack("<I", v)                # total frames
        if j > 0 and j + 20 < len(d) and rng.random() < 0.7:
            d[j + 12:j + 20] = struct.pack("<II", v, v2)
    else:
        b = bytes(d)
        for tag, off, fmt in ((b"tkhd", 84, ">II"), (b"stsd", 40, ">HH"), (b"mvhd", 20, ">II"),
                              (b"mdhd", 20, ">II"), (b"stsz", 12, ">II"), (b"stts", 12, ">II")):
            i = b.find(tag)
            if i > 0 and i + off + struct.calcsize(fmt) <= len(d) and rng.random() < 0.5:
                mask = 0xFFFF if "H" in fmt else 0xFFFFFFFF
                d[i + off:i + off + struct.calcsize(fmt)] = struct.pack(fmt, v & mask, v2 & mask)
        if bytes(d) == b:
            return m_int_interesting(rng, data, corpus)
    return bytes(d)


def m_png_deep(rng, data, corpus):
    """Mutate inside a PNG chunk and repair the CRCs so the decoder goes deeper."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return m_len_tamper(rng, data, corpus)
    d = bytearray(m_byte_interesting(rng, data, corpus) if rng.random() < 0.5
                  else m_bitflip(rng, data, corpus))
    d[:8] = data[:8]
    _png_fix_crc(d)
    return bytes(d)


MUTATORS: List[Tuple[str, Callable, int]] = [
    ("bitflip", m_bitflip, 10), ("bitflip_header", m_bitflip_header, 10),
    ("byte_interesting", m_byte_interesting, 8), ("int_interesting", m_int_interesting, 8),
    ("truncate_boundary", m_truncate_boundary, 10), ("truncate_random", m_truncate_random, 5),
    ("len_tamper", m_len_tamper, 14), ("dup_segment", m_dup_segment, 6),
    ("swap_segments", m_swap_segments, 5), ("drop_segment", m_drop_segment, 5),
    ("shuffle_blocks", m_shuffle_blocks, 2), ("fill", m_fill, 4),
    ("insert_random", m_insert_random, 3), ("splice", m_splice, 5),
    ("append_other", m_append_other, 2), ("magic_swap", m_magic_swap, 3),
    ("huge_dims", m_huge_dims, 10), ("png_deep", m_png_deep, 2),
]
_MUT_BY_NAME = {n: f for n, f, _ in MUTATORS}


def make_case(corpus: Dict[str, bytes], seed: int, index: int
              ) -> Tuple[str, bytes, Dict[str, Any]]:
    """Deterministically derive case ``index`` of campaign ``seed``: (file name, bytes, meta)."""
    rng = random.Random(f"basira-fuzz:{seed}:{index}")
    names = sorted(corpus)
    src = names[index % len(names)]
    data = corpus[src]
    muts: List[str] = []
    for _ in range(rng.choice((1, 1, 1, 2, 2, 3))):
        name = rng.choices([m[0] for m in MUTATORS], weights=[m[2] for m in MUTATORS])[0]
        if not data:
            break
        data = _MUT_BY_NAME[name](rng, data, corpus)[:MAX_CASE_BYTES]
        muts.append(name)
    ext = os.path.splitext(src)[1]
    if rng.random() < 0.06:                    # extension / magic mismatch
        ext = rng.choice(ALL_EXT)
        muts.append("ext=" + (ext or "none"))
    stem = os.path.splitext(src)[0]
    fname = f"c{index:06d}_{stem}_{'+'.join(m.replace('=', '') for m in muts)}"[:90] + ext
    return fname, data, {"index": index, "seed_file": src, "kind": seed_kind(src),
                         "mutators": muts, "bytes": len(data)}


# ============================================================================== #
# Oracle (runs inside the worker process)
# ============================================================================== #
def _dec(a) -> str:
    return os.fsdecode(a) if isinstance(a, (bytes, os.PathLike)) else str(a)


class _Audit:
    """Records process creation and socket activity through the interpreter audit hook."""

    def __init__(self) -> None:
        self.events: List[Tuple[str, Any]] = []
        self.active = False
        sys.addaudithook(self._hook)

    def _hook(self, event: str, args) -> None:
        if not self.active:
            return
        if event == "subprocess.Popen":
            exe, argv = args[0], args[1]
            self.events.append(("spawn", [str(exe)] + [_dec(a) for a in (argv or [])]))
        elif event == "os.posix_spawn" and self.events and self.events[-1][0] == "spawn" \
                and self.events[-1][1][1:] == [_dec(a) for a in (args[1] or [])]:
            # Python 3.13+ on Linux starts subprocess.Popen children through posix_spawn:
            # the same child as the Popen event just recorded (and checked), not a second one.
            pass
        elif event in ("os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
                       "os.forkpty", "pty.spawn"):
            self.events.append(("spawn", [event, repr(args)[:200]]))
        elif event.startswith("socket.") and event not in ("socket.__new__", "socket.gethostname"):
            self.events.append(("socket", [event, repr(args)[:200]]))
        elif event in ("urllib.Request", "http.client.connect", "ftplib.connect",
                       "smtplib.connect", "webbrowser.open"):
            self.events.append(("socket", [event, repr(args)[:200]]))


_AUDIT: Optional[_Audit] = None


# Child processes that are not started by Basira and do not depend on the input file:
# NumPy 2.0's ``numpy.testing`` (imported lazily by SciPy) runs ``lscpu`` once at import to
# look for ARM SVE, and imageio-ffmpeg validates its binary once with ``ffmpeg -version``.
BENIGN_SPAWNS = (["lscpu"],)


def _check_spawn(argv: List[str], case_path: str) -> Optional[str]:
    """Return a description of what is wrong with a child-process command line, or None."""
    from basira.core.io import ffmpeg_exe

    exe = ffmpeg_exe()
    if argv[0] in ("os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
                   "pty.spawn"):
        return f"process creation through {argv[0]}: {argv[1]}"
    if argv[1:] in [list(b) for b in BENIGN_SPAWNS]:
        return None
    if os.path.realpath(argv[0]) != os.path.realpath(exe or ""):
        return f"child process is not the FFmpeg binary: {argv[0]!r}"
    args = argv[2:]                                # argv[0] is the executable, argv[1] its copy
    if args == ["-version"]:
        return None
    if "-i" not in args:
        return f"unexpected FFmpeg invocation: {args[:6]!r}"
    if "-protocol_whitelist" not in args or "-format_whitelist" not in args:
        return "FFmpeg started without -protocol_whitelist / -format_whitelist"
    wl = args[args.index("-protocol_whitelist") + 1]
    if not set(wl.split(",")) <= {"file"}:
        return f"FFmpeg protocol whitelist is too wide: {wl!r}"
    fl = set(args[args.index("-format_whitelist") + 1].split(","))
    if fl & {"hls", "concat", "sdp", "rtsp", "rtp", "sap", "image2", "lavfi", "dash", "tty",
             "applehttp", "concatf", "vobsub"}:
        return f"FFmpeg format whitelist admits a reference format: {sorted(fl)!r}"
    if "-nostdin" not in args:
        return "FFmpeg started without -nostdin"
    if max(args.index("-protocol_whitelist"), args.index("-format_whitelist")) > args.index("-i"):
        return "whitelists are given after the input they must apply to"
    inp = args[args.index("-i") + 1]
    if inp != "file:" + os.path.abspath(case_path):
        return f"FFmpeg input is not file:<absolute path of the analysed file>: {inp!r}"
    return None


def _undecodable(path: str, media_type: str) -> Optional[str]:
    """Independent check with the raw decoders: None if *something* decodes, else why not."""
    try:
        if media_type == "image":
            from PIL import Image, ImageFile
            import numpy as np

            try:
                with Image.open(path) as im:
                    im.load()
                    return None
            except Exception as strict_err:
                old = ImageFile.LOAD_TRUNCATED_IMAGES
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                try:
                    with Image.open(path) as im:
                        a = np.asarray(im.convert("RGB"))
                    if a.size and int(a.max()) != int(a.min()):
                        return None
                    return f"strict decode failed ({type(strict_err).__name__}) and the tolerant " \
                           f"decode is a single flat colour"
                except Exception as e2:
                    return f"no decoder could read it ({type(e2).__name__})"
                finally:
                    ImageFile.LOAD_TRUNCATED_IMAGES = old
        if media_type == "audio":
            import soundfile as sf

            try:
                frames = sf.info(path).frames
                if frames == 0:
                    return None            # a well-formed but empty file: nothing to decode
                if len(sf.read(path, frames=64, dtype="float32")[0]):
                    return None
            except Exception:
                pass
            from basira.core.io import _load_audio_ffmpeg

            try:
                return None if _load_audio_ffmpeg(path, max_seconds=5)[0].size else \
                    "soundfile and FFmpeg both decode zero samples"
            except Exception as e:
                return f"soundfile and FFmpeg both fail ({type(e).__name__})"
        if media_type == "video":
            import cv2

            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
            ok = cap.isOpened() and cap.read()[0]
            cap.release()
            return None if ok else "OpenCV cannot decode a single frame"
    except Exception as e:                       # the cross-check itself must never be a finding
        return None if isinstance(e, MemoryError) else None
    return None


def _sig_from_tb(exc: BaseException) -> Tuple[str, str]:
    tb = traceback.extract_tb(exc.__traceback__)
    ours = [f for f in tb if os.sep + "basira" + os.sep in f.filename and "site-packages" not in f.filename]
    last = tb[-1] if tb else None
    inner = ours[-1] if ours else last
    where = f"{os.path.relpath(inner.filename, REPO)}:{inner.name}" if inner else "?"
    deep = f"{os.path.basename(last.filename)}:{last.name}" if last else "?"
    return where, deep


def _walk_numbers(obj: Any, path: str = "") -> Optional[str]:
    import math

    if isinstance(obj, float) and not math.isfinite(obj):
        return path or "<root>"
    if isinstance(obj, dict):
        for k, v in obj.items():
            r = _walk_numbers(v, f"{path}.{k}")
            if r:
                return r
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            r = _walk_numbers(v, f"{path}[{i}]")
            if r:
                return r
    return None


def _rss() -> int:
    """Peak resident size so far, self plus the largest reaped child, in bytes."""
    import resource

    scale = 1 if sys.platform == "darwin" else 1024
    return max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * scale


def check_case(path: str) -> Dict[str, Any]:
    """Run one file through the oracle. Returns a record with a (possibly empty) list of
    ``findings``; each finding has a ``kind`` and a ``sig`` used for de-duplication."""
    global _AUDIT
    import basira
    from basira.core.report_html import render_html

    if _AUDIT is None:
        _AUDIT = _Audit()
    rec: Dict[str, Any] = {"path": path, "outcome": "", "findings": [], "soft": []}
    findings: List[Dict[str, str]] = rec["findings"]
    directory = os.path.dirname(os.path.abspath(path))
    real_dir = os.path.realpath(directory)
    size = os.path.getsize(path)
    rss0 = _rss()
    _AUDIT.events = []
    _AUDIT.active = True
    t0 = time.time()
    report = None
    try:
        report = basira.analyze(path)
        rec["outcome"] = "report"
    except ValueError as e:
        rec["outcome"] = "rejected"
        msg = str(e)
        rec["message"] = msg[:300]
        if directory in msg or real_dir in msg or os.path.abspath(path) in msg:
            findings.append({"kind": "path_leak", "sig": "path_leak:ValueError:" + _sig_from_tb(e)[0],
                             "detail": msg[:300]})
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        rec["outcome"] = "exception"
        where, deep = _sig_from_tb(e)
        findings.append({"kind": "exception", "sig": f"exception:{type(e).__name__}:{where}:{deep}",
                         "detail": f"{type(e).__name__}: {str(e)[:300]}",
                         "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__))[-3000:]})
    finally:
        _AUDIT.active = False
    rec["seconds"] = round(time.time() - t0, 3)

    if report is not None:
        rec["verdict"] = report.verdict
        rec["level"] = report.verdict_level
        try:
            d = report.to_dict()
            bad = _walk_numbers(d)
            if bad:
                findings.append({"kind": "non_finite_json", "sig": "non_finite_json:" +
                                 "".join(c for c in bad if not c.isdigit()), "detail": bad})
            try:
                text = json.dumps(d, allow_nan=False)
            except ValueError as e:
                text = json.dumps(d)
                if not bad:
                    findings.append({"kind": "non_finite_json", "sig": "non_finite_json:dumps",
                                     "detail": str(e)[:200]})
            # The report echoes the path it was given; anything *else* that names the
            # directory (decoder messages, exception texts) is a leak.
            stripped = text.replace(json.dumps(path)[1:-1], "")
            if directory in stripped or real_dir in stripped:
                i = stripped.find(directory if directory in stripped else real_dir)
                findings.append({"kind": "path_leak_in_report", "sig": "path_leak_in_report",
                                 "detail": stripped[max(0, i - 160):i + 120]})
            errs = [e.get("error") for e in d.get("evidences", []) if e.get("error")]
            for c in d.get("children", []):
                errs += [e.get("error") for e in c.get("evidences", []) if e.get("error")]
            rec["soft"] = [str(x)[:160] for x in errs]
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            where, deep = _sig_from_tb(e)
            findings.append({"kind": "to_dict", "sig": f"to_dict:{type(e).__name__}:{where}:{deep}",
                             "detail": f"{type(e).__name__}: {str(e)[:300]}",
                             "traceback": traceback.format_exc()[-3000:]})
        try:
            html = render_html(report)
            if not isinstance(html, str) or "<html" not in html:
                findings.append({"kind": "render_html", "sig": "render_html:not-a-page",
                                 "detail": "render_html did not return an HTML page"})
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            where, deep = _sig_from_tb(e)
            findings.append({"kind": "render_html",
                             "sig": f"render_html:{type(e).__name__}:{where}:{deep}",
                             "detail": f"{type(e).__name__}: {str(e)[:300]}",
                             "traceback": traceback.format_exc()[-3000:]})
        if report.verdict_level == 0:
            why = _undecodable(path, report.media_type)
            if why:
                findings.append({"kind": "clean_verdict_for_undecodable_file",
                                 "sig": f"clean_verdict_for_undecodable_file:{report.media_type}",
                                 "detail": f"verdict {report.verdict!r} but {why}"})

    for kind, payload in _AUDIT.events:
        if kind == "spawn":
            problem = _check_spawn(payload, path)
            if problem:
                findings.append({"kind": "process", "sig": "process:" + problem.split(":")[0],
                                 "detail": problem + " | argv=" + repr(payload)[:400]})
        else:
            findings.append({"kind": "network", "sig": "network:" + payload[0],
                             "detail": " ".join(payload)})
    rec["spawns"] = sum(1 for k, _ in _AUDIT.events if k == "spawn")
    rss1 = _rss()
    rec["rss_mb"] = round(rss1 / (1 << 20), 1)
    rec["rss_delta_mb"] = round((rss1 - rss0) / (1 << 20), 1)
    if rss1 > RSS_LIMIT_BYTES and rss0 <= RSS_LIMIT_BYTES and size <= MAX_CASE_BYTES:
        findings.append({"kind": "memory", "sig": f"memory:{rec.get('outcome')}",
                         "detail": f"peak RSS {rss1 / (1 << 30):.2f} GB for a {size} byte file"})
    if rec["seconds"] > CASE_TIMEOUT_S:
        findings.append({"kind": "slow", "sig": "slow", "detail": f"{rec['seconds']} s"})
    return rec


# ============================================================================== #
# Worker process and the watchdog that drives it
# ============================================================================== #
def _worker_main() -> int:
    """Read one JSON request per line on stdin, answer with one JSON line on a private fd."""
    out = os.fdopen(os.dup(1), "w")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)                      # nothing a library prints can corrupt the protocol
    os.dup2(devnull, 2)
    import warnings

    warnings.simplefilter("ignore")
    import basira  # noqa: F401  (pay the import cost before the first timed case)
    import cv2  # noqa: F401

    out.write(json.dumps({"ready": True, "pid": os.getpid()}) + "\n")
    out.flush()
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("selftest") == "hang":    # lets the test-suite prove the watchdog works
            time.sleep(3600)
        if req.get("selftest") == "crash":
            os._exit(3)
        try:
            rec = check_case(req["path"])
        except BaseException as e:           # the harness itself broke: report, do not hide
            rec = {"path": req["path"], "outcome": "harness_error", "findings": [
                {"kind": "harness_error", "sig": f"harness_error:{type(e).__name__}",
                 "detail": traceback.format_exc()[-2000:]}], "soft": []}
        out.write(json.dumps(rec) + "\n")
        out.flush()
    return 0


class Worker:
    """One analysis process. ``run`` returns the oracle record, or a hang / crash record
    after killing the whole process group (so stray FFmpeg children die too)."""

    def __init__(self, timeout: float = CASE_TIMEOUT_S, recycle_after: int = 150):
        self.timeout, self.recycle_after = timeout, recycle_after
        self.proc: Optional[subprocess.Popen] = None
        self.served = 0
        self.buf = b""

    def _start(self) -> None:
        self.proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0,
                                     start_new_session=True, cwd=REPO)
        self.served, self.buf = 0, b""
        if self._readline(120.0) is None:
            raise RuntimeError("fuzz worker did not start")

    def _readline(self, timeout: float) -> Optional[bytes]:
        deadline = time.time() + timeout
        fd = self.proc.stdout.fileno()
        while b"\n" not in self.buf:
            left = deadline - time.time()
            if left <= 0:
                return None
            r, _, _ = select.select([fd], [], [], min(left, 1.0))
            if r:
                chunk = os.read(fd, 65536)
                if not chunk:
                    return b""                # worker died
                self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line

    def kill(self) -> None:
        if self.proc is not None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=10)
            except Exception:
                pass
            self.proc = None

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                pass
        self.kill()

    def run(self, path: str, selftest: Optional[str] = None) -> Dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None or self.served >= self.recycle_after:
            self.close()
            self._start()
        self.served += 1
        t0 = time.time()
        try:
            self.proc.stdin.write((json.dumps({"path": path, "selftest": selftest}) + "\n").encode())
        except (BrokenPipeError, OSError):
            self.kill()
            return self.run(path, selftest)
        line = self._readline(self.timeout)
        if line is None:
            self.kill()
            return {"path": path, "outcome": "hang", "seconds": round(time.time() - t0, 2),
                    "soft": [], "findings": [{"kind": "hang", "sig": "hang",
                                              "detail": f"no result after {self.timeout:.0f} s"}]}
        if line == b"":
            code = self.proc.poll()
            self.kill()
            return {"path": path, "outcome": "crash", "seconds": round(time.time() - t0, 2),
                    "soft": [], "findings": [{"kind": "crash", "sig": f"crash:exit={code}",
                                              "detail": f"worker process died (exit status {code})"}]}
        rec = json.loads(line)
        if any(f["kind"] == "memory" for f in rec.get("findings", [])) or \
                rec.get("rss_mb", 0) > 1200:
            self.close()                      # peak RSS is monotonic: start the next case fresh
        return rec


def run_campaign(out_dir: str, n_cases: int, seed: int = 1, jobs: int = 2,
                 timeout: float = CASE_TIMEOUT_S, kinds: Optional[Sequence[str]] = None,
                 fast_corpus: bool = False, start: int = 0, keep_cases: bool = False,
                 corpus_dir: Optional[str] = None, progress: bool = False) -> Dict[str, Any]:
    """Run cases ``start .. start + n_cases`` of campaign ``seed`` and return the summary."""
    os.makedirs(out_dir, exist_ok=True)
    corpus = build_corpus(corpus_dir or os.path.join(out_dir, "corpus"), fast_only=fast_corpus)
    if kinds:
        corpus = {n: b for n, b in corpus.items() if seed_kind(n) in kinds}
    case_dir = os.path.join(out_dir, "cases")
    find_dir = os.path.join(out_dir, "findings")
    os.makedirs(case_dir, exist_ok=True)
    lock = threading.Lock()
    indices = list(range(start, start + n_cases))
    cursor = [0]
    records: List[Dict[str, Any]] = []
    t_start = time.time()
    log = open(os.path.join(out_dir, "results.jsonl"), "a")

    def loop() -> None:
        w = Worker(timeout=timeout)
        try:
            while True:
                with lock:
                    if cursor[0] >= len(indices):
                        return
                    idx = indices[cursor[0]]
                    cursor[0] += 1
                fname, data, meta = make_case(corpus, seed, idx)
                path = os.path.join(case_dir, fname)
                with open(path, "wb") as f:
                    f.write(data)
                rec = w.run(path)
                rec.update(meta)
                rec["campaign_seed"] = seed
                with lock:
                    records.append(rec)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    for fd in rec["findings"]:
                        d = os.path.join(find_dir, hashlib.sha1(fd["sig"].encode()).hexdigest()[:12])
                        if not os.path.isdir(d):
                            os.makedirs(d)
                            shutil.copy(path, os.path.join(d, fname))
                            with open(os.path.join(d, "finding.json"), "w") as f:
                                json.dump({**fd, **meta, "file": fname, "campaign_seed": seed}, f, indent=1)
                    if progress and len(records) % 100 == 0:
                        print(f"  {len(records)}/{len(indices)} cases, "
                              f"{time.time() - t_start:.0f} s", file=sys.stderr, flush=True)
                if not keep_cases:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        finally:
            w.close()

    threads = [threading.Thread(target=loop, daemon=True) for _ in range(max(1, jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.close()
    summary = summarize(records)
    summary.update({"seed": seed, "start": start, "cases": len(records), "jobs": jobs,
                    "wall_seconds": round(time.time() - t_start, 1), "seeds_in_corpus": len(corpus)})
    with open(os.path.join(out_dir, f"summary_seed{seed}_{start}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    outcomes: Dict[str, int] = {}
    by_kind: Dict[str, Dict[str, int]] = {}
    findings: Dict[str, Dict[str, Any]] = {}
    soft: Dict[str, int] = {}
    for r in records:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        k = by_kind.setdefault(r.get("kind", "?"), {})
        k[r["outcome"]] = k.get(r["outcome"], 0) + 1
        for fd in r["findings"]:
            g = findings.setdefault(fd["sig"], {"kind": fd["kind"], "count": 0, "detail": fd["detail"],
                                                "example": os.path.basename(r["path"]),
                                                "index": r.get("index"),
                                                "traceback": fd.get("traceback", "")})
            g["count"] += 1
        for s in r.get("soft", []):
            key = s.split("(")[0] + ":" + s[:90]
            soft[key] = soft.get(key, 0) + 1
    secs = sorted(r.get("seconds", 0) for r in records)
    return {"outcomes": outcomes, "by_kind": by_kind, "findings": findings,
            "detector_errors": dict(sorted(soft.items(), key=lambda kv: -kv[1])[:60]),
            "max_rss_mb": max([r.get("rss_mb", 0) for r in records] or [0]),
            "max_rss_delta_mb": max([r.get("rss_delta_mb", 0) for r in records] or [0]),
            "seconds_median": secs[len(secs) // 2] if secs else 0,
            "seconds_p99": secs[int(len(secs) * 0.99)] if secs else 0,
            "seconds_max": secs[-1] if secs else 0}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", help="output directory (NOT inside the repository)")
    ap.add_argument("--cases", type=int, default=500)
    ap.add_argument("--start", type=int, default=0, help="first case index")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=CASE_TIMEOUT_S)
    ap.add_argument("--kinds", help="comma-separated subset of image,audio,video")
    ap.add_argument("--fast-corpus", action="store_true", help="skip the slower video seeds")
    ap.add_argument("--keep-cases", action="store_true")
    ap.add_argument("--one", help="run the oracle on one existing file and print the record")
    args = ap.parse_args(argv)
    if args.worker:
        return _worker_main()
    if args.one:
        w = Worker(timeout=args.timeout)
        try:
            print(json.dumps(w.run(os.path.abspath(args.one)), indent=1))
        finally:
            w.close()
        return 0
    if not args.out:
        ap.error("--out is required")
    if os.path.commonpath([os.path.abspath(args.out), REPO]) == REPO:
        ap.error("--out must be outside the repository")
    s = run_campaign(args.out, args.cases, seed=args.seed, jobs=args.jobs, timeout=args.timeout,
                     kinds=args.kinds.split(",") if args.kinds else None,
                     fast_corpus=args.fast_corpus, start=args.start, keep_cases=args.keep_cases,
                     progress=True)
    print(json.dumps({k: v for k, v in s.items() if k not in ("findings", "detector_errors")}, indent=1))
    print(f"\n{len(s['findings'])} distinct finding signature(s):")
    for sig, g in sorted(s["findings"].items(), key=lambda kv: -kv[1]["count"]):
        print(f"  [{g['count']:5d}x] {sig}\n           {g['detail'][:200]}\n           e.g. {g['example']}")
    if s["detector_errors"]:
        print("\ndetector errors caught inside reports (not oracle findings):")
        for k, n in list(s["detector_errors"].items())[:25]:
            print(f"  [{n:5d}x] {k.split(':', 1)[1]}")
    return 1 if s["findings"] else 0


if __name__ == "__main__":
    sys.exit(main())
