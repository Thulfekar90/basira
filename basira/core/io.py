"""Media type sniffing and decoding (no system dependencies required).

* Images: Pillow (pixels are analysed in *stored* orientation, never EXIF-rotated,
  because rotating would destroy the JPEG 8x8 grid alignment detectors rely on).
* Audio: libsndfile via ``soundfile`` (wav/flac/ogg/mp3), falling back to an
  ``ffmpeg`` binary (the one bundled by ``imageio-ffmpeg``, else one on ``PATH``).
* Video: OpenCV (bundled FFmpeg) for frames, ``ffmpeg`` for audio tracks/metadata.

Everything here handles untrusted files. The rules (documented in ``SECURITY.md``):

* a file is never handed to a decoder unless it starts like real media; playlists,
  session descriptions, scripts and any other text file are refused in :func:`sniff`;
* FFmpeg (the binary and the copy inside OpenCV) may only open local files, and only
  with demuxers of real media containers -- so even a reference format that slipped
  past :func:`sniff` could not be followed to another file or to the network;
* the decoder is given ``file:<absolute path>``, never the caller's spelling of the
  name, so no file name can be read as an option, a URL scheme or a pattern;
* every decode is bounded: a time limit, a sample-rate / channel sanity check and a
  cap on the number of decoded samples that is applied *before* memory is allocated.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".tif", ".tiff",
             ".webp", ".gif", ".heic", ".heif", ".avif"}
AUDIO_EXT = {".wav", ".wave", ".flac", ".mp3", ".ogg", ".oga", ".opus", ".m4a",
             ".aac", ".aif", ".aiff", ".wma", ".amr"}
VIDEO_EXT = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv",
             ".3gp", ".mts", ".m2ts", ".ts", ".mpg", ".mpeg"}

# ---- explicit resource limits (see SECURITY.md) ------------------------------------
MAX_AUDIO_VALUES = 64_000_000        # decoded samples x channels kept in memory (256 MB float32)
MAX_AUDIO_CHANNELS = 64
MIN_SAMPLE_RATE, MAX_SAMPLE_RATE = 1_000, 768_000
FFMPEG_PROBE_TIMEOUT_S = 30
FFMPEG_DECODE_TIMEOUT_S = 120

# Text formats that are *references* to other media (playlists, concat scripts, SDP).
# FFmpeg would follow them, including to remote URLs — never hand them to a decoder.
_REFERENCE_MAGIC = (b"#EXTM3U", b"ffconcat", b"[playlist]", b"v=0", b"<?xml", b"<smil",
                    b"<asx", b"#!", b"file ", b"[Reference]", b"RTSPtext", b"<MPD")
# Decoders may only read local files ...
_FF_PROTOCOLS = "file"
# ... and only with demuxers of real media containers. This is a positive list: HLS, DASH,
# concat, SDP / RTSP / SAP, image sequences ("%03d" patterns), subtitle and script formats
# and capture devices are not on it, so FFmpeg refuses them itself whatever the file is
# called. (FFmpeg matches each comma-separated part of a demuxer's name, so "mov" covers
# "mov,mp4,m4a,3gp,3g2,mj2" and "matroska" covers "matroska,webm".)
_FF_FORMATS = "mov,matroska,avi,mpegts,mpeg,flv,asf,wav,w64,aiff,flac,mp3,ogg,aac,amr,caf"
# Largest single block the FFmpeg binary may allocate. A tampered length field in a 19 kB
# AVI made the demuxer allocate over 1 GB; real files stay far below this (an 8K frame is
# about 100 MB, the index of a ten-hour recording about 50 MB).
FFMPEG_MAX_ALLOC = 512 << 20
_FF_SAFE = ["-protocol_whitelist", _FF_PROTOCOLS, "-format_whitelist", _FF_FORMATS]
_FF_GLOBAL = ["-hide_banner", "-nostdin", "-max_alloc", str(FFMPEG_MAX_ALLOC)]


class MediaDecodeError(ValueError):
    """The file could not be decoded (or was refused). The message never contains a
    directory name, so it is safe to show to whoever supplied the file."""


def _enforce_opencv_whitelist() -> None:
    """Force the local-files-only protocol whitelist and the media-containers-only demuxer
    whitelist into OpenCV's FFmpeg options, keeping any other option the user already set
    (format: ``key;value|key;value``).

    Note that this variable is process-wide: once Basira is imported, *every*
    ``cv2.VideoCapture`` in the process is restricted the same way."""
    cur = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    kept = [o for o in cur.split("|") if o and not o.lower().startswith(
        ("protocol_whitelist", "format_whitelist"))]
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(
        kept + [f"format_whitelist;{_FF_FORMATS}", "protocol_whitelist;file,pipe"])


_enforce_opencv_whitelist()

_TEXT_BYTES = frozenset(range(0x20, 0x7F)) | {0x09, 0x0A, 0x0D}


def sniff(path: str) -> str:
    """Return "image", "audio" or "video" using magic bytes first, extension second.

    Raises ``ValueError`` (never naming a directory) for anything that must not reach a
    decoder: non-regular files, empty files, playlists / reference formats and plain text.
    """
    name = os.path.basename(path)
    try:
        st = os.stat(path)
    except OSError as e:
        raise ValueError(f"Cannot read {name}: {e.strerror or type(e).__name__}") from None
    if not stat.S_ISREG(st.st_mode):
        # A FIFO or device would block or never end; a directory is not media.
        raise ValueError(f"Unsupported file: {name} is not a regular file")
    try:
        with open(path, "rb") as f:
            head = f.read(64)
    except OSError as e:
        raise ValueError(f"Cannot read {name}: {e.strerror or type(e).__name__}") from None
    if not head:
        raise ValueError(f"Unsupported file: {name} is empty")
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    is_amr = head.startswith(b"#!AMR")           # the only real media format that is "#!..."
    if not is_amr and any(stripped.lower().startswith(m.lower()) for m in _REFERENCE_MAGIC):
        raise ValueError(f"Unsupported file: {name} is a playlist / "
                         f"reference file, not media")
    if head[:3] == b"\xff\xd8\xff" or head[:8] == b"\x89PNG\r\n\x1a\n" or head[:2] == b"BM" \
            or head[:4] in (b"II*\x00", b"MM\x00*") or head[:6] in (b"GIF87a", b"GIF89a"):
        return "image"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "video"
    if head[:4] in (b"fLaC", b"OggS") or head[:3] == b"ID3" or head[:4] == b"FORM":
        return "audio"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"heic", b"heix", b"mif1", b"msf1", b"avif"):
            return "image"
        if brand in (b"M4A ", b"M4B ", b"M4P "):
            return "audio"
        return "video"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "video"
    if is_amr:
        return "audio"
    # No magic matched. Every media format we would now accept by extension starts with
    # binary data, so a head made only of printable text is a script, a session
    # description or a playlist in disguise (e.g. an SDP file without its "v=0" line).
    if len(head) >= 8 and all(b in _TEXT_BYTES for b in head):
        raise ValueError(f"Unsupported file: {name} is a text file, not media")
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "audio"  # raw MPEG audio frame sync
    raise ValueError(f"Unrecognised media type: {name}")


_FFMPEG_EXE: List[Optional[str]] = []


def ffmpeg_exe() -> Optional[str]:
    """The FFmpeg binary to use: the pinned build bundled by ``imageio-ffmpeg`` (the one
    the containment tests were run against); a binary on ``PATH`` only as a fallback."""
    if _FFMPEG_EXE:
        return _FFMPEG_EXE[0]
    exe: Optional[str] = None
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = None
    if not exe:
        exe = shutil.which("ffmpeg")
    _FFMPEG_EXE.append(exe)
    return exe


def decoder_path(path: str) -> str:
    """How a file must be named to an FFmpeg-based decoder: ``file:`` + absolute path.

    FFmpeg parses its input as a URL. A relative name such as ``pipe:0.mp4`` or
    ``concat:a|b`` would select another protocol; an absolute path cannot. The explicit
    ``file:`` scheme also stops any later parsing of the name."""
    return "file:" + os.path.abspath(path)


def _scrub(text: str, *paths: str) -> str:
    """Remove directory names from decoder output before it is shown or stored."""
    for p in paths:
        if not p:
            continue
        ap = os.path.abspath(p)
        for full in {ap, os.path.realpath(ap)}:
            text = text.replace("file:" + full, os.path.basename(full)).replace(
                full, os.path.basename(full))
        for d in {os.path.dirname(ap), os.path.dirname(os.path.realpath(ap))}:
            if len(d) > 1:
                text = text.replace(d + os.sep, "").replace(d, "")
    return text


def ffmpeg_probe_text(path: str) -> str:
    """``ffmpeg -i`` banner (stream + metadata description), or '' if unavailable.
    Directory names are removed from the text."""
    exe = ffmpeg_exe()
    if not exe:
        return ""
    try:
        r = subprocess.run([exe, *_FF_GLOBAL, *_FF_SAFE, "-i", decoder_path(path)],
                           capture_output=True, text=True, timeout=FFMPEG_PROBE_TIMEOUT_S,
                           errors="replace", stdin=subprocess.DEVNULL)
        return _scrub(r.stderr, path)
    except Exception:
        return ""


def _check_audio_shape(sr: int, channels: int, name: str) -> None:
    if not (MIN_SAMPLE_RATE <= int(sr) <= MAX_SAMPLE_RATE):
        raise MediaDecodeError(f"{name}: sample rate {int(sr)} Hz is outside the supported "
                               f"range {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE} Hz")
    if not (1 <= int(channels) <= MAX_AUDIO_CHANNELS):
        raise MediaDecodeError(f"{name}: {int(channels)} channels; the limit is "
                               f"{MAX_AUDIO_CHANNELS}")


def _frame_budget(sr: int, channels: int, max_seconds: Optional[float]) -> int:
    """Frames to decode: the requested duration, capped so that at most
    ``MAX_AUDIO_VALUES`` samples are ever held in memory."""
    budget = max(1, MAX_AUDIO_VALUES // max(int(channels), 1))
    if max_seconds is not None and max_seconds > 0:
        budget = min(budget, max(1, int(max_seconds * sr)))
    return budget


def load_audio(path: str, max_seconds: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """Decode audio to float32 array of shape (n_samples, n_channels) and sample rate.

    At most ``max_seconds`` and at most ``MAX_AUDIO_VALUES`` samples x channels are decoded
    (whichever is shorter), so a small file that declares hours of audio, hundreds of
    channels or an absurd sample rate cannot exhaust memory. Raises ``ValueError``
    (:class:`MediaDecodeError`) if the file cannot be decoded or is refused."""
    name = os.path.basename(path)
    info = None
    try:
        import soundfile as sf

        info = sf.info(path)
    except Exception:
        info = None
    if info is not None:
        _check_audio_shape(info.samplerate, info.channels, name)
        frames = _frame_budget(info.samplerate, info.channels, max_seconds)
        try:
            data, sr = sf.read(path, frames=frames, dtype="float32", always_2d=True)
        except MemoryError:
            raise MediaDecodeError(f"{name}: not enough memory to decode") from None
        except Exception:
            data, sr = None, 0
        if data is not None and (data.size or info.frames == 0):
            return data, int(sr)       # (a well-formed file with zero samples is returned empty)
    data, sr = _load_audio_ffmpeg(path, max_seconds=max_seconds)
    if not data.size:
        # Neither decoder produced a single sample: this is a damaged file, not silence,
        # and must not be reported as "no traces found".
        raise MediaDecodeError(f"{name}: no audio samples could be decoded")
    return data, sr


def _load_audio_ffmpeg(path: str, sr: Optional[int] = None, max_seconds: Optional[float] = None
                       ) -> Tuple[np.ndarray, int]:
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("Cannot decode this audio format: install ffmpeg or imageio-ffmpeg")
    import soundfile as sf

    name = os.path.basename(path)
    with tempfile.TemporaryDirectory(prefix="basira_") as td:
        out = os.path.join(td, "a.wav")
        cmd = [exe, *_FF_GLOBAL, "-loglevel", "error", "-y", *_FF_SAFE,
               "-i", decoder_path(path), "-vn", "-sn", "-dn", "-acodec", "pcm_f32le"]
        if sr:
            cmd += ["-ar", str(sr)]
        if max_seconds:
            cmd += ["-t", str(max_seconds)]
        # Hard cap on what FFmpeg may write, whatever rate / channel count the file claims.
        cmd += ["-fs", str(MAX_AUDIO_VALUES * 4), "-f", "wav", "file:" + out]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                               timeout=FFMPEG_DECODE_TIMEOUT_S, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise MediaDecodeError(f"{name}: audio decoding took longer than "
                                   f"{FFMPEG_DECODE_TIMEOUT_S} s and was stopped") from None
        if r.returncode != 0 or not os.path.exists(out):
            lines = [ln.strip() for ln in _scrub(r.stderr, path, out).splitlines() if ln.strip()]
            why = re.sub(r"\[[^\]]*@ 0x[0-9a-f]+\]\s*", "", lines[-1] if lines else "no output")
            raise MediaDecodeError(f"{name}: FFmpeg could not decode audio ({why[:200]})")
        try:
            info = sf.info(out)
            _check_audio_shape(info.samplerate, info.channels, name)
            data, rate = sf.read(out, frames=_frame_budget(info.samplerate, info.channels, None),
                                 dtype="float32", always_2d=True)
        except MediaDecodeError:
            raise
        except MemoryError:
            raise MediaDecodeError(f"{name}: not enough memory to decode") from None
        except Exception as e:
            raise MediaDecodeError(f"{name}: decoded audio is unreadable "
                                   f"({type(e).__name__})") from None
    return data, int(rate)


def extract_audio_track(path: str, max_seconds: Optional[float] = None
                        ) -> Optional[Tuple[np.ndarray, int]]:
    """Audio track of a video, or None if it has none.

    Only a confirmed absence of audio gives None. A declared audio stream that cannot be
    decoded, or that is refused by a limit (e.g. too many channels), raises
    MediaDecodeError, so the report says part of the analysis failed instead of quietly
    leaving the sound track out of a clean-looking verdict."""
    try:
        probe = ffmpeg_probe_text(path)
    except Exception:
        probe = ""
    streams = re.findall(r"Stream #\d+:\d+.*?: (Video|Audio|Subtitle|Data|Attachment):", probe)
    if streams and "Audio" not in streams:
        return None                                     # a parsed inventory without audio
    data, sr = _load_audio_ffmpeg(path, max_seconds=max_seconds)
    if not data.size:                                   # declared (or unknown) but empty
        raise MediaDecodeError(f"{os.path.basename(path)}: no audio samples could be decoded")
    return data, sr
