"""Media type sniffing and decoding (no system dependencies required).

* Images: Pillow (pixels are analysed in *stored* orientation, never EXIF-rotated,
  because rotating would destroy the JPEG 8x8 grid alignment detectors rely on).
* Audio: libsndfile via ``soundfile`` (wav/flac/ogg/mp3), falling back to an
  ``ffmpeg`` binary (system or the one bundled by ``imageio-ffmpeg``).
* Video: OpenCV (bundled FFmpeg) for frames, ``ffmpeg`` for audio tracks/metadata.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".tif", ".tiff",
             ".webp", ".gif", ".heic", ".heif", ".avif"}
AUDIO_EXT = {".wav", ".wave", ".flac", ".mp3", ".ogg", ".oga", ".opus", ".m4a",
             ".aac", ".aif", ".aiff", ".wma", ".amr"}
VIDEO_EXT = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv",
             ".3gp", ".mts", ".m2ts", ".ts", ".mpg", ".mpeg"}


# Text formats that are *references* to other media (playlists, concat scripts, SDP).
# FFmpeg would follow them, including to remote URLs — never hand them to a decoder.
_REFERENCE_MAGIC = (b"#EXTM3U", b"ffconcat", b"[playlist]", b"v=0", b"<?xml", b"<smil",
                    b"<asx", b"#!", b"file ", b"[Reference]", b"RTSPtext")
# Decoders may only read local files.
_FF_SAFE = ["-protocol_whitelist", "file,pipe"]


def _enforce_opencv_whitelist() -> None:
    """Force the local-files-only protocol whitelist into OpenCV's FFmpeg options, keeping
    any other option the user already set (format: ``key;value|key;value``)."""
    cur = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    kept = [o for o in cur.split("|") if o and not o.lower().startswith("protocol_whitelist")]
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(kept + ["protocol_whitelist;file,pipe"])


_enforce_opencv_whitelist()


def sniff(path: str) -> str:
    """Return "image", "audio" or "video" using magic bytes first, extension second."""
    with open(path, "rb") as f:
        head = f.read(64)
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    if any(stripped.lower().startswith(m.lower()) for m in _REFERENCE_MAGIC):
        raise ValueError(f"Unsupported file: {os.path.basename(path)} is a playlist / "
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
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "audio"  # raw MPEG audio frame sync
    raise ValueError(f"Unrecognised media type: {os.path.basename(path)}")


def ffmpeg_exe() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ffmpeg_probe_text(path: str) -> str:
    """``ffmpeg -i`` banner (stream + metadata description), or '' if unavailable."""
    exe = ffmpeg_exe()
    if not exe:
        return ""
    try:
        r = subprocess.run([exe, "-hide_banner", "-nostdin", *_FF_SAFE, "-i", path],
                           capture_output=True, text=True, timeout=60, errors="replace")
        return r.stderr
    except Exception:
        return ""


def load_audio(path: str, max_seconds: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """Decode audio to float32 array of shape (n_samples, n_channels) and sample rate."""
    try:
        import soundfile as sf

        info = sf.info(path)
        frames = -1 if max_seconds is None else int(max_seconds * info.samplerate)
        data, sr = sf.read(path, frames=frames, dtype="float32", always_2d=True)
        if data.size:
            return data, int(sr)
    except Exception:
        pass
    return _load_audio_ffmpeg(path, max_seconds=max_seconds)


def _load_audio_ffmpeg(path: str, sr: Optional[int] = None, max_seconds: Optional[float] = None
                       ) -> Tuple[np.ndarray, int]:
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("Cannot decode this audio format: install ffmpeg or imageio-ffmpeg")
    import soundfile as sf

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "a.wav")
        cmd = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *_FF_SAFE,
               "-i", path, "-vn", "-acodec", "pcm_f32le"]
        if sr:
            cmd += ["-ar", str(sr)]
        if max_seconds:
            cmd += ["-t", str(max_seconds)]
        cmd.append(out)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"ffmpeg could not decode audio: {r.stderr.strip()[:300]}")
        data, rate = sf.read(out, dtype="float32", always_2d=True)
    return data, int(rate)


def extract_audio_track(path: str, max_seconds: Optional[float] = None
                        ) -> Optional[Tuple[np.ndarray, int]]:
    """Audio track of a video, or None if it has none."""
    try:
        data, sr = _load_audio_ffmpeg(path, max_seconds=max_seconds)
        return (data, sr) if data.size else None
    except Exception:
        return None
