"""Basira audio engine: explainable signal-processing forensics for recordings.

>>> from basira.audio import analyze_audio
>>> report = analyze_audio("interview.wav")
>>> report.verdict, report.scores

Detectors (each one module, each a pure function returning one ``Evidence``):

=========================  ===============  =========================================
detector                   category         question
=========================  ===============  =========================================
``audio.metadata``         provenance / AI  what do container, tags and encoder say?
``audio.enf``              manipulation     is the mains-hum (ENF) phase continuous?
``audio.splice``           manipulation     does the room / noise floor change? clicks?
``audio.silence``          manipulation     digital silence inside a live recording?
``audio.bandwidth``        AI generation    upsampled from a lower rate (TTS pipeline)?
``audio.bandwidth_shift``  manipulation     does the bandwidth change mid-file?
``audio.synthetic``        AI generation    clean pauses / vocoder lines (weak cues)
=========================  ===============  =========================================
"""
from __future__ import annotations

import os
from typing import Callable, List, Sequence

import numpy as np

from ..core.fusion import finalize
from ..core.types import Category, Evidence, Report, Segment
from ..core.utils import colormap
from ._dsp import to_mono
from .bandwidth import (DETECTOR as BW_ID, DETECTOR_SHIFT as BWS_ID, TITLE as BW_TITLE,
                        TITLE_SHIFT as BWS_TITLE, detect_bandwidth, detect_bandwidth_shift,
                        spectra)
from .enf import DETECTOR as ENF_ID, TITLE as ENF_TITLE, detect_enf
from .metadata import DETECTOR as META_ID, TITLE as META_TITLE, detect_metadata, is_lossy, read_tags
from .silence import DETECTOR as SIL_ID, TITLE as SIL_TITLE, detect_silence
from .splice import DETECTOR as SPL_ID, TITLE as SPL_TITLE, detect_splice
from .synthetic import DETECTOR as SYN_ID, TITLE as SYN_TITLE, detect_synthetic

__all__ = ["analyze_audio", "spectrogram_image", "detect_metadata", "detect_enf", "detect_splice",
           "detect_silence", "detect_bandwidth", "detect_bandwidth_shift", "detect_synthetic"]

PREVIEW_WIDTH = 1200
PREVIEW_HEIGHT = 256
PREVIEW_RANGE_DB = 90.0
SEGMENT_COLOR = np.array([0, 220, 255], dtype=np.float32)   # cyan bands for suspicious time
SEGMENT_ALPHA = 0.32


def _safe(fn: Callable[[], Evidence], det: str, title: str, cat: Category) -> Evidence:
    try:
        return fn()
    except Exception as err:  # a broken detector must never sink the report
        return Evidence.failed(det, title, cat, err)


def _is_video_path(path: str) -> bool:
    from ..core.io import VIDEO_EXT, sniff

    if not path or not os.path.isfile(path):
        return False
    try:
        return sniff(path) == "video"
    except Exception:
        return os.path.splitext(path)[1].lower() in VIDEO_EXT


def spectrogram_image(x: np.ndarray, sr: int, segments: Sequence[Segment] = (),
                      width: int = PREVIEW_WIDTH, height: int = PREVIEW_HEIGHT) -> np.ndarray:
    """RGB uint8 spectrogram (time left->right, frequency bottom->top, linear).

    Each column averages up to 8 frames spread over its time span, so long files
    stay cheap. Suspicious ``segments`` are overlaid as translucent cyan bands.
    """
    import cv2

    n = len(x)
    nfft = 1024 if sr >= 32000 else 512
    if n < nfft:
        x = np.pad(x, (0, nfft - n))
        n = nfft
    n_cols = int(max(1, min(width, (n - nfft) // (nfft // 4) + 1)))
    per = int(max(1, min(8, (n - nfft) // max(1, n_cols * nfft // 2) + 1)))
    starts = np.linspace(0, n - nfft, n_cols * per).astype(np.int64)
    win = np.hanning(nfft).astype(np.float32)
    frames = x[starts[:, None] + np.arange(nfft)[None, :]] * win
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    spec = spec.reshape(n_cols, per, -1).mean(axis=1)
    sdb = 10 * np.log10(spec + 1e-20)
    top = np.percentile(sdb, 99.5)
    img = np.clip((sdb - (top - PREVIEW_RANGE_DB)) / PREVIEW_RANGE_DB, 0, 1).T[::-1]
    img = cv2.resize(img.astype(np.float32), (max(n_cols, 2), height), interpolation=cv2.INTER_AREA)
    if n_cols < width:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    rgb = colormap(img).astype(np.float32)
    dur = len(x) / sr
    W = rgb.shape[1]
    for s in segments:
        c0 = int(np.clip(np.floor(s.start / dur * W), 0, W - 1))
        c1 = int(np.clip(np.ceil(s.end / dur * W), c0 + 1, W))
        rgb[:, c0:c1] = rgb[:, c0:c1] * (1 - SEGMENT_ALPHA) + SEGMENT_COLOR * SEGMENT_ALPHA
        rgb[:, c0] = SEGMENT_COLOR
        rgb[:, c1 - 1] = SEGMENT_COLOR
    return np.clip(rgb, 0, 255).astype(np.uint8)


def analyze_audio(path: str, *, samples: "np.ndarray|None" = None, sr: "int|None" = None,
                  max_seconds: "float|None" = 600.0, label: "str|None" = None, **_) -> Report:
    """Analyse an audio file (or an already-decoded track) and return a fused Report.

    ``samples`` / ``sr``: analyse these instead of decoding ``path`` (shape (n,)
    or (n, ch)); used by the video engine for extracted tracks. When ``path`` is
    then a video container, file-level metadata is left to the video engine.
    ``max_seconds`` bounds the analysed duration (None = whole file).
    """
    from ..core.io import load_audio

    from_samples = samples is not None
    sfinfo, tags, raw_meta = {}, {}, b""
    if from_samples:
        if not sr:
            raise ValueError("analyze_audio(samples=...) also needs sr=")
        data = np.asarray(samples, dtype=np.float32)
        data = data[:, None] if data.ndim == 1 else data
        if data.ndim == 2 and data.shape[0] < data.shape[1] and data.shape[0] <= 8:
            data = data.T
        total_frames = data.shape[0]
        if max_seconds:
            data = data[: int(max_seconds * sr)]
        sr = int(sr)
    else:
        data, sr = load_audio(path, max_seconds=max_seconds)
        total_frames = None
    is_video = from_samples and _is_video_path(path)
    if path and os.path.isfile(path) and not is_video:
        try:
            tags, raw_meta, sfinfo = read_tags(path)
        except Exception:
            tags, raw_meta, sfinfo = {}, b"", {}
        total_frames = sfinfo.get("frames", total_frames)

    x = to_mono(data)
    n = len(x)
    duration = n / sr if sr else 0.0
    lossy = is_lossy(sfinfo, path) if sfinfo or path else False
    peak = float(np.max(np.abs(x))) if n else 0.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if n else 0.0
    info = {
        "duration": round(duration, 3), "sample_rate": int(sr),
        "channels": int(data.shape[1]) if data.ndim == 2 else 1,
        "frames": int(n), "source": "samples" if from_samples else "file",
        "format": sfinfo.get("format"), "format_info": sfinfo.get("format_info"),
        "subtype": sfinfo.get("subtype"), "subtype_info": sfinfo.get("subtype_info"),
        "lossy": bool(lossy),
        "peak_dbfs": round(20 * np.log10(peak + 1e-12), 2),
        "rms_dbfs": round(20 * np.log10(rms + 1e-12), 2),
        "clipped_fraction": float(np.mean(np.abs(x) >= 0.999)) if n else 0.0,
    }
    if label:
        info["label"] = label
    if total_frames and total_frames > n:
        info["truncated"] = True
        info["total_duration"] = round(total_frames / sr, 3)
    if path and os.path.isfile(path) and not from_samples:
        size = os.path.getsize(path)
        info["file_size"] = size
        if lossy and duration > 0:
            info["approx_bitrate_kbps"] = round(size * 8 / 1000 / (total_frames / sr
                                                                  if total_frames else duration), 1)
    encoder_keys = ("software", "encoder", "encoder_settings", "encoded_by", "vendor", "tool",
                    "bext_coding_history")
    hints = [f"{k}: {tags[k]}" for k in encoder_keys if tags.get(k)]
    if hints:
        info["codec_hints"] = hints

    report = Report(path=path, media_type="audio", info=info)
    if n == 0:
        report.evidences.append(Evidence.not_applicable(SPL_ID, SPL_TITLE, Category.MANIPULATION,
                                                        "empty audio"))
        return finalize(report)

    skip = "metadata of the video container is analysed by the video engine" if is_video else None
    report.evidences.append(_safe(
        lambda: detect_metadata(path, skip_reason=skip, tags=tags if sfinfo else None,
                                raw=raw_meta if sfinfo else None, sfinfo=sfinfo if sfinfo else None),
        META_ID, META_TITLE, Category.PROVENANCE))
    report.evidences.append(_safe(lambda: detect_enf(x, sr), ENF_ID, ENF_TITLE,
                                  Category.MANIPULATION))
    report.evidences.append(_safe(lambda: detect_splice(x, sr), SPL_ID, SPL_TITLE,
                                  Category.MANIPULATION))
    report.evidences.append(_safe(lambda: detect_silence(x, sr), SIL_ID, SIL_TITLE,
                                  Category.MANIPULATION))
    spec: List = []

    def _spec():
        if not spec:
            spec.append(spectra(x, sr))
        return spec[0]

    report.evidences.append(_safe(lambda: detect_bandwidth(x, sr, lossy=lossy, spec=_spec()),
                                  BW_ID, BW_TITLE, Category.AI_GENERATION))
    report.evidences.append(_safe(lambda: detect_bandwidth_shift(x, sr, lossy=lossy, spec=_spec()),
                                  BWS_ID, BWS_TITLE, Category.MANIPULATION))
    report.evidences.append(_safe(lambda: detect_synthetic(x, sr, lossy=lossy), SYN_ID, SYN_TITLE,
                                  Category.AI_GENERATION))

    try:
        marked = [s for e in report.evidences if e.error is None and e.score >= 0.5
                  and e.confidence > 0 for s in e.segments if s.score >= 0.5]
        report.preview = spectrogram_image(x, sr, marked)
    except Exception:
        report.preview = None
    return finalize(report)
