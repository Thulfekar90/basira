"""Basira (بصيرة) — explainable forensics for AI-generated and manipulated media.

>>> import basira
>>> report = basira.analyze("photo.jpg")
>>> report.verdict, report.scores
"""
from __future__ import annotations

import os as _os

# Silence FFmpeg decoder chatter coming through OpenCV (must be set before cv2 opens a file).
_os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

__version__ = "0.1.0"

from .core.types import Category, Evidence, Region, Report, Segment  # noqa: E402


def analyze(path: str, **kwargs) -> Report:
    """Analyse any supported image, audio or video file and return a :class:`Report`."""
    import time

    from .core.io import sniff

    kind = kwargs.pop("media_type", None) or sniff(path)
    t0 = time.time()
    if kind == "image":
        from .image import analyze_image as fn
    elif kind == "audio":
        from .audio import analyze_audio as fn
    elif kind == "video":
        from .video import analyze_video as fn
    else:  # pragma: no cover
        raise ValueError(kind)
    report = fn(path, **kwargs)
    report.elapsed = time.time() - t0
    report.version = __version__
    return report


__all__ = ["analyze", "Report", "Evidence", "Category", "Region", "Segment", "__version__"]
