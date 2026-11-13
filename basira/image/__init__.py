"""Image engine: localise edits and detect AI generation in still images."""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from ..core.fusion import finalize
from ..core.types import Category, Evidence, Report
from ..core.utils import thumbnail
from .copymove import analyze_copymove
from .ela import analyze_ela
from .jpeg import analyze_double_quantization
from .load import ImageData, from_array, load_image
from .metadata import analyze_metadata
from .noise import analyze_noise
from .spectral import analyze_cfa, analyze_spectral

PIXEL_DETECTORS = [
    ("image.jpeg_dq", "JPEG double-quantisation map", Category.MANIPULATION, analyze_double_quantization),
    ("image.ela", "Error level analysis", Category.MANIPULATION, analyze_ela),
    ("image.noise", "Sensor-noise consistency", Category.MANIPULATION, analyze_noise),
    ("image.cfa", "Camera demosaicing (CFA) consistency", Category.MANIPULATION, analyze_cfa),
    ("image.copymove", "Copy-move (cloning)", Category.MANIPULATION, analyze_copymove),
    ("image.spectral", "Generator fingerprint", Category.AI_GENERATION, analyze_spectral),
]

# Detectors that are meaningful on decoded video frames (no container-level JPEG data).
FRAME_DETECTORS = {"image.noise", "image.copymove", "image.spectral"}


def _run(det_id: str, title: str, cat: Category, fn: Callable, d: ImageData) -> Evidence:
    try:
        return fn(d)
    except Exception as e:  # a broken detector must never break the report
        return Evidence.failed(det_id, title, cat, e)


def analyze_image(path: str, *, skip: Optional[List[str]] = None, **_) -> Report:
    skip = set(skip or [])
    d = load_image(path)
    h, w = d.rgb.shape[:2]
    info = {"format": d.format, "width": w, "height": h, "jpeg_quality": d.jpeg_quality,
            "jpeg_standard_table": d.quant_is_standard, "subsampling": d.subsampling,
            "has_alpha": d.has_alpha, "bytes": len(d.raw)}
    if d.notes:
        info["notes"] = "; ".join(d.notes)
    report = Report(path=path, media_type="image", info=info)
    try:
        report.evidences.extend(analyze_metadata(d))
    except Exception as e:
        report.evidences.append(Evidence.failed("image.provenance", "Metadata & provenance",
                                                Category.PROVENANCE, e))
    for det_id, title, cat, fn in PIXEL_DETECTORS:
        if det_id not in skip:
            report.evidences.append(_run(det_id, title, cat, fn, d))
    report.preview = thumbnail(d.rgb, 1600)
    return finalize(report, spatial_shape=(h, w))


def analyze_frame(rgb: np.ndarray, name: str = "<frame>", detectors=FRAME_DETECTORS
                  ) -> List[Evidence]:
    """Run the pixel detectors that make sense on a decoded video frame."""
    d = from_array(rgb, name)
    return [_run(i, t, c, fn, d) for i, t, c, fn in PIXEL_DETECTORS if i in detectors]


__all__ = ["analyze_image", "analyze_frame", "load_image", "ImageData"]
