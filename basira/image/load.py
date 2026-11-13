"""Image loading that preserves what forensic detectors need."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# Standard IJG (libjpeg) base tables, natural (row-major) order.
IJG_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)

ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])


def ijg_table(q: int) -> np.ndarray:
    q = int(np.clip(q, 1, 100))
    scale = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((IJG_LUMA * scale + 50) / 100), 1, 255)


@dataclass
class ImageData:
    path: str
    raw: bytes
    format: str
    rgb: np.ndarray                      # HxWx3 uint8, stored orientation
    pil_info: Dict[str, Any]
    exif: Dict[int, Any] = field(default_factory=dict)
    exif_sub: Dict[int, Any] = field(default_factory=dict)
    quant: Optional[np.ndarray] = None   # 8x8 luma table, natural order
    quant_chroma: Optional[np.ndarray] = None
    luma_y: Optional[np.ndarray] = None  # decoder Y plane (no colour conversion), JPEG only
    jpeg_quality: Optional[int] = None   # IJG-equivalent estimate
    quant_is_standard: Optional[bool] = None
    subsampling: Optional[str] = None
    has_alpha: bool = False
    palette: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def is_jpeg(self) -> bool:
        return self.format == "JPEG"

    @property
    def lossless(self) -> bool:
        return self.format in ("PNG", "BMP", "TIFF", "GIF") or \
            (self.format == "WEBP" and self.pil_info.get("lossless", False))


def _table_natural(tbl) -> np.ndarray:
    """Pillow returns DQT tables in file (zig-zag) order in some versions and natural
    order in others; detect which by comparing against the IJG shape."""
    t = np.asarray(list(tbl), dtype=np.float64)
    nat = np.empty(64)
    nat[ZIGZAG] = t
    # In natural order the table grows along rows *and* columns; zig-zag-read-as-natural
    # breaks monotonic structure. Pick the interpretation that correlates better with IJG.
    c_as_is = np.corrcoef(t, IJG_LUMA)[0, 1] if t.std() > 0 else 1
    c_nat = np.corrcoef(nat, IJG_LUMA)[0, 1] if nat.std() > 0 else 1
    return (nat if c_nat > c_as_is else t).reshape(8, 8)


def estimate_quality(q8: np.ndarray):
    best, best_err = None, 1e18
    for q in range(1, 101):
        err = np.abs(ijg_table(q).reshape(8, 8) - q8).sum()
        if err < best_err:
            best, best_err = q, err
    return best, bool(best_err == 0)


def load_image(path: str, max_pixels: int = 60_000_000) -> ImageData:
    with open(path, "rb") as f:
        raw = f.read()
    Image.MAX_IMAGE_PIXELS = max_pixels
    notes: List[str] = []
    im = Image.open(io.BytesIO(raw))
    fmt = im.format or "UNKNOWN"
    info = dict(im.info)
    exif, exif_sub = {}, {}
    try:
        ex = im.getexif()
        exif = {int(k): v for k, v in ex.items()}
        try:
            exif_sub = {int(k): v for k, v in ex.get_ifd(0x8769).items()}
        except Exception:
            pass
    except Exception:
        pass

    quant = quant_c = None
    q_est = q_std = None
    subsampling = None
    if fmt == "JPEG":
        qt = getattr(im, "quantization", None) or {}
        if 0 in qt:
            quant = _table_natural(qt[0])
            q_est, q_std = estimate_quality(quant)
        if 1 in qt:
            quant_c = _table_natural(qt[1])
        try:
            from PIL import JpegImagePlugin

            ss = JpegImagePlugin.get_sampling(im)
            subsampling = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(ss, str(ss))
        except Exception:
            pass

    has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in info
    palette = im.mode in ("P", "PA")
    try:
        frame = im.convert("RGB")
    except OSError:
        # A truncated file is still evidence: decode what is there and say so.
        from PIL import ImageFile

        old_flag = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            im = Image.open(io.BytesIO(raw))
            frame = im.convert("RGB")
            notes.append("file is truncated; the missing part was decoded as blank")
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = old_flag
    rgb = np.asarray(frame, dtype=np.uint8).copy()

    luma = None
    if fmt == "JPEG":
        try:
            if notes:
                raise OSError("truncated")
            imy = Image.open(io.BytesIO(raw))
            imy.draft("YCbCr", imy.size)  # decoder output without colour conversion
            if imy.mode == "YCbCr":
                luma = np.asarray(imy, dtype=np.float64)[..., 0].copy()
            elif imy.mode == "L":
                luma = np.asarray(imy, dtype=np.float64).copy()
        except Exception:
            luma = None
        if luma is None:
            luma = np.asarray(Image.fromarray(rgb).convert("YCbCr"), dtype=np.float64)[..., 0]

    return ImageData(path=path, raw=raw, format=fmt, rgb=rgb, pil_info=info, exif=exif,
                     exif_sub=exif_sub, quant=quant, quant_chroma=quant_c, luma_y=luma,
                     jpeg_quality=q_est, quant_is_standard=q_std, subsampling=subsampling,
                     has_alpha=has_alpha, palette=palette, notes=notes)


def from_array(rgb: np.ndarray, name: str = "<frame>") -> ImageData:
    """Wrap a decoded frame (e.g. from video) so pixel-domain detectors can run on it."""
    rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
    return ImageData(path=name, raw=b"", format="RAW", rgb=rgb, pil_info={})
