"""Image loading that preserves what forensic detectors need."""
from __future__ import annotations

import io
import os
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageFile

# ---- explicit limits for untrusted files (see SECURITY.md) --------------------------
MAX_IMAGE_PIXELS = 60_000_000        # refused above this, from the header, before decoding
MAX_IMAGE_BYTES = 300 << 20
# A large canvas stored in almost no bytes is a decompression bomb, not a photograph.
# Measured pixels per file byte: flat-colour PNG ~315, flat baseline JPEG 64, bilevel G4
# TIFF > 10000; camera JPEGs 2-10, a photo crushed to JPEG quality 5 about 40. Only
# images above DENSITY_CHECK_PIXELS are tested, so small flat graphics are unaffected.
DENSITY_CHECK_PIXELS = 12_000_000
MAX_PIXELS_PER_BYTE = 48.0


def _limit(env: str, default: float) -> float:
    """A limit, optionally overridden by the operator through the environment
    (``BASIRA_MAX_IMAGE_PIXELS``, ``BASIRA_MAX_PIXELS_PER_BYTE``; 0 disables the latter)."""
    try:
        v = float(os.environ.get(env, ""))
        return v if v >= 0 else default
    except ValueError:
        return default


# Only these Pillow decoders may touch an untrusted file. Pillow would otherwise pick any
# of ~45 formats by content -- including EPS, which starts a Ghostscript process.
ALLOWED_FORMATS = ("JPEG", "MPO", "PNG", "GIF", "BMP", "TIFF", "WEBP", "AVIF", "HEIF")

# ``ImageFile.LOAD_TRUNCATED_IMAGES`` is a process-wide switch. Decoding is serialised so
# that one thread's tolerant retry can never make another thread accept a truncated file
# silently (the local server analyses uploads on several threads).
DECODE_LOCK = threading.RLock()


def open_untrusted(raw: bytes, formats=ALLOWED_FORMATS) -> Image.Image:
    """``Image.open`` restricted to the allowed decoders (header only, nothing decoded).
    Pillow's own bomb warning is silenced because the caller applies explicit limits."""
    known = [f for f in formats if f in Image.OPEN] or list(formats)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        return Image.open(io.BytesIO(raw), formats=known)

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
    # True when the decoder could not read the whole file and the rest of the frame is
    # blank filler. Detectors that compare regions should not read that as an edit.
    truncated: bool = False

    @property
    def is_jpeg(self) -> bool:
        return self.format == "JPEG"

    @property
    def lossless(self) -> bool:
        return self.format in ("PNG", "BMP", "TIFF", "GIF") or \
            (self.format == "WEBP" and self.pil_info.get("lossless", False))


def _table_natural(tbl) -> Optional[np.ndarray]:
    """Pillow (>= 8.3) returns DQT tables already de-zigzagged, in natural order."""
    a = np.asarray(list(tbl), dtype=np.float64)
    return a.reshape(8, 8) if a.size == 64 and np.all(np.isfinite(a)) else None


def estimate_quality(q8: np.ndarray):
    best, best_err = None, 1e18
    for q in range(1, 101):
        err = np.abs(ijg_table(q).reshape(8, 8) - q8).sum()
        if err < best_err:
            best, best_err = q, err
    return best, bool(best_err == 0)


def _why(e: BaseException) -> str:
    """Short reason for a decode failure, without object addresses or paths."""
    msg = " ".join(str(e).split())
    if "<" in msg or os.sep in msg:
        msg = msg.split("<")[0].split(os.sep)[0].strip(" :'\"")
    return f"{type(e).__name__}: {msg[:120]}" if msg else type(e).__name__


def load_image(path: str, max_pixels: Optional[int] = None, max_bytes: int = MAX_IMAGE_BYTES,
               max_pixels_per_byte: Optional[float] = None) -> ImageData:
    """Read and decode an untrusted image. Every refusal or decode failure is a
    ``ValueError`` that names the file but never its directory. The size limits are
    checked on the header, before any pixel is decoded.

    ``max_pixels`` / ``max_pixels_per_byte`` default to ``MAX_IMAGE_PIXELS`` /
    ``MAX_PIXELS_PER_BYTE`` (or their environment overrides); pass ``0`` for the latter
    to analyse a legitimately huge, almost empty canvas such as a blank bilevel scan."""
    name = os.path.basename(path)
    if max_pixels is None:
        max_pixels = int(_limit("BASIRA_MAX_IMAGE_PIXELS", MAX_IMAGE_PIXELS))
    if max_pixels_per_byte is None:
        max_pixels_per_byte = _limit("BASIRA_MAX_PIXELS_PER_BYTE", MAX_PIXELS_PER_BYTE)
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(f"image file is {size >> 20} MB; the limit is {max_bytes >> 20} MB")
    with open(path, "rb") as f:
        raw = f.read()
    notes: List[str] = []
    Image.init()
    try:
        im = open_untrusted(raw)          # header only: nothing is decoded yet
        width, height = int(im.width), int(im.height)
    except Image.DecompressionBombError:
        raise ValueError(f"{name}: the header declares more pixels than any image this tool "
                         f"accepts (limit {max_pixels / 1e6:.0f} MP)") from None
    except Exception as e:
        raise ValueError(f"{name}: not a readable image ({_why(e)})") from None
    if width < 1 or height < 1:
        raise ValueError(f"{name}: image has no pixels ({width}x{height})")
    if width * height > max_pixels:
        raise ValueError(f"image is {width}x{height} "
                         f"({width * height / 1e6:.0f} MP); the limit is "
                         f"{max_pixels / 1e6:.0f} MP")
    if max_pixels_per_byte and width * height > DENSITY_CHECK_PIXELS and \
            width * height > max_pixels_per_byte * max(size, 1):
        raise ValueError(f"{name}: {width}x{height} ({width * height / 1e6:.0f} MP) stored in "
                         f"{size / 1024:.0f} KB is {width * height / max(size, 1):.0f} pixels per "
                         f"byte; images over {DENSITY_CHECK_PIXELS / 1e6:.0f} MP may hold at most "
                         f"{max_pixels_per_byte:.0f} (decompression-bomb guard)")
    fmt = im.format or "UNKNOWN"
    try:
        info = dict(im.info)
    except Exception:
        info = {}
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
        try:
            qt = getattr(im, "quantization", None) or {}
            if 0 in qt:
                quant = _table_natural(qt[0])
                if quant is not None:
                    q_est, q_std = estimate_quality(quant)
            if 1 in qt:
                quant_c = _table_natural(qt[1])
        except Exception:
            quant = quant_c = q_est = q_std = None
        try:
            from PIL import JpegImagePlugin

            ss = JpegImagePlugin.get_sampling(im)
            subsampling = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(ss, str(ss))
        except Exception:
            pass

    has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in info
    palette = im.mode in ("P", "PA")
    mode = im.mode
    truncated = False
    with DECODE_LOCK:
        try:
            # A fresh handle: ``getexif()`` above may already have tried (and failed) to
            # load this one, after which Pillow would hand back a blank frame silently.
            frame = open_untrusted(raw).convert("RGB")
        except MemoryError:
            raise ValueError(f"{name}: not enough memory to decode {width}x{height}") from None
        except Exception as strict_err:
            # A truncated file is still evidence: decode what is there and say so.
            old_flag = ImageFile.LOAD_TRUNCATED_IMAGES
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                im = open_untrusted(raw)
                frame = im.convert("RGB")
            except MemoryError:
                raise ValueError(f"{name}: not enough memory to decode "
                                 f"{width}x{height}") from None
            except Exception as e:
                raise ValueError(f"{name}: image data cannot be decoded ({_why(e)})") from None
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = old_flag
            truncated = True
            notes.append("file is truncated or damaged; the missing part was decoded as blank")
            lo, hi = frame.convert("L").getextrema()
            if lo == hi:
                # Nothing but the canvas came out: there is no content to analyse, and a
                # blank frame must not be reported as "no traces found".
                raise ValueError(f"{name}: no pixel data could be decoded "
                                 f"({_why(strict_err)})") from None
        rgb = np.asarray(frame, dtype=np.uint8).copy()
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
            raise ValueError(f"{name}: decoder returned an unusable frame {rgb.shape}")

        luma = None
        if fmt == "JPEG":
            try:
                if truncated:
                    raise OSError("truncated")
                imy = open_untrusted(raw, formats=("JPEG",))
                imy.draft("YCbCr", imy.size)  # decoder output without colour conversion
                if imy.mode == "YCbCr":
                    luma = np.asarray(imy, dtype=np.float64)[..., 0].copy()
                elif imy.mode == "L":
                    luma = np.asarray(imy, dtype=np.float64).copy()
                if luma is not None and luma.shape != rgb.shape[:2]:
                    luma = None
            except Exception:
                luma = None
            # CMYK / YCCK / RGB-coded JPEGs have no luma plane that matches table 0: leave
            # ``luma_y`` empty so coefficient-domain detectors declare themselves not applicable
            # instead of dividing by the wrong quantisers.
            if luma is None and not truncated:
                notes.append(f"JPEG colour layout ({mode}) has no Y plane; DCT analysis skipped")

    return ImageData(path=path, raw=raw, format=fmt, rgb=rgb, pil_info=info, exif=exif,
                     exif_sub=exif_sub, quant=quant, quant_chroma=quant_c, luma_y=luma,
                     jpeg_quality=q_est, quant_is_standard=q_std, subsampling=subsampling,
                     has_alpha=has_alpha, palette=palette, notes=notes, truncated=truncated)


def from_array(rgb: np.ndarray, name: str = "<frame>") -> ImageData:
    """Wrap a decoded frame (e.g. from video) so pixel-domain detectors can run on it."""
    rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
    return ImageData(path=name, raw=b"", format="RAW", rgb=rgb, pil_info={})
