"""Provenance evidence: generator markers, C2PA, editing software, EXIF consistency,
embedded-thumbnail mismatch and JPEG quantisation fingerprint."""
from __future__ import annotations

import io
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..core.types import Category, Evidence, Region
from ..core.utils import calibrate, iptc_source_types, scan_signatures
from .load import ImageData

EXIF_MAKE, EXIF_MODEL, EXIF_SOFTWARE, EXIF_DATETIME = 271, 272, 305, 306
EXIF_DT_ORIG, EXIF_DT_DIGI = 36867, 36868

# PNG text keys written by popular generators (decisive when present with content).
GEN_TEXT_KEYS = {
    "parameters": "AUTOMATIC1111 / Forge / SD.Next",
    "prompt": "ComfyUI",
    "workflow": "ComfyUI",
    "invokeai_metadata": "InvokeAI",
    "sd-metadata": "InvokeAI (legacy)",
    "Dream": "InvokeAI (legacy)",
    "fooocus_scheme": "Fooocus",
    "generation_data": "Civitai / Tensor.art",
}

# Typical native output resolutions of popular generators (w, h).
GEN_SIZES = {(512, 512), (768, 768), (1024, 1024), (512, 768), (768, 512), (1024, 1792),
             (1792, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832), (1344, 768),
             (768, 1344), (1536, 640), (640, 1536), (1536, 1024), (1024, 1536),
             (2048, 2048), (1456, 816), (816, 1456), (1232, 928), (928, 1232)}

DATE_TOLERANCE_S = 2.0
THUMB_MISMATCH_THRESHOLD = 0.12   # p99 of smoothed |thumbnail - image| (0..1)
THUMB_SOFTNESS = 0.04


def _parse_dt(v: Any) -> Optional[datetime]:
    if not v:
        return None
    s = v.decode("latin-1", "ignore") if isinstance(v, bytes) else str(v)
    s = s.strip("\x00 ").strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


def _xmp_text(d: ImageData) -> str:
    for k in ("xmp", "XML:com.adobe.xmp"):
        v = d.pil_info.get(k)
        if v:
            return v.decode("utf-8", "ignore") if isinstance(v, bytes) else str(v)
    m = re.search(rb"<x:xmpmeta[\s\S]{0,200000}?</x:xmpmeta>", d.raw[: 4 << 20])
    return m.group(0).decode("utf-8", "ignore") if m else ""


def _valid_generator_payload(key: str, txt: str) -> bool:
    """Is this text chunk really what the generator writes, not just a chunk with the
    same name? ("parameters" = "ordinary camera parameters" must not count.)"""
    t = txt.strip()
    if key == "parameters":            # AUTOMATIC1111 / Forge infotext
        return bool(re.search(r"Steps: \d+", t) and
                    re.search(r"(Sampler|CFG scale|Seed|Model hash|Size): ", t))
    if key == "Dream":                 # legacy InvokeAI command line
        return bool(re.search(r"\s-[sSCWHA] ?\d", t))
    if key == "fooocus_scheme":
        return t.lower() in ("fooocus", "a1111", "civitai")
    try:
        js = json.loads(t)
    except Exception:
        return False
    gen_node = re.compile(r"Sampler|CLIPTextEncode|CheckpointLoader|UNETLoader|VAEDecode|"
                          r"EmptyLatent|LoraLoader|ControlNet|SaveImage", re.I)
    if key == "prompt":                # ComfyUI API graph: {"3": {"class_type": ..., "inputs": ...}}
        return (isinstance(js, dict) and len(js) > 0 and
                all(isinstance(v, dict) and "class_type" in v for v in js.values()) and
                any(gen_node.search(str(v.get("class_type", ""))) for v in js.values()))
    if key == "workflow":              # ComfyUI editor graph: {"nodes": [{"type": ...}, ...]}
        nodes = js.get("nodes") if isinstance(js, dict) else None
        return (isinstance(nodes, list) and len(nodes) > 0 and
                all(isinstance(n, dict) for n in nodes) and
                any(gen_node.search(str(n.get("type", ""))) for n in nodes))
    if key in ("invokeai_metadata", "sd-metadata"):
        keys = {"model", "generation_mode", "positive_prompt", "image", "app_id", "seed",
                "steps", "cfg_scale", "scheduler"}
        return isinstance(js, dict) and len(keys & {k for k, v in js.items() if v not in (None, "", [], {})}) >= 2
    if key == "generation_data":
        keys = {"prompt", "models", "seed", "steps", "baseModel", "sampler"}
        return isinstance(js, dict) and len(keys & {k for k, v in js.items() if v not in (None, "", [], {})}) >= 2
    return False


def _generator_from_text(info: Dict[str, Any]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for key, label in GEN_TEXT_KEYS.items():
        v = info.get(key)
        if not v:
            continue
        txt = v.decode("utf-8", "ignore") if isinstance(v, bytes) else str(v)
        if len(txt.strip()) < 8:
            continue
        valid = _valid_generator_payload(key, txt)
        hits.append({"label": label if valid else f"'{key}' text chunk (unrecognised content)",
                     "key": key, "snippet": txt.strip().replace("\n", " ")[:220],
                     "strong": valid})
    # A generator named in the Software field is the writer identifying itself. The same
    # name in a title or description is ordinary prose ("Firefly at night"): a hint only.
    for key in ("Software", "Comment", "Description", "Source", "Title"):
        v = info.get(key)
        if isinstance(v, (str, bytes)):
            txt = v.decode("utf-8", "ignore") if isinstance(v, bytes) else v
            if re.search(r"NovelAI|Stable Diffusion|Midjourney|DALL.?E\b|Adobe Firefly|"
                         r"Imagen\b|FLUX\.1", txt):
                hits.append({"label": txt.strip()[:80], "key": key, "snippet": txt[:220],
                             "strong": key == "Software"})
    return hits


def extract_exif_thumbnail(raw: bytes) -> Optional[np.ndarray]:
    """Return the JPEG thumbnail embedded in the EXIF APP1 segment, if any."""
    if raw[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 < len(raw) and raw[i] == 0xFF:
        marker = raw[i + 1]
        if marker in (0xD9, 0xDA):
            break
        seglen = int.from_bytes(raw[i + 2:i + 4], "big")
        if marker == 0xE1 and raw[i + 4:i + 10] == b"Exif\x00\x00":
            seg = raw[i + 10:i + 2 + seglen]
            j = seg.find(b"\xff\xd8\xff", 8)
            if j > 0:
                k = seg.find(b"\xff\xd9", j)
                try:
                    th = Image.open(io.BytesIO(seg[j:k + 2 if k > 0 else len(seg)]))
                    return np.asarray(th.convert("RGB"))
                except Exception:
                    return None
        i += 2 + seglen
    return None


def analyze_metadata(d: ImageData) -> List[Evidence]:
    out: List[Evidence] = []
    data = d.raw
    sigs = scan_signatures(data[: 4 << 20] + data[-(2 << 20):] if len(data) > (6 << 20) else data)
    xmp = _xmp_text(d)
    src_types = iptc_source_types(data[: 4 << 20] + xmp.encode("utf-8", "ignore"))
    gen_text = _generator_from_text(d.pil_info)

    # ---------------- 1. Generator markers (decisive) ---------------- #
    ai_sigs = [s for s in sigs if s["kind"] == "ai"]
    ai_types = [t for t in src_types if "Algorithmic" in t or "Synthetic" in t or
                t in ("trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia")]
    if gen_text or ai_types or ai_sigs:
        strong = bool(ai_types) or any(g.get("strong") for g in gen_text)
        labels = [g["label"] for g in gen_text] + [f"IPTC:{t}" for t in ai_types] + \
            [s["label"] for s in ai_sigs]
        partial = any("composite" in t.lower() for t in ai_types)
        out.append(Evidence(
            detector="image.generator_metadata", title="Generator metadata",
            category=Category.AI_GENERATION,
            score=0.97 if strong else 0.8, confidence=1.0 if strong else 0.6,
            decisive=strong,
            summary=("The file carries metadata written by an AI generator: "
                     + ", ".join(dict.fromkeys(labels))[:300]
                     + (". The IPTC source type says the image is a composite that "
                        "includes AI-generated parts." if partial else ".")
                     + ("" if strong else " (byte-level match only; verify manually)")),
            details={"generator_text": gen_text, "iptc_source_types": src_types,
                     "signatures": ai_sigs}))
    else:
        out.append(Evidence(
            detector="image.generator_metadata", title="Generator metadata",
            category=Category.AI_GENERATION, score=0.0, confidence=0.3,
            summary="No AI-generator metadata found. Metadata is easily stripped, so its "
                    "absence does not prove anything.",
            details={"iptc_source_types": src_types}))

    # ---------------- 2. Provenance / editing software ---------------- #
    reasons, pscore = [], 0.0
    sw = d.exif.get(EXIF_SOFTWARE) or d.pil_info.get("Software") or ""
    sw = sw.decode("latin-1", "ignore") if isinstance(sw, bytes) else str(sw)
    m_tool = re.search(r"<xmp:CreatorTool>([^<]+)<", xmp) or \
        re.search(r'xmp:CreatorTool="([^"]+)"', xmp)
    creator = m_tool.group(1) if m_tool else ""
    edit_sigs = [s for s in sigs if s["kind"] == "edit"]
    history = re.findall(r'stEvt:action="?([a-z]+)"?[\s\S]{0,300}?stEvt:softwareAgent="?([^"<]+)',
                         xmp)
    if edit_sigs:
        reasons.append("editing software markers: " + ", ".join(s["label"] for s in edit_sigs))
        pscore = max(pscore, 0.75)
    if history:
        acts = sorted({f"{a} by {s.strip()}" for a, s in history})[:6]
        reasons.append("XMP edit history: " + "; ".join(acts))
        pscore = max(pscore, 0.8)
    c2pa = [s for s in sigs if s["kind"] == "c2pa"]
    if c2pa:
        reasons.append("C2PA Content Credentials present (verify the signature with c2patool "
                       "or contentcredentials.org/verify)")
        pscore = max(pscore, 0.55)
    dt = _parse_dt(d.exif.get(EXIF_DATETIME))
    dto = _parse_dt(d.exif_sub.get(EXIF_DT_ORIG))
    if dt and dto and abs((dt - dto).total_seconds()) > DATE_TOLERANCE_S:
        reasons.append(f"modification date {dt} differs from capture date {dto}")
        pscore = max(pscore, 0.7)
    make = str(d.exif.get(EXIF_MAKE, "") or "").strip("\x00 ")
    model = str(d.exif.get(EXIF_MODEL, "") or "").strip("\x00 ")
    if d.is_jpeg and d.quant_is_standard and make:
        reasons.append(f"camera EXIF ({make}) but a standard libjpeg table at Q={d.jpeg_quality}"
                       " — typical of re-saving in software rather than in-camera encoding")
        pscore = max(pscore, 0.6)
    transc = [s["label"] for s in sigs if s["kind"] == "transcode"]
    summary = ("; ".join(reasons) + ".") if reasons else \
        "No editing-software traces in metadata (metadata may have been stripped)."
    out.append(Evidence(
        detector="image.provenance", title="Metadata & provenance",
        category=Category.PROVENANCE, score=pscore, confidence=0.8 if reasons else 0.3,
        summary=summary[:600],
        details={"format": d.format, "software": sw, "creator_tool": creator,
                 "make": make, "model": model,
                 "datetime": str(dt) if dt else None, "datetime_original": str(dto) if dto else None,
                 "jpeg_quality_estimate": d.jpeg_quality, "jpeg_standard_table": d.quant_is_standard,
                 "subsampling": d.subsampling, "c2pa": bool(c2pa), "transcoders": transc,
                 "exif_tags": len(d.exif) + len(d.exif_sub), "xmp_bytes": len(xmp),
                 "signatures": edit_sigs + c2pa}))

    # ---------------- 3. EXIF thumbnail vs image ---------------- #
    try:
        thumb = extract_exif_thumbnail(d.raw) if d.is_jpeg else None
        if thumb is not None and thumb.size > 0:
            out.append(_thumbnail_evidence(d.rgb, thumb))
    except Exception as e:   # one broken sub-check must not discard the other metadata
        out.append(Evidence.failed("image.thumbnail", "EXIF thumbnail consistency",
                                   Category.MANIPULATION, e))

    # ---------------- 4. Weak prior: generator-like file shape ---------------- #
    h, w = d.rgb.shape[:2]
    no_camera = not make and not model and len(d.exif_sub) == 0
    size_hit = (w, h) in GEN_SIZES
    mult64 = (w % 64 == 0 and h % 64 == 0)
    prior = 0.0
    bits = []
    if no_camera:
        prior += 0.3
        bits.append("no camera EXIF")
    if size_hit:
        prior += 0.35
        bits.append(f"{w}x{h} is a common native generator resolution")
    elif mult64:
        prior += 0.15
        bits.append(f"{w}x{h} are multiples of 64")
    if d.format == "PNG" and no_camera:
        prior += 0.1
        bits.append("lossless PNG without camera data")
    out.append(Evidence(
        detector="image.file_prior", title="File-shape prior",
        category=Category.AI_GENERATION, score=min(prior + 0.05, 0.9), confidence=0.2,
        summary=("Weak contextual hint: " + ", ".join(bits) + ".") if bits else
        "Dimensions and EXIF look like an ordinary camera/phone file.",
        details={"width": w, "height": h, "no_camera_exif": no_camera,
                 "generator_resolution": size_hit, "multiple_of_64": mult64}))
    return out


def _thumbnail_evidence(rgb: np.ndarray, thumb: np.ndarray) -> Evidence:
    import cv2

    th, tw = thumb.shape[:2]
    ih, iw = rgb.shape[:2]
    ar_img, ar_th = iw / ih, tw / th
    # Thumbnails are often letter-boxed: crop black bars before comparing.
    g = thumb.mean(axis=2)
    rows = np.where(g.mean(axis=1) > 8)[0]
    cols = np.where(g.mean(axis=0) > 8)[0]
    if rows.size and cols.size:
        thumb = thumb[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        th, tw = thumb.shape[:2]
    small = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    t = thumb.astype(np.float32) / 255
    diff = np.abs(cv2.GaussianBlur(small, (3, 3), 0) - cv2.GaussianBlur(t, (3, 3), 0)).mean(axis=2)
    # A local edit changes a small part of the frame, so judge by the strongest *area*
    # of disagreement (99th percentile of the smoothed difference), not the global mean.
    mad = float(np.percentile(cv2.GaussianBlur(diff, (0, 0), 2.0), 99))
    ar_mismatch = abs(np.log((tw / th) / ar_img)) > 0.08
    score = max(calibrate(mad, THUMB_MISMATCH_THRESHOLD, THUMB_SOFTNESS), 0.7 if ar_mismatch else 0)
    heat = cv2.resize(diff, (iw, ih), interpolation=cv2.INTER_LINEAR)
    heat = np.clip((heat - 0.05) / 0.25, 0, 1)
    regions = []
    if score > 0.5:
        ys, xs = np.where(heat > 0.5)
        if xs.size:
            regions.append(Region(int(xs.min()), int(ys.min()), int(np.ptp(xs) + 1),
                                  int(np.ptp(ys) + 1), score, "differs from EXIF thumbnail"))
    return Evidence(
        detector="image.thumbnail", title="EXIF thumbnail consistency",
        category=Category.MANIPULATION, score=score, confidence=0.75,
        summary=(f"The embedded EXIF thumbnail differs from the image (peak area difference "
                 f"{mad:.3f}{', different aspect ratio' if ar_mismatch else ''}). Editors often "
                 f"forget to regenerate it, so it can show the ORIGINAL content."
                 if score > 0.5 else
                 f"Embedded thumbnail matches the image (peak area difference {mad:.3f})."),
        details={"thumb_size": [int(tw), int(th)], "p99_smoothed_diff": mad,
                 "aspect_thumb": ar_th, "aspect_image": ar_img},
        heatmap=heat if score > 0.5 else None, regions=regions,
        visuals={"exif_thumbnail": thumb})
