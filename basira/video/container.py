"""Container-level video forensics: metadata, edit lists, double-compression traces.

The MP4 / MOV sample tables are parsed directly (no ffprobe needed): frame sizes
(``stsz``), key frames (``stss``), edit lists (``elst``) and timestamps.
"""
from __future__ import annotations

import os
import re
import struct
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..core.io import ffmpeg_probe_text
from ..core.types import Category, Evidence
from ..core.utils import calibrate, iptc_source_types, read_head_tail, scan_signatures

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta", b"meta", b"ilst"}
MIN_REPEATS = 5             # a periodic spike must repeat at least this often
PERIOD_Z_THRESHOLD = 6.0     # clean encodes measured <= 3.0; double encodes without B-frames >= 8
PERIOD_Z_SOFTNESS = 1.0
# INFORMATIONAL weight. With these confidences the detector alone can never move the verdict
# (provenance needs >= 0.5 for "Inconclusive"): a re-encode is what every shared video has
# been through, the same fact that ``video.provenance`` reports at score 0.4 for a
# transcoder tag. Measured on real footage (benchmarks/video_robustness.py, 585 files of
# 6-8 s): it fired on 9 of 261 clean / natural files, first-generation encodes among them
# (periodic content makes frame sizes periodic too). On re-encoded clean clips it was
# either not applicable (one key frame in 8 s) or blind by construction (old GOP a divisor
# of the new one); the few remaining cases were missed. Clips that short cannot settle
# whether it works on real footage, so it does not get a vote.
DOUBLE_CONF, DOUBLE_CONF_BFRAMES = 0.3, 0.15


def _boxes(buf: bytes, start: int, end: int) -> Iterator[Tuple[bytes, int, int]]:
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[pos:pos + 8])
        hs = 8
        if size == 1 and pos + 16 <= end:
            size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
            hs = 16
        elif size == 0:
            size = end - pos
        if size < hs or pos + size > end:
            break
        yield typ, pos + hs, pos + size
        pos += size


def _find_moov(path: str) -> Optional[bytes]:
    """Return the raw ``moov`` box payload by walking top-level boxes on disk."""
    import os

    size = os.path.getsize(path)
    with open(path, "rb") as f:
        pos = 0
        while pos + 8 <= size:
            f.seek(pos)
            hdr = f.read(16)
            if len(hdr) < 8:
                return None
            bsz, typ = struct.unpack(">I4s", hdr[:8])
            hs = 8
            if bsz == 1:
                bsz = struct.unpack(">Q", hdr[8:16])[0]
                hs = 16
            elif bsz == 0:
                bsz = size - pos
            if bsz < hs:
                return None
            if typ == b"moov":
                if bsz > 256 << 20:
                    return None
                f.seek(pos + hs)
                return f.read(bsz - hs)
            pos += bsz
    return None


MEDIA_BOXES = {b"mdat", b"free", b"skip", b"wide", b"idat"}
TABLE_BOXES = {b"stts", b"ctts", b"stsc", b"stsz", b"stz2", b"stco", b"co64", b"stss", b"sdtp",
               b"sbgp", b"sgpd", b"subs", b"saiz", b"saio", b"senc", b"trun", b"tfra", b"sidx"}
META_PROBE = 64 << 10        # bytes read from the start of a media-data box and from the file tail
META_BOX_CAP = 64 << 20      # a single metadata box larger than this is not read in full
META_TOTAL_CAP = 8 << 20     # upper bound on the bytes handed to the signature scan


def _strip_tables(buf: bytes) -> bytes:
    """``buf`` (payload of a moov / moof box) without its sample tables, which are large,
    binary and never hold provenance text."""
    drop: List[Tuple[int, int]] = []
    stack = [(0, len(buf))]
    while stack:
        a, b = stack.pop()
        for typ, s, e in _boxes(buf, a, b):
            if typ in TABLE_BOXES and e - s > 4096:
                drop.append((s, e))
            elif typ in CONTAINERS or typ in (b"traf", b"mvex"):
                stack.append((s + (4 if typ == b"meta" else 0), e))
    if not drop:
        return buf
    out, pos = [], 0
    for s, e in sorted(drop):
        out.append(buf[pos:s])
        pos = e
    out.append(buf[pos:])
    return b"".join(out)


def _metadata_bytes(path: str) -> bytes:
    """The parts of a file that can carry provenance text.

    For MP4 / MOV: every top-level box except the media data, with the sample tables
    removed, plus the first bytes of each media box (encoder strings) and the file tail.
    This is a few kilobytes instead of the 6 MB head-and-tail window, which the signature
    scan (about sixty regular expressions) needed seconds to search. Other containers, and
    anything that does not parse, fall back to the head-and-tail window.
    """
    size = os.path.getsize(path)
    parts: List[bytes] = []
    total = 0
    try:
        with open(path, "rb") as f:
            pos, boxes = 0, 0
            while pos + 8 <= size and total < META_TOTAL_CAP:
                f.seek(pos)
                hdr = f.read(16)
                if len(hdr) < 8:
                    break
                bsz, typ = struct.unpack(">I4s", hdr[:8])
                hs = 8
                if bsz == 1 and len(hdr) == 16:
                    bsz = struct.unpack(">Q", hdr[8:16])[0]
                    hs = 16
                elif bsz == 0:
                    bsz = size - pos
                if bsz < hs or not all(32 <= c < 127 for c in typ):
                    break
                f.seek(pos)
                if typ in MEDIA_BOXES:
                    chunk = f.read(min(bsz, META_PROBE))
                else:
                    chunk = f.read(min(bsz, META_BOX_CAP, size - pos))
                    if typ in (b"moov", b"moof"):
                        chunk = chunk[:hs] + _strip_tables(chunk[hs:])
                parts.append(chunk[:META_TOTAL_CAP - total])
                total += len(parts[-1])
                boxes += 1
                pos += bsz
            if boxes < 2 or not any(p[4:8] in (b"moov", b"moof") for p in parts):
                return read_head_tail(path)
            f.seek(max(0, size - META_PROBE))
            parts.append(f.read())                 # trailing bytes: appended tags, junk
    except (OSError, struct.error):
        return read_head_tail(path)
    return b"".join(parts)


def parse_mp4(path: str) -> Optional[Dict[str, Any]]:
    """Sample tables of the first video track, or None if not an ISO-BMFF file."""
    try:
        moov = _find_moov(path)
    except Exception:
        return None
    if not moov:
        return None
    try:
        return _parse_moov(moov)
    except (IndexError, ValueError, struct.error):
        return None        # malformed tables: treat as "no usable sample table"


def _parse_moov(moov: bytes) -> Dict[str, Any]:
    out: Dict[str, Any] = {"tracks": 0, "edit_entries": 0}
    for typ, s, e in _boxes(moov, 0, len(moov)):
        if typ == b"mvhd" and s + 20 <= e:
            ver = moov[s]
            if ver == 0:
                c, m, ts, dur = struct.unpack(">IIII", moov[s + 4:s + 20])
                out.update(creation=c, modification=m, timescale=ts, duration=dur / max(ts, 1))
        if typ != b"trak":
            continue
        out["tracks"] += 1
        trk: Dict[str, Any] = {"edits": []}
        stack = [(s, e)]
        while stack:
            a, b = stack.pop()
            for t2, s2, e2 in _boxes(moov, a, b):
                if t2 in CONTAINERS:
                    stack.append((s2 + (4 if t2 == b"meta" else 0), e2))
                elif t2 == b"hdlr" and s2 + 12 <= e2:
                    trk["handler"] = moov[s2 + 8:s2 + 12]
                elif t2 == b"elst" and s2 + 8 <= e2:
                    ver = moov[s2]
                    n = struct.unpack(">I", moov[s2 + 4:s2 + 8])[0]
                    step = 20 if ver == 1 else 12
                    p = s2 + 8
                    # Every entry must lie inside this box; never read a sibling's bytes.
                    for _ in range(min(n, 64)):
                        if ver not in (0, 1) or p + step > e2:
                            break
                        if ver == 1:
                            d, mt = struct.unpack(">Qq", moov[p:p + 16])
                        else:
                            d, mt = struct.unpack(">Ii", moov[p:p + 8])
                        p += step
                        trk["edits"].append((d, mt))
                elif t2 == b"stsz" and s2 + 12 <= e2:
                    fixed, n = struct.unpack(">II", moov[s2 + 4:s2 + 12])
                    if fixed == 0 and 0 < n <= 20_000_000 and s2 + 12 + 4 * n <= e2:
                        trk["sizes"] = np.frombuffer(moov, dtype=">u4", count=n, offset=s2 + 12
                                                     ).astype(np.float64)
                elif t2 == b"stss" and s2 + 8 <= e2:
                    n = struct.unpack(">I", moov[s2 + 4:s2 + 8])[0]
                    if 0 < n and s2 + 8 + 4 * n <= e2:
                        trk["sync"] = np.frombuffer(moov, dtype=">u4", count=n, offset=s2 + 8
                                                    ).astype(np.int64) - 1
                elif t2 == b"ctts":
                    trk["has_ctts"] = True
                elif t2 == b"stsd" and s2 + 16 <= e2:
                    trk["codec"] = moov[s2 + 12:s2 + 16].decode("latin-1", "replace")
        if trk.get("handler") == b"vide" and "video" not in out:
            out["video"] = trk
        # Real media edits: entries beyond a single full-length edit (empty edits have mt=-1).
        out["edit_entries"] = max(out["edit_entries"],
                                  len([1 for d, mt in trk["edits"] if mt >= 0]))
    return out


def analyze_container(path: str) -> Tuple[List[Evidence], Dict[str, Any]]:
    """Provenance evidence from container metadata. Returns (evidences, info)."""
    data = _metadata_bytes(path)
    sigs = scan_signatures(data)
    src_types = iptc_source_types(data)
    probe = ffmpeg_probe_text(path)
    tags = dict(re.findall(r"^\s{2,}([A-Za-z_\-\.]+)\s*:\s*(.+)$", probe, flags=re.M))
    info = {k: tags[k].strip() for k in ("major_brand", "encoder", "creation_time", "comment",
                                         "title", "software", "handler_name", "make", "model",
                                         "com.apple.quicktime.make", "com.apple.quicktime.model",
                                         "com.apple.quicktime.software", "com.android.version")
            if k in tags}
    streams = re.findall(r"Stream #\d+:\d+.*?: (Video|Audio): ([^\n]+)", probe)
    info["streams"] = [f"{k}: {v[:110]}" for k, v in streams]
    mp4 = parse_mp4(path)
    out: List[Evidence] = []

    ai = [s for s in sigs if s["kind"] == "ai"]
    ai_types = [t for t in src_types if "lgorithmic" in t or "ynthetic" in t]
    if ai or ai_types:
        out.append(Evidence(
            detector="video.generator_metadata", title="Generator metadata",
            category=Category.AI_GENERATION, score=0.97, confidence=1.0 if ai_types else 0.85,
            decisive=False,
            summary="Metadata declares AI generation (unverified): " + ", ".join(
                dict.fromkeys([s["label"] for s in ai] + [f"IPTC:{t}" for t in ai_types]))[:300] + ".",
            details={"unverified_generator_claim": True,
                     "signatures": ai, "iptc_source_types": src_types}))
    else:
        out.append(Evidence(
            detector="video.generator_metadata", title="Generator metadata",
            category=Category.AI_GENERATION, score=0.0, confidence=0.3,
            summary="No AI-generator metadata or Content Credentials AI assertion found "
                    "(metadata is easily stripped; platforms remove it on upload)."))

    reasons, score = [], 0.0
    edits = [s for s in sigs if s["kind"] == "edit"]
    if edits:
        reasons.append("editing software: " + ", ".join(s["label"] for s in edits))
        score = max(score, 0.75)
    if any(s["kind"] == "c2pa" for s in sigs):
        reasons.append("C2PA Content Credentials present (verify with c2patool)")
        score = max(score, 0.55)
    transc = [s["label"] for s in sigs if s["kind"] == "transcode"]
    if transc:
        reasons.append("written by " + ", ".join(transc) + " (a transcoder, not a camera — "
                       "normal for anything downloaded from the web)")
        score = max(score, 0.4)
    if mp4 and mp4.get("edit_entries", 0) > 1:
        reasons.append(f"edit list with {mp4['edit_entries']} media segments (the timeline was "
                       f"cut or re-arranged without re-encoding)")
        score = max(score, 0.7)
    out.append(Evidence(
        detector="video.provenance", title="Container & provenance", category=Category.PROVENANCE,
        score=score, confidence=0.8 if reasons else 0.3,
        summary=("; ".join(reasons) + ".") if reasons else
        "No editing or transcoding traces in the container metadata.",
        details={**info, "signatures": edits, "transcoders": transc,
                 "edit_entries": mp4.get("edit_entries") if mp4 else None}))
    if mp4 and "video" in mp4:
        info["container_codec"] = mp4["video"].get("codec")
    return out, {"tags": info, "mp4": mp4}


def _fold_peak(r: np.ndarray, L: int) -> Tuple[float, int]:
    """How much one phase of period L stands out from the other phases (z-score)."""
    n = (len(r) // L) * L
    if n < MIN_REPEATS * L:
        return 0.0, 0
    m = r[:n].reshape(-1, L)
    cnt = np.isfinite(m).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.where(cnt >= MIN_REPEATS - 1, np.nansum(m, axis=0) / np.maximum(cnt, 1), np.nan)
    if np.isfinite(mean).sum() < 4:
        return 0.0, 0
    top = int(np.nanargmax(mean))
    rest = np.delete(mean, top)
    rest = rest[np.isfinite(rest)]
    return float((mean[top] - rest.mean()) / (rest.std() + 0.05)), top


def _class_residual(sizes: np.ndarray, sync: np.ndarray) -> np.ndarray:
    """log(frame size) relative to nearby frames of the same kind (P-like vs B-like).

    Without parsing the bitstream, frame kinds are separated by size: B-frames are
    several times smaller than the P-frames around them.
    """
    from scipy.ndimage import median_filter, percentile_filter

    ls = np.log(np.maximum(sizes, 1.0))
    ls[sync] = np.nan
    filled = np.where(np.isnan(ls), np.nanmedian(ls), ls)
    local = median_filter(filled, size=9, mode="nearest")
    hi = percentile_filter(filled, 85, size=9, mode="nearest")   # upper (P-frame) envelope
    two_classes = float(np.nanmedian(hi - local)) > 0.7        # clear P/B size gap
    r = np.full_like(ls, np.nan)
    if two_classes:
        split = (local + hi) / 2
        for cls in (filled >= split, filled < split):
            idx = np.where(cls & np.isfinite(ls))[0]
            if len(idx) >= 9:
                r[idx] = ls[idx] - median_filter(ls[idx], size=9, mode="nearest")
    else:
        idx = np.where(np.isfinite(ls))[0]
        r[idx] = ls[idx] - median_filter(ls[idx], size=9, mode="nearest")
    return r


def analyze_double_compression(mp4: Optional[Dict[str, Any]]) -> Evidence:
    """Periodic frame-size spikes left by the key frames of a *previous* encoding.

    A frame that was an I-frame in the first encoding is harder to predict when the
    video is encoded again with a different GOP, so its size stands out at the old
    GOP period (after Vázquez-Padín et al., 2012). We fold the detrended sizes of
    non-key frames at every candidate period and look for one phase that sticks out.
    """
    det, title = "video.double_compression", "Double compression (GOP periodicity)"
    v = (mp4 or {}).get("video")
    if not v or "sizes" not in v or "sync" not in v:
        return Evidence.not_applicable(det, title, Category.PROVENANCE,
                                       "no MP4/MOV sample table with key-frame index")
    sizes = v["sizes"].copy()
    n = len(sizes)
    sync = v["sync"][(v["sync"] >= 0) & (v["sync"] < n)]
    if n < 90 or len(sync) < 2 or len(sync) > 0.5 * n:
        return Evidence.not_applicable(det, title, Category.PROVENANCE,
                                       "too few frames or all-intra stream", frames=int(n))
    gaps = np.diff(sync)
    gop = int(np.median(gaps)) if len(gaps) else 0
    r = _class_residual(sizes, sync)
    zs: Dict[int, Tuple[float, int]] = {}
    for L in range(5, min(301, n // MIN_REPEATS)):
        if gop and (L % gop == 0 or gop % L == 0):
            continue                      # the current GOP explains this period
        zs[L] = _fold_peak(r, L)
    best = (0.0, 0, 0)
    if zs:
        L = max(zs, key=lambda k: zs[k][0])
        z = zs[L][0]
        for d in sorted(zs):              # prefer the fundamental over its multiples
            if d < L and L % d == 0 and zs[d][0] >= 0.6 * z:
                L = d
                break
        best = (zs[L][0], L, zs[L][1])
    z, L, ph = best
    score = calibrate(z, PERIOD_Z_THRESHOLD, PERIOD_Z_SOFTNESS)
    bframes = bool(v.get("has_ctts"))
    return Evidence(
        detector=det, title=title, category=Category.PROVENANCE, score=score,
        confidence=DOUBLE_CONF_BFRAMES if bframes else DOUBLE_CONF,
        summary=(f"Frame sizes spike every {L} frames (z={z:.1f}) although the current key-frame "
                 f"interval is {gop}: the stream was very likely encoded before with a GOP of "
                 f"{L} — i.e. this is probably not the original encoding. Informational: "
                 f"content that repeats at that period produces the same pattern, and a "
                 f"re-encode alone says nothing about editing." if score >= 0.5 else
                 f"No frame-size periodicity from an earlier encoding (best z={z:.1f}; current "
                 f"GOP {gop})." + (" The stream uses B-frames, which mask this trace: a negative "
                                   "result means little here." if bframes else "")),
        details={"frames": int(n), "current_gop": gop, "candidate_period": L, "phase": ph,
                 "z": z, "threshold": PERIOD_Z_THRESHOLD, "b_frames": bframes},
        series={"frame_size": {"x": list(range(min(n, 1500))),
                               "y": [float(x) for x in sizes[:1500]],
                               "xlabel": "frame", "ylabel": "bytes"}})
