"""audio.metadata -- container, codec and tag provenance.

What it reads
-------------
* ``soundfile.info`` (container / codec / sample format) and libsndfile's own
  tag reader (``SoundFile.copy_metadata``),
* RIFF ``LIST/INFO`` sub-chunks (``ISFT`` software, ``ICMT`` comment, ...) and the
  Broadcast-WAV ``bext`` originator / coding history,
* ID3v2 text frames (``TSSE`` encoder settings, ``TENC``, ``TXXX``, ``COMM``),
* FLAC / Ogg Vorbis / Opus comment blocks, MP4 ``(c)too`` encoder atoms,
* a raw byte scan of the *metadata regions* with
  :func:`basira.core.utils.scan_signatures` (generator / editor / transcoder
  markers, C2PA manifests, IPTC digital-source-type codes).

The audio payload itself is excluded from the byte scan: PCM / compressed
frames are near-random bytes, and short signatures such as ``GIMP`` or
``Grok`` would match by chance in a few MB of samples.

How it scores
-------------
``raw`` = strongest marker class: 0 none, 1 transcoder (FFmpeg, LAME ...),
2 editing software (Audacity, Audition ...), 3 AI generator.
``score = calibrate(raw, MARKER_THRESHOLD=1.5, MARKER_SOFTNESS=0.5)`` so a
transcoder alone stays below the 0.5 decision threshold, editing software lands
around 0.73 and a generator marker at ~0.95. Only *explicit* generator markers
(voice-generator names, IPTC ``trainedAlgorithmicMedia``, C2PA AI assertions)
are marked ``decisive``. Metadata is trivially removable or forgeable, so
absence of markers is never evidence of authenticity.
"""
from __future__ import annotations

import os
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

from ..core import utils
from ..core.types import Category, Evidence

DETECTOR = "audio.metadata"
TITLE = "Audio metadata & provenance"

MARKER_THRESHOLD = 1.5
MARKER_SOFTNESS = 0.5
MAX_CHUNK_BYTES = 4 << 20       # never read more than this per metadata chunk
GENERIC_SCAN_BYTES = 256 << 10  # head / tail scanned for containers we do not parse

# Generator labels (from core.utils.SIGNATURES) that are explicit enough to be
# treated as decisive when they appear in metadata.
DECISIVE_AI_LABELS = {
    "ElevenLabs", "Commercial TTS service",
    "IPTC DigitalSourceType = trainedAlgorithmicMedia",
    "IPTC DigitalSourceType = compositeWithTrainedAlgorithmicMedia",
    "IPTC DigitalSourceType = algorithmicMedia",
    "C2PA AI assertion",
}
AI_SOURCE_TYPES = {"trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia",
                   "algorithmicMedia", "compositeSynthetic"}
LOSSY_SUBTYPES = {"MPEG_LAYER_I", "MPEG_LAYER_II", "MPEG_LAYER_III", "VORBIS", "OPUS",
                  "GSM610", "G721_32", "G723_24", "G723_40", "VOX_ADPCM", "IMA_ADPCM",
                  "MS_ADPCM", "ULAW", "ALAW", "NMS_ADPCM_16", "NMS_ADPCM_24", "NMS_ADPCM_32"}
LOSSY_EXT = {".mp3", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".wma", ".amr", ".mp4"}

RIFF_INFO_NAMES = {
    b"ISFT": "software", b"ICMT": "comment", b"INAM": "title", b"IART": "artist",
    b"IENG": "engineer", b"ITCH": "technician", b"ICRD": "date", b"ISRC": "source",
    b"IPRD": "product", b"ICOP": "copyright", b"IGNR": "genre", b"IKEY": "keywords",
    b"ISBJ": "subject",
}
ID3_NAMES = {"TSSE": "encoder_settings", "TENC": "encoded_by", "TIT2": "title",
             "TPE1": "artist", "TALB": "album", "TDRC": "date", "TYER": "year",
             "TCON": "genre", "TCOP": "copyright", "TPUB": "publisher", "TOOL": "tool"}


# --------------------------------------------------------------------------- #
# Container parsers (best effort, bounded reads, never raise)
# --------------------------------------------------------------------------- #
def _riff(path: str) -> Tuple[Dict[str, str], bytes]:
    """Walk RIFF/RF64 chunks, skipping the audio payload. Returns (tags, raw meta bytes)."""
    tags: Dict[str, str] = {}
    raw = bytearray()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64", b"BW64") or head[8:12] != b"WAVE":
            return tags, b""
        pos = 12
        n_chunks = 0
        while pos + 8 <= size and n_chunks < 4096:
            f.seek(pos)
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, csz = hdr[:4], struct.unpack("<I", hdr[4:8])[0]
            if cid == b"data" and csz == 0xFFFFFFFF:  # RF64: real size lives in ds64
                break
            n_chunks += 1
            if cid != b"data":
                payload = f.read(min(csz, MAX_CHUNK_BYTES))
                raw += cid + payload
                if cid == b"LIST" and payload[:4] == b"INFO":
                    p = 4
                    while p + 8 <= len(payload):
                        sid, ssz = payload[p:p + 4], struct.unpack("<I", payload[p + 4:p + 8])[0]
                        val = payload[p + 8:p + 8 + ssz].split(b"\x00")[0]
                        key = RIFF_INFO_NAMES.get(sid, sid.decode("latin-1", "replace"))
                        if val:
                            tags[key] = val.decode("utf-8", "replace").strip()
                        p += 8 + ssz + (ssz & 1)
                elif cid == b"bext" and len(payload) >= 602:
                    desc = payload[0:256].split(b"\x00")[0]
                    orig = payload[256:288].split(b"\x00")[0]
                    hist = payload[602:].split(b"\x00")[0]
                    for k, v in (("bext_description", desc), ("bext_originator", orig),
                                 ("bext_coding_history", hist)):
                        if v.strip():
                            tags[k] = v.decode("latin-1", "replace").strip()
            pos += 8 + csz + (csz & 1)
    return tags, bytes(raw)


def _decode_id3_text(enc: int, b: bytes) -> str:
    try:
        if enc == 0:
            return b.decode("latin-1")
        if enc == 1:
            return b.decode("utf-16")
        if enc == 2:
            return b.decode("utf-16-be")
        return b.decode("utf-8")
    except Exception:
        return b.decode("latin-1", "replace")


def _id3v2(data: bytes) -> Tuple[Dict[str, str], int]:
    """Parse an ID3v2.3/2.4 tag at the start of ``data``. Returns (tags, tag_size)."""
    tags: Dict[str, str] = {}
    if len(data) < 10 or data[:3] != b"ID3":
        return tags, 0
    ver = data[3]
    sz = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) | ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
    end = min(len(data), 10 + sz)
    p = 10
    if ver < 3:
        return tags, 10 + sz
    while p + 10 <= end:
        fid = data[p:p + 4]
        if not re.match(rb"[A-Z0-9]{4}", fid):
            break
        b = data[p + 4:p + 8]
        fsz = (((b[0] & 0x7F) << 21) | ((b[1] & 0x7F) << 14) | ((b[2] & 0x7F) << 7) | (b[3] & 0x7F)) \
            if ver == 4 else struct.unpack(">I", b)[0]
        body = data[p + 10:p + 10 + fsz]
        name = fid.decode("ascii")
        if body:
            enc = body[0]
            if name == "TXXX":
                parts = _decode_id3_text(enc, body[1:]).split("\x00", 1)
                if len(parts) == 2:
                    tags[f"TXXX:{parts[0].strip()}"] = parts[1].strip("\x00 ")
            elif name == "COMM" and len(body) > 4:
                txt = _decode_id3_text(enc, body[4:]).split("\x00")
                tags["comment"] = txt[-1].strip()
            elif name.startswith("T"):
                tags[ID3_NAMES.get(name, name)] = _decode_id3_text(enc, body[1:]).strip("\x00 ")
        p += 10 + fsz
    return tags, 10 + sz


def _vorbis_comments(block: bytes) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    try:
        vl = struct.unpack("<I", block[:4])[0]
        tags["vendor"] = block[4:4 + vl].decode("utf-8", "replace")
        p = 4 + vl
        n = struct.unpack("<I", block[p:p + 4])[0]
        p += 4
        for _ in range(min(n, 1000)):
            ln = struct.unpack("<I", block[p:p + 4])[0]
            kv = block[p + 4:p + 4 + ln].decode("utf-8", "replace")
            p += 4 + ln
            if "=" in kv:
                k, v = kv.split("=", 1)
                tags[k.lower()] = v
    except Exception:
        pass
    return tags


def _flac(path: str) -> Tuple[Dict[str, str], bytes]:
    tags: Dict[str, str] = {}
    raw = bytearray()
    with open(path, "rb") as f:
        data = f.read(10)
        off = 0
        if data[:3] == b"ID3":
            t, off = _id3v2(data + f.read(1 << 20))
            tags.update(t)
        f.seek(off)
        if f.read(4) != b"fLaC":
            return tags, b""
        for _ in range(256):
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            last, btype = hdr[0] & 0x80, hdr[0] & 0x7F
            ln = int.from_bytes(hdr[1:4], "big")
            if btype == 6:  # PICTURE: skip image payload
                f.seek(ln, 1)
            else:
                block = f.read(min(ln, MAX_CHUNK_BYTES))
                raw += block
                if btype == 4:
                    tags.update(_vorbis_comments(block))
                if ln > MAX_CHUNK_BYTES:
                    f.seek(ln - MAX_CHUNK_BYTES, 1)
            if last:
                break
    return tags, bytes(raw)


def _generic(path: str, ext: str) -> Tuple[Dict[str, str], bytes]:
    """MP3 / Ogg / MP4 and anything else: tags from known structures, bounded raw scan."""
    tags: Dict[str, str] = {}
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(GENERIC_SCAN_BYTES)
        id3_size = 0
        if head[:3] == b"ID3":
            t, id3_size = _id3v2(head)
            if id3_size > len(head):
                f.seek(0)
                t, id3_size = _id3v2(f.read(min(id3_size, MAX_CHUNK_BYTES)))
            tags.update(t)
        f.seek(max(0, size - GENERIC_SCAN_BYTES))
        tail = f.read()
    # For MPEG audio, keep the ID3 tag + the first frames (Xing/LAME header) only.
    if ext == ".mp3" or head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF
                                                and (head[1] & 0xE0) == 0xE0):
        head = head[:id3_size + 4096]
        tail = tail[-512:]  # ID3v1 / APE footer
    for marker in (b"\x03vorbis", b"OpusTags"):
        i = head.find(marker)
        if i >= 0:
            tags.update(_vorbis_comments(head[i + len(marker):]))
    blob = head + tail
    j = blob.find(b"\xa9too")  # MP4 encoder atom: ©too -> data atom
    if j >= 0 and blob[j + 8:j + 12] == b"data":
        ln = struct.unpack(">I", blob[j + 4:j + 8])[0]
        tags["encoder"] = blob[j + 20:j + 4 + ln].decode("utf-8", "replace")
    return tags, blob


def read_tags(path: str) -> Tuple[Dict[str, str], bytes, Dict[str, Any]]:
    """Return (tags, metadata bytes for signature scanning, soundfile info dict)."""
    import soundfile as sf

    sfinfo: Dict[str, Any] = {}
    tags: Dict[str, str] = {}
    try:
        inf = sf.info(path)
        sfinfo = {"format": inf.format, "format_info": inf.format_info,
                  "subtype": inf.subtype, "subtype_info": inf.subtype_info,
                  "sample_rate": int(inf.samplerate), "channels": int(inf.channels),
                  "frames": int(inf.frames), "duration": float(inf.duration)}
        with sf.SoundFile(path) as fh:
            for k, v in fh.copy_metadata().items():
                if v:
                    tags[k] = str(v)
    except Exception:
        pass
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        magic = f.read(12)
    if magic[:4] in (b"RIFF", b"RF64", b"BW64") and magic[8:12] == b"WAVE":
        t, raw = _riff(path)
    elif magic[:4] == b"fLaC" or ext == ".flac":
        t, raw = _flac(path)
    else:
        t, raw = _generic(path, ext)
    tags.update(t)
    return tags, raw, sfinfo


def is_lossy(sfinfo: Dict[str, Any], path: str = "") -> bool:
    sub = str(sfinfo.get("subtype", "")).upper()
    if sub in LOSSY_SUBTYPES:
        return True
    return os.path.splitext(path)[1].lower() in LOSSY_EXT


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
def detect_metadata(path: str, *, skip_reason: Optional[str] = None,
                    tags: Optional[Dict[str, str]] = None, raw: Optional[bytes] = None,
                    sfinfo: Optional[Dict[str, Any]] = None) -> Evidence:
    if skip_reason:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.PROVENANCE, skip_reason)
    if not path or not os.path.isfile(path):
        return Evidence.not_applicable(DETECTOR, TITLE, Category.PROVENANCE,
                                       "no file on disk (samples were passed directly)")
    if tags is None or raw is None or sfinfo is None:
        tags, raw, sfinfo = read_tags(path)

    tag_text = "\n".join(f"{k}={v}" for k, v in tags.items()).encode("utf-8", "replace")
    blob = raw + b"\n" + tag_text
    hits = utils.scan_signatures(blob)
    src_types = utils.iptc_source_types(blob)
    ai = [h for h in hits if h["kind"] == "ai"]
    for code in src_types:
        if code in AI_SOURCE_TYPES and not any(code in h["label"] for h in ai):
            ai.append({"label": f"IPTC DigitalSourceType = {code}", "kind": "ai",
                       "offset": "", "context": ""})
    edit = [h for h in hits if h["kind"] == "edit"]
    trans = [h for h in hits if h["kind"] == "transcode"]
    c2pa = [h for h in hits if h["kind"] == "c2pa"]

    raw_level = 3.0 if ai else 2.0 if edit else 1.0 if trans else 0.0
    score = utils.calibrate(raw_level, MARKER_THRESHOLD, MARKER_SOFTNESS)
    decisive = any(h["label"] in DECISIVE_AI_LABELS for h in ai)
    category = Category.AI_GENERATION if ai else Category.PROVENANCE

    software = [tags[k] for k in ("software", "encoder", "encoder_settings", "encoded_by",
                                  "tool", "vendor", "bext_originator", "bext_coding_history")
                if tags.get(k)]
    parts: List[str] = []
    if ai:
        parts.append("Generator marker(s) found in metadata: "
                     + ", ".join(h["label"] for h in ai) + ".")
    if edit:
        parts.append("Editing software recorded: " + ", ".join(h["label"] for h in edit) + ".")
    if trans:
        parts.append("Re-encoded by: " + ", ".join(h["label"] for h in trans)
                     + " (transcoding alone is not manipulation).")
    if c2pa:
        parts.append("A C2PA content-credentials manifest is present (its signature is not "
                     "verified by Basira).")
    if not parts:
        parts.append("No generator, editor or transcoder markers found. Metadata is easy to "
                     "strip, so this is not evidence of authenticity.")
    parts.append(f"Marker level {raw_level:.0f}/3 (0 none, 1 transcoder, 2 editor, 3 generator).")

    if ai:
        confidence = 0.9 if decisive else 0.6
    elif edit:
        confidence = 0.6
    elif trans:
        confidence = 0.4
    else:
        confidence = 0.2
    details = {
        "raw_marker_level": raw_level, "threshold": MARKER_THRESHOLD,
        "container": sfinfo, "lossy": is_lossy(sfinfo, path),
        "tags": {k: (v[:200] if isinstance(v, str) else v) for k, v in tags.items()},
        "software": software, "signatures": hits, "iptc_source_types": src_types,
    }
    return Evidence(detector=DETECTOR, title=TITLE, category=category, score=score,
                    confidence=confidence, summary=" ".join(parts), details=details,
                    decisive=bool(decisive and score >= 0.9))
