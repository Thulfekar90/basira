"""Core data model shared by every Basira engine.

Every detector returns one :class:`Evidence`. An engine (image / audio / video)
collects evidences into a :class:`Report`, which fuses them into calibrated
category scores and a human-readable verdict.

Design rules
------------
* ``score`` answers "how anomalous is this input according to this detector"
  on a 0..1 scale, where ~0.5 is the detector's own decision threshold.
* ``confidence`` answers "how much should this detector be trusted *on this
  particular input*" (e.g. ELA on a PNG, or ENF on a recording without mains
  hum, gets a low confidence). Fusion weighs by it.
* Localisation is optional: a spatial ``heatmap`` (H x W, 0..1) for pixels, and
  ``segments`` (time ranges) for audio / video.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class Category(str, Enum):
    """What kind of question a piece of evidence speaks to."""

    MANIPULATION = "manipulation"      # local edits: splicing, copy-move, cuts
    AI_GENERATION = "ai_generation"    # fully / partially synthetic content
    PROVENANCE = "provenance"          # metadata, editing software, C2PA


@dataclass
class Segment:
    """A suspicious time range (seconds) in audio or video."""

    start: float
    end: float
    score: float
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "score": round(float(self.score), 3), "label": self.label}


@dataclass
class Region:
    """A suspicious axis-aligned box in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    score: float
    label: str = ""
    frame: Optional[int] = None  # video frame index, if any

    def to_dict(self) -> Dict[str, Any]:
        d = {"x": int(self.x), "y": int(self.y), "w": int(self.w), "h": int(self.h),
             "score": round(float(self.score), 3), "label": self.label}
        if self.frame is not None:
            d["frame"] = int(self.frame)
        return d


@dataclass
class Evidence:
    detector: str                     # stable id, e.g. "image.ela"
    title: str                        # human title
    category: Category
    score: float                      # 0..1 anomaly score (0.5 ~ threshold)
    confidence: float                 # 0..1 trust in this detector for this input
    summary: str                      # one or two sentences, plain language
    details: Dict[str, Any] = field(default_factory=dict)
    heatmap: Optional[np.ndarray] = None          # HxW float32 0..1
    segments: List[Segment] = field(default_factory=list)
    regions: List[Region] = field(default_factory=list)
    visuals: Dict[str, np.ndarray] = field(default_factory=dict)  # extra RGB uint8 images
    series: Dict[str, Any] = field(default_factory=dict)          # 1-D curves for plots
    decisive: bool = False            # legacy field; not signature verification or level-4 authority
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not (np.isfinite(self.score) and np.isfinite(self.confidence)):
            # A detector that returns NaN has failed; never let it poison the fusion.
            self.error = self.error or "detector returned a non-finite score"
            self.score, self.confidence = 0.0, 0.0
        self.score = float(np.clip(self.score, 0.0, 1.0))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if self.heatmap is not None:
            self.heatmap = np.clip(np.asarray(self.heatmap, dtype=np.float32), 0.0, 1.0)

    @property
    def weight(self) -> float:
        return self.score * self.confidence

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict (documented in docs/JSON.md). Strictly serialisable: no NaN,
        no Infinity, no numpy types."""
        score = round(float(self.score), 4)
        if self.score < 0.5 <= score:
            score = 0.4999        # rounding must not carry a quiet detector over its threshold
        return _jsonable({
            "detector": self.detector,
            "title": self.title,
            "category": getattr(self.category, "value", self.category),
            "score": score,
            "confidence": round(float(self.confidence), 4),
            "summary": self.summary,
            "decisive": bool(self.decisive),
            "details": self.details,
            "segments": [s.to_dict() for s in self.segments],
            "regions": [r.to_dict() for r in self.regions],
            "has_heatmap": self.heatmap is not None,
            "error": self.error,
        })

    @classmethod
    def failed(cls, detector: str, title: str, category: Category, err: Exception) -> "Evidence":
        return cls(detector=detector, title=title, category=category, score=0.0,
                   confidence=0.0, summary=f"Detector failed: {err}", error=repr(err))

    @classmethod
    def not_applicable(cls, detector: str, title: str, category: Category, why: str,
                       **details: Any) -> "Evidence":
        return cls(detector=detector, title=title, category=category, score=0.0,
                   confidence=0.0, summary=f"Not applicable: {why}", details=dict(details))


@dataclass
class Report:
    path: str
    media_type: str                    # "image" | "audio" | "video"
    info: Dict[str, Any]
    evidences: List[Evidence] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    verdict: str = ""
    verdict_level: int = 0             # 0 clean .. 4 decisive
    explanation: List[str] = field(default_factory=list)
    fused_heatmap: Optional[np.ndarray] = None
    fused_segments: List[Segment] = field(default_factory=list)
    preview: Optional[np.ndarray] = None   # RGB uint8 thumbnail / key frame
    children: List["Report"] = field(default_factory=list)  # e.g. audio track of a video
    elapsed: float = 0.0
    version: str = ""

    def get(self, detector: str) -> Optional[Evidence]:
        for e in self.evidences:
            if e.detector == detector:
                return e
        return None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict: the stable machine-readable output (documented field by field
        in docs/JSON.md). ``json.dumps(report.to_dict(), allow_nan=False)`` always works."""
        info = dict(self.info)
        # Always present, so a consumer can tell "nothing found" from "part of it crashed".
        info.setdefault("detectors_failed", sum(1 for e in self.evidences if e.error))
        info.setdefault("children_failed", sum(
            1 for c in self.children
            if c.info.get("detectors_failed") or c.info.get("children_failed")))
        return _jsonable({
            "tool": "basira",
            "schema_version": SCHEMA_VERSION,
            "version": self.version,
            "path": self.path,
            "media_type": self.media_type,
            "info": info,
            "verdict": self.verdict,
            "verdict_level": int(self.verdict_level),
            "scores": {k: round(float(v), 4) for k, v in self.scores.items()},
            "explanation": list(self.explanation),
            "segments": [s.to_dict() for s in self.fused_segments],
            "evidences": [e.to_dict() for e in self.evidences],
            "children": [c.to_dict() for c in self.children],
            "elapsed_sec": round(float(self.elapsed), 3),
        })


# Version of the JSON layout produced by ``Report.to_dict``. It changes only when a
# documented field is removed, renamed or changes type (see docs/JSON.md).
SCHEMA_VERSION = 1


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy / exotic types into strict JSON values.

    The result contains only dict (with str keys), list, str, int, float, bool and None,
    and every float is finite: NaN and +-Infinity become ``None`` because they are not
    valid JSON. Unknown objects are converted with ``str`` rather than raising."""
    if isinstance(obj, Enum):                # before str: Category is a str subclass
        return _jsonable(obj.value)
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return round(v, 6) if math.isfinite(v) else None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_jsonable(v) for v in sorted(obj, key=repr)]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist()) if obj.size <= 256 else f"<array {obj.shape}>"
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj[:64]).decode("latin-1", "replace")
    if isinstance(obj, np.generic):          # np.str_, np.bytes_, np.datetime64, complex ...
        try:
            return _jsonable(obj.item())
        except Exception:
            return str(obj)
    if isinstance(obj, complex):
        return str(obj)
    try:                                     # Fraction, Decimal, PIL IFDRational, ...
        v = float(obj)
        return round(v, 6) if math.isfinite(v) else None
    except Exception:
        return str(obj)
