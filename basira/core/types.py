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
    decisive: bool = False            # hard proof (e.g. generator metadata)
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
        return {
            "detector": self.detector,
            "title": self.title,
            "category": self.category.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "summary": self.summary,
            "decisive": self.decisive,
            "details": _jsonable(self.details),
            "segments": [s.to_dict() for s in self.segments],
            "regions": [r.to_dict() for r in self.regions],
            "has_heatmap": self.heatmap is not None,
            "error": self.error,
        }

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
        return {
            "tool": "basira",
            "version": self.version,
            "path": self.path,
            "media_type": self.media_type,
            "info": _jsonable(self.info),
            "verdict": self.verdict,
            "verdict_level": self.verdict_level,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "explanation": self.explanation,
            "segments": [s.to_dict() for s in self.fused_segments],
            "evidences": [e.to_dict() for e in self.evidences],
            "children": [c.to_dict() for c in self.children],
            "elapsed_sec": round(self.elapsed, 3),
        }


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy / exotic types into JSON-serialisable values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 256 else f"<array {obj.shape}>"
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, bytes):
        return obj[:64].decode("latin-1", "replace")
    if isinstance(obj, Enum):
        return obj.value
    return obj
