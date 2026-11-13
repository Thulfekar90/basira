"""Video engine: temporal edits, container provenance, key-frame and audio-track analysis."""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from ..core.fusion import finalize
from ..core.io import extract_audio_track
from ..core.types import Category, Evidence, Report
from ..core.utils import thumbnail
from .container import analyze_container, analyze_double_compression
from .temporal import (FrameScan, detect_freeze, detect_jump, detect_repeat, detect_source,
                       scan_frames)

KEYFRAME_DETECTORS = ("image.copymove",)


def _keyframe_evidence(fs: FrameScan) -> List[Evidence]:
    """Run still-image detectors on key frames and keep only what persists across them."""
    from ..image import analyze_frame

    per: dict = {}
    for fi, rgb in fs.keyframes:
        for e in analyze_frame(rgb, f"frame {fi}", detectors=set(KEYFRAME_DETECTORS)):
            per.setdefault(e.detector, []).append((fi, e))
    out = []
    for det, items in per.items():
        good = [(fi, e) for fi, e in items if e.error is None and e.confidence > 0]
        if not good:
            continue
        scores = np.array([e.score for _, e in good])
        fi_best, best = max(good, key=lambda t: t[1].score)
        persistent = float(np.median(scores))
        hits = int((scores >= 0.5).sum())
        e = Evidence(
            detector=det.replace("image.", "video.frame_"), title=best.title + " — key frames",
            category=best.category, score=persistent, confidence=best.confidence * 0.7,
            summary=(f"Detected in {hits} of {len(good)} key frames (strongest at frame "
                     f"{fi_best}, {fi_best / fs.fps:.2f}s): {best.summary}" if hits else
                     f"Not detected in any of {len(good)} key frames."),
            details={"per_frame": {str(fi): round(e.score, 3) for fi, e in good},
                     "best_frame": int(fi_best), **{k: v for k, v in best.details.items()
                                                    if k in ("transforms", "area_fraction")}},
            heatmap=best.heatmap if persistent >= 0.5 else None,
            visuals=best.visuals if persistent >= 0.5 else {})
        out.append(e)
    return out


def analyze_video(path: str, *, max_frames: int = 3000, max_decode: int = 5400,
                  analyze_audio_track: bool = True, **_) -> Report:
    report = Report(path=path, media_type="video", info={"bytes": os.path.getsize(path)})

    try:
        evs, cinfo = analyze_container(path)
        report.evidences.extend(evs)
        report.info.update({k: v for k, v in cinfo["tags"].items() if k != "streams"})
        if cinfo["tags"].get("streams"):
            report.info["streams"] = " | ".join(cinfo["tags"]["streams"])
        mp4 = cinfo["mp4"]
    except Exception as e:
        mp4 = None
        report.evidences.append(Evidence.failed("video.provenance", "Container & provenance",
                                                Category.PROVENANCE, e))
    try:
        report.evidences.append(analyze_double_compression(mp4))
    except Exception as e:
        report.evidences.append(Evidence.failed("video.double_compression", "Double compression",
                                                Category.PROVENANCE, e))

    fs: Optional[FrameScan] = None
    try:
        fs = scan_frames(path, max_frames=max_frames, max_decode=max_decode)
    except Exception as e:
        report.evidences.append(Evidence.failed("video.decode", "Frame decoding",
                                                Category.MANIPULATION, e))
    shape = None
    if fs is not None:
        report.info.update({"width": fs.width, "height": fs.height, "fps": round(fs.fps, 3),
                            "frames": fs.n_total, "duration": round(fs.duration, 3),
                            "codec": fs.codec, "analysed_every_nth_frame": fs.stride,
                            "analysed_seconds": round(fs.analysed_frames / fs.fps, 2),
                            "truncated": fs.truncated})
        cuts: List[int] = []
        for name, title, fn in (("video.freeze", "Frozen / duplicated frames", detect_freeze),
                                ("video.repeat", "Replayed footage", detect_repeat)):
            try:
                report.evidences.append(fn(fs))
            except Exception as e:
                report.evidences.append(Evidence.failed(name, title, Category.MANIPULATION, e))
        try:
            ev, cuts = detect_jump(fs)
            report.evidences.append(ev)
        except Exception as e:
            report.evidences.append(Evidence.failed("video.jump", "Temporal jumps",
                                                    Category.MANIPULATION, e))
        try:
            report.evidences.append(detect_source(fs, cuts))
        except Exception as e:
            report.evidences.append(Evidence.failed("video.source", "Source consistency",
                                                    Category.MANIPULATION, e))
        try:
            report.evidences.extend(_keyframe_evidence(fs))
        except Exception as e:
            report.evidences.append(Evidence.failed("video.frame_copymove", "Key-frame analysis",
                                                    Category.MANIPULATION, e))
        if fs.keyframes:
            key = fs.keyframes[len(fs.keyframes) // 2][1]
            for e in report.evidences:
                if e.heatmap is not None and e.details.get("best_frame") is not None:
                    key = dict(fs.keyframes).get(e.details["best_frame"], key)
            report.preview = thumbnail(key, 1600)
            shape = key.shape[:2]

    if analyze_audio_track:
        try:
            track = extract_audio_track(path, max_seconds=600.0)
            if track is not None and track[0].shape[0] > track[1]:   # > 1 s of audio
                from ..audio import analyze_audio

                child = analyze_audio(path, samples=track[0], sr=track[1], label="audio track")
                report.children.append(child)
        except Exception as e:
            report.evidences.append(Evidence.failed("video.audio_track", "Audio track",
                                                    Category.MANIPULATION, e))
    return finalize(report, spatial_shape=shape)


__all__ = ["analyze_video"]
