"""Evidence fusion: many weak, explainable detectors -> one honest verdict.

Each detector's score is calibrated so 0.5 is its own decision threshold. Only
the *excess above threshold* contributes, scaled by the detector's confidence on
this input, and contributions are combined with a noisy-OR. Consequences:

* detectors that are quiet (score < 0.5) never push the verdict up,
* one strong, trusted detector is enough to flag,
* several moderate, independent detectors that agree reinforce each other,
* "decisive" evidence (e.g. a generator's own metadata) short-circuits.

Basira never outputs "authentic": absence of evidence is reported as
"no significant traces found".
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .types import Category, Evidence, Report, Segment
from .utils import upsample_to

LEVELS = {
    0: "No significant traces found",
    1: "Inconclusive (weak anomalies)",
    2: "Suspicious",
    3: "Likely {what}",
    4: "{What} (provenance evidence)",
}


def _excess(e: Evidence) -> float:
    return float(np.clip((e.score - 0.5) / 0.5, 0.0, 1.0)) * e.confidence


def category_score(evs: List[Evidence], cat: Category) -> float:
    rel = [e for e in evs if e.category == cat and e.error is None and e.confidence > 0]
    if not rel:
        return 0.0
    if any(e.decisive and e.score >= 0.9 for e in rel):
        return max(e.score for e in rel if e.decisive)
    p_clean = 1.0
    for e in rel:
        p_clean *= 1.0 - _excess(e)
    return float(1.0 - p_clean)


def fuse_heatmap(evs: List[Evidence], shape) -> Optional[np.ndarray]:
    """Localisation map built only from detectors that actually fired (score >= 0.5),
    weighted by how far above threshold and how trustworthy each one is. Quiet
    detectors contribute nothing, so their background texture never leaks in."""
    maps, ws = [], []
    for e in evs:
        if e.heatmap is None or e.error or e.confidence <= 0 or e.score < 0.5:
            continue
        w = _excess(e) + 1e-3
        h = e.heatmap if e.heatmap.shape == tuple(shape) else upsample_to(e.heatmap, shape)
        maps.append(h * w)
        ws.append(w)
    if not maps:
        return None
    fused = np.sum(maps, axis=0) / (np.sum(ws) + 1e-9)
    m = fused.max()
    return (fused / m).astype(np.float32) if m > 0 else fused.astype(np.float32)


def merge_segments(evs: List[Evidence], gap: float = 0.25) -> List[Segment]:
    segs = sorted((s for e in evs if e.error is None for s in e.segments),
                  key=lambda s: s.start)
    out: List[Segment] = []
    for s in segs:
        if out and s.start <= out[-1].end + gap:
            last = out[-1]
            labels = [lb for lb in (last.label, s.label) if lb]
            out[-1] = Segment(last.start, max(last.end, s.end), max(last.score, s.score),
                              " + ".join(dict.fromkeys(" + ".join(labels).split(" + "))))
        else:
            out.append(Segment(s.start, s.end, s.score, s.label))
    return out


def finalize(report: Report, spatial_shape=None) -> Report:
    evs = report.evidences
    manip = category_score(evs, Category.MANIPULATION)
    ai = category_score(evs, Category.AI_GENERATION)
    prov = category_score(evs, Category.PROVENANCE)
    # Child reports (e.g. a video's audio track) feed the parent.
    for c in report.children:
        manip = 1 - (1 - manip) * (1 - 0.8 * c.scores.get("manipulation", 0.0))
        ai = 1 - (1 - ai) * (1 - 0.8 * c.scores.get("ai_generation", 0.0))
    overall = 1 - (1 - manip) * (1 - ai) * (1 - 0.5 * prov)
    report.scores = {"manipulation": manip, "ai_generation": ai, "provenance": prov,
                     "overall": overall}

    decisive = [e for e in evs if e.decisive and e.score >= 0.9]
    what, top = ("AI-generated", ai) if ai >= manip else ("manipulated", manip)
    if decisive:
        level = 4
        kind = "AI-generated" if any(e.category == Category.AI_GENERATION for e in decisive) \
            else "Edited"
        verdict = LEVELS[4].format(What=kind)
    # The verdict follows the stronger single category. Weak hints from different
    # categories (say, a mild pixel anomaly plus "512x512 PNG without EXIF") must not add
    # up to an accusation; ``overall`` is reported but does not drive the level.
    elif top >= 0.75:
        level, verdict = 3, LEVELS[3].format(what=what)
    elif top >= 0.45:
        level, verdict = 2, LEVELS[2]
    elif top >= 0.2 or prov >= 0.5:
        level, verdict = 1, LEVELS[1]
    else:
        level, verdict = 0, LEVELS[0]
    report.verdict, report.verdict_level = verdict, level

    ranked = sorted((e for e in evs if e.error is None and e.confidence > 0),
                    key=lambda e: (e.decisive, _excess(e), e.score * e.confidence), reverse=True)
    report.explanation = [f"[{e.title}] {e.summary}" for e in ranked if e.score >= 0.5][:6]
    if not report.explanation:
        report.explanation = ["No detector exceeded its threshold. This is not proof of "
                              "authenticity: skilled edits and strong re-compression can "
                              "erase the traces these detectors rely on."]

    # Findings of child reports (a video's sound track) must be visible on the parent.
    child_segments: List[Segment] = []
    for c in report.children:
        tag = f"{c.media_type} track"
        if max(c.scores.get("manipulation", 0), c.scores.get("ai_generation", 0)) >= 0.2:
            lines = [x for x in c.explanation if x.startswith("[")][:3]
            if report.explanation and report.explanation[0].startswith("No detector"):
                report.explanation = []
            report.explanation.extend(f"[{tag}] {x}" for x in lines)
        child_segments.extend(Segment(s.start, s.end, s.score, f"{tag}: {s.label}")
                              for s in c.fused_segments)

    # A detector crash or an undecodable file is not a clean bill of health. Only a
    # content detector that actually ran counts; a child report (sound track) cannot
    # vouch for its parent's pixels, and a failed child counts as a failure here.
    def ran_ok(e: Evidence) -> bool:
        return e.error is None and e.confidence > 0 and e.category == Category.MANIPULATION

    failed = [e for e in evs if e.error]
    child_failed = [c for c in report.children if c.info.get("detectors_failed") and
                    not any(ran_ok(e) for e in c.evidences)]
    content_ok = any(ran_ok(e) for e in evs)
    report.info["detectors_failed"] = len(failed)
    if report.verdict_level == 0 and (failed or child_failed):
        report.verdict_level = 1
        if not content_ok:
            report.verdict = "Inconclusive (content could not be analysed)"
            tail = ("No content detector could run on this file, so the absence of "
                    "findings means nothing.")
        else:
            report.verdict = "Inconclusive (part of the analysis failed)"
            tail = "The remaining detectors found nothing, but the analysis is incomplete."
        report.explanation = [f"[{e.title}] {e.summary}" for e in failed[:4]] + \
            [f"[{c.media_type} track] could not be analysed" for c in child_failed] + [tail]

    if spatial_shape is not None:
        report.fused_heatmap = fuse_heatmap(
            [e for e in evs if e.category == Category.MANIPULATION], spatial_shape)
    report.fused_segments = merge_segments(evs)
    if child_segments:
        report.fused_segments = sorted(report.fused_segments + child_segments,
                                       key=lambda s: s.start)
    return report


def scores_table(report: Report) -> Dict[str, float]:
    return dict(report.scores)
