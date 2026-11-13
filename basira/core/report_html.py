"""Self-contained HTML report (one file, no network, light + dark)."""
from __future__ import annotations

import html
import json
import os
from typing import List, Optional

import numpy as np

from .types import Category, Evidence, Report
from .utils import jpeg_b64, overlay, png_b64

LEVEL_CLASS = {0: "ok", 1: "weak", 2: "warn", 3: "bad", 4: "bad"}
CAT_LABEL = {Category.MANIPULATION: "Manipulation", Category.AI_GENERATION: "AI generation",
             Category.PROVENANCE: "Provenance"}

CSS = """
:root{--bg:#f6f5f1;--card:#fff;--ink:#1c1b19;--mute:#6b6861;--line:#e3e0d8;--ok:#2f7d4f;
--weak:#8a7a2c;--warn:#c2700f;--bad:#c0392b;--accent:#3b5bdb;--chip:#efede7}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;
--mute:#9b978d;--line:#33322e;--ok:#5fbf85;--weak:#d1bd55;--warn:#f0a04b;--bad:#f0705f;
--accent:#8aa2ff;--chip:#2a2927}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:15px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--mute);margin:0 0 18px}
h2{font-size:18px;margin:34px 0 12px}h3{font-size:15px;margin:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:12px 0}
.verdict{display:flex;gap:20px;align-items:center;flex-wrap:wrap}
.badge{font-size:24px;font-weight:700;line-height:1.2}
.ok{color:var(--ok)}.weak{color:var(--weak)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.file{color:var(--mute);font-size:13px;word-break:break-all}
.scores{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:16px}
.score b{font-size:22px;font-variant-numeric:tabular-nums}.score span{color:var(--mute);font-size:13px;display:block}
.bar{position:relative;height:8px;border-radius:4px;background:var(--chip);margin-top:6px;overflow:hidden}
.bar i{position:absolute;inset:0 auto 0 0;border-radius:4px;background:var(--accent)}
.bar u{position:absolute;left:50%;top:0;bottom:0;width:2px;background:var(--mute);opacity:.6}
ul.why{margin:12px 0 0;padding-left:20px}ul.why li{margin:4px 0}
.viewer{position:relative;display:inline-block;max-width:100%;line-height:0;border-radius:8px;overflow:hidden}
.viewer img{max-width:100%;height:auto;display:block}.viewer img.heat{position:absolute;inset:0;width:100%;height:100%}
.ctl{display:flex;gap:10px;align-items:center;color:var(--mute);font-size:13px;margin-top:8px}
.ev{display:grid;grid-template-columns:minmax(0,1fr);gap:10px}
.evhead{display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap}
.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12px;color:var(--mute);white-space:nowrap}
.meta{color:var(--mute);font-size:13px;font-variant-numeric:tabular-nums}
.thumbs{display:flex;gap:10px;flex-wrap:wrap}.thumbs figure{margin:0;max-width:330px}
.thumbs img{max-width:100%;border-radius:6px;display:block}.thumbs figcaption{font-size:12px;color:var(--mute);margin-top:3px}
details{font-size:13px}summary{cursor:pointer;color:var(--mute)}
pre{background:var(--chip);padding:10px;border-radius:8px;overflow:auto;font-size:12px;max-height:320px}
svg{max-width:100%;height:auto;display:block}.quiet{opacity:.62}
table{border-collapse:collapse;font-size:13px}td{padding:2px 14px 2px 0;vertical-align:top}td:first-child{color:var(--mute)}
footer{color:var(--mute);font-size:12.5px;margin-top:36px;border-top:1px solid var(--line);padding-top:14px}
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _bar(v: float, cls: str = "") -> str:
    return (f'<div class="bar"><i style="width:{v * 100:.0f}%"></i><u></u></div>')


def _series_svg(s: dict, w: int = 720, h: int = 120) -> str:
    x = np.asarray(s.get("x", []), dtype=float)
    y = np.asarray(s.get("y", []), dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return ""
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(np.percentile(y, 0.5)), float(np.percentile(y, 99.5))
    if y1 - y0 < 1e-12:
        y0, y1 = y0 - 1, y1 + 1
    pad = 26
    px = pad + (x - x0) / max(x1 - x0, 1e-12) * (w - pad - 6)
    py = h - 18 - (np.clip(y, y0, y1) - y0) / (y1 - y0) * (h - 28)
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))
    marks = "".join(
        f'<line x1="{pad + (m - x0) / max(x1 - x0, 1e-12) * (w - pad - 6):.1f}" y1="6" '
        f'x2="{pad + (m - x0) / max(x1 - x0, 1e-12) * (w - pad - 6):.1f}" y2="{h - 18}" '
        f'stroke="var(--bad)" stroke-width="1.5" stroke-dasharray="3 3"/>'
        for m in s.get("markers", []) if x0 <= m <= x1)
    return (f'<svg viewBox="0 0 {w} {h}" role="img"><rect x="{pad}" y="6" width="{w - pad - 6}" '
            f'height="{h - 24}" fill="none" stroke="var(--line)"/>{marks}'
            f'<polyline fill="none" stroke="var(--accent)" stroke-width="1.2" points="{pts}"/>'
            f'<text x="{pad}" y="{h - 4}" font-size="10" fill="var(--mute)">{x0:.4g}</text>'
            f'<text x="{w - 6}" y="{h - 4}" font-size="10" fill="var(--mute)" text-anchor="end">'
            f'{x1:.4g} {_esc(s.get("xlabel", ""))}</text>'
            f'<text x="2" y="14" font-size="10" fill="var(--mute)">{y1:.4g}</text>'
            f'<text x="2" y="{h - 20}" font-size="10" fill="var(--mute)">{y0:.4g}</text>'
            f'<text x="{w / 2}" y="{h - 4}" font-size="10" fill="var(--mute)" text-anchor="middle">'
            f'{_esc(s.get("ylabel", ""))}</text></svg>')


def _timeline_svg(segments, duration: float, w: int = 1000) -> str:
    if not duration or duration <= 0:
        return ""
    rects = "".join(
        f'<rect x="{s.start / duration * w:.1f}" y="4" width="{max((s.end - s.start) / duration * w, 3):.1f}" '
        f'height="22" fill="var(--bad)" opacity="{0.35 + 0.6 * s.score:.2f}"><title>'
        f'{s.start:.2f}–{s.end:.2f}s {_esc(s.label)}</title></rect>' for s in segments)
    ticks = "".join(
        f'<text x="{i / 10 * w:.0f}" y="40" font-size="10" fill="var(--mute)" '
        f'text-anchor="{"start" if i == 0 else "end" if i == 10 else "middle"}">'
        f'{duration * i / 10:.1f}s</text>' for i in range(0, 11, 2))
    return (f'<svg viewBox="0 0 {w} 44"><rect x="0" y="4" width="{w}" height="22" rx="4" '
            f'fill="var(--chip)"/>{rects}{ticks}</svg>')


def _evidence_html(e: Evidence, preview: Optional[np.ndarray]) -> str:
    quiet = " quiet" if (e.score < 0.5 or e.confidence <= 0.05) else ""
    parts = [f'<div class="card ev{quiet}"><div class="evhead"><h3>{_esc(e.title)}</h3>'
             f'<span class="chip">{CAT_LABEL[e.category]}</span></div>']
    if e.error:
        parts.append(f'<div class="meta">detector error: {_esc(e.error)}</div></div>')
        return "".join(parts)
    parts.append(f'<div class="meta">score {e.score:.2f} · confidence {e.confidence:.2f}'
                 f'{" · decisive" if e.decisive else ""} · <code>{_esc(e.detector)}</code></div>'
                 f'{_bar(e.score)}<div>{_esc(e.summary)}</div>')
    figs: List[str] = []
    if e.heatmap is not None and preview is not None and e.heatmap.max() > 0.05:
        figs.append(f'<figure><img src="{jpeg_b64(overlay(preview, e.heatmap), 660)}" alt="">'
                    f'<figcaption>localisation map</figcaption></figure>')
    for name, img in e.visuals.items():
        figs.append(f'<figure><img src="{jpeg_b64(img, 660)}" alt=""><figcaption>'
                    f'{_esc(name.replace("_", " "))}</figcaption></figure>')
    if figs:
        parts.append('<div class="thumbs">' + "".join(figs) + "</div>")
    for name, s in e.series.items():
        svg = _series_svg(s)
        if svg:
            parts.append(f'<div><div class="meta">{_esc(name.replace("_", " "))}</div>{svg}</div>')
    if e.segments:
        rows = "".join(f"<tr><td>{s.start:.2f}–{s.end:.2f}s</td><td>{s.score:.2f}</td>"
                       f"<td>{_esc(s.label)}</td></tr>" for s in e.segments[:20])
        parts.append(f"<table>{rows}</table>")
    if e.details:
        parts.append("<details><summary>raw measurements</summary><pre>"
                     + _esc(json.dumps(e.to_dict()["details"], indent=2, ensure_ascii=False))
                     + "</pre></details>")
    parts.append("</div>")
    return "".join(parts)


def _report_body(r: Report, nested: bool = False) -> str:
    cls = LEVEL_CLASS.get(r.verdict_level, "weak")
    out = []
    out.append('<div class="card"><div class="verdict"><div>'
               f'<div class="badge {cls}">{_esc(r.verdict)}</div>'
               f'<div class="file">{_esc(os.path.basename(r.path))} · {_esc(r.media_type)}'
               f'{" · " + format(r.elapsed, ".1f") + "s" if r.elapsed else ""}</div></div></div>'
               '<div class="scores">')
    for key, label in (("manipulation", "Manipulation"), ("ai_generation", "AI generation"),
                       ("provenance", "Provenance flags")):
        v = r.scores.get(key, 0.0)
        out.append(f'<div class="score"><b>{v * 100:.0f}%</b><span>{label}</span>{_bar(v)}</div>')
    out.append('</div><ul class="why">' + "".join(f"<li>{_esc(x)}</li>" for x in r.explanation)
               + "</ul></div>")

    if r.preview is not None:
        uid = f"v{abs(hash(r.path + r.media_type)) % 10 ** 8}"
        heat = ""
        ctl = ""
        if r.fused_heatmap is not None and r.fused_heatmap.max() > 0.05 and \
                r.scores.get("manipulation", 0) >= 0.2:
            from .utils import colormap

            hm = r.fused_heatmap
            rgba = np.dstack([colormap(hm), (np.clip(hm, 0, 1) ** 0.8 * 255).astype(np.uint8)])
            heat = f'<img class="heat" id="{uid}" src="{png_b64(rgba, 800)}" alt="" style="opacity:.75">'
            ctl = (f'<div class="ctl">fused localisation map <input type="range" min="0" max="100" '
                   f'value="75" oninput="document.getElementById(\'{uid}\').style.opacity=this.value/100">'
                   f'</div>')
        out.append(f'<div class="card"><div class="viewer"><img src="{jpeg_b64(r.preview, 1400)}" '
                   f'alt="analysed media">{heat}</div>{ctl}</div>')
    dur = r.info.get("duration") or r.info.get("duration_sec")
    if r.fused_segments and dur:
        out.append('<div class="card"><h3>Suspicious time ranges</h3>'
                   + _timeline_svg(r.fused_segments, float(dur)) + "<table>"
                   + "".join(f"<tr><td>{s.start:.2f}–{s.end:.2f}s</td><td>{s.score:.2f}</td>"
                             f"<td>{_esc(s.label)}</td></tr>" for s in r.fused_segments[:30])
                   + "</table></div>")

    out.append("<h2>Evidence</h2>")
    ranked = sorted(r.evidences, key=lambda e: (e.error is not None, -(e.score * e.confidence)))
    out.extend(_evidence_html(e, r.preview if r.media_type != "audio" else None) for e in ranked)

    info_rows = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>"
                        for k, v in r.info.items() if v is not None and not isinstance(v, (dict, list)))
    out.append(f'<h2>File</h2><div class="card"><table>{info_rows}</table></div>')
    for c in r.children:
        out.append(f"<h2>Embedded {_esc(c.media_type)} track</h2>" + _report_body(c, True))
    return "".join(out)


def render_html(r: Report) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Basira report — {_esc(os.path.basename(r.path))}</title><style>{CSS}</style>'
            f'</head><body><main><h1>Basira · forensic report</h1>{_report_body(r)}'
            f'<footer>Basira {_esc(r.version)} reports <b>evidence, not proof</b>. Scores are '
            f'calibrated so that 0.5 is each detector\'s decision threshold (vertical tick). '
            f'"No significant traces" does not mean authentic: recompression, resizing and '
            f'careful editing remove the traces these detectors rely on, and no detector here '
            f'can certify that content is real. Treat the result as one input to human judgement.'
            f'</footer></main></body></html>')


def save_html(r: Report, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(r))
    return path
