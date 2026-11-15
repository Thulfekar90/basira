"""The HTML report: self-contained, safe with hostile input, bilingual, within budget."""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import time
from html.parser import HTMLParser

import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import basira
from basira.core import fusion
from basira.core import report_html as RH
from basira.core.io import ffmpeg_exe
from basira.core.report_html import render_html, save_html
from basira.core.types import Category, Evidence, Region, Report, Segment

import synth_audio as A
import synth_image as S
import synth_video as V

VOID = {"meta", "img", "input", "br", "hr", "link", "source", "wbr", "col", "area", "base",
        "embed", "track"}
FORBIDDEN = {"iframe", "object", "embed", "link", "base", "form", "video", "audio", "source",
             "frame", "applet"}
# No network reference of any kind: no URL, no protocol-relative host, no CSS import, and
# no url() other than a data URI.
EXTERNAL = re.compile(r"https?://|//cdn|@import|(?<![a-z])url\((?!\s*['\"]?data:)", re.I)
WHEN = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)


class Doc(HTMLParser):
    """Strict structural check: every non-void element must be closed in order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack, self.problems, self.tags, self.scripts, self.text = [], [], [], [], []
        self.ui = []            # text outside engine-written or file-derived (LTR) spans
        self._script, self._quoted = None, []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag not in VOID:
            self.stack.append(tag)
            if tag in ("bdi", "pre", "code") or "say" in (dict(attrs).get("class") or "").split():
                self._quoted.append(len(self.stack))
        if tag == "script":
            self._script = []

    def handle_startendtag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        if tag == "script" and self._script is not None:
            self.scripts.append("".join(self._script))
            self._script = None
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"</{tag}> closes {self.stack[-3:]}")
        else:
            if self._quoted and self._quoted[-1] == len(self.stack):
                self._quoted.pop()
            self.stack.pop()

    def handle_data(self, data):
        if self._script is not None:
            self._script.append(data)
            return
        self.text.append(data)
        if not self._quoted:
            self.ui.append(data)

    def attrs(self, name):
        return [a[name] for _, a in self.tags if a.get(name) is not None]


def check(html: str) -> Doc:
    """Everything that must hold for *any* report, whatever went into it."""
    assert html.startswith("<!doctype html>")
    html.encode("utf-8")                                   # writable as UTF-8
    assert not EXTERNAL.search(html), EXTERNAL.search(html).group(0)
    doc = Doc()
    doc.feed(html)
    doc.close()
    assert doc.problems == [] and doc.stack == [], (doc.problems[:3], doc.stack)
    # exactly the JSON block and the one static script; nothing else can execute
    assert len(doc.scripts) == 2 and html.count("<script") == 2
    json.loads(doc.scripts[0])
    assert doc.scripts[1] == RH.JS
    names = {t for t, _ in doc.tags}
    assert not names & FORBIDDEN, names & FORBIDDEN
    for tag, attrs in doc.tags:
        assert not [k for k in attrs if k.lower().startswith("on")], (tag, attrs)
        if tag == "img":
            assert attrs.get("src", "").startswith("data:image/"), attrs.get("src", "")[:40]
            assert attrs.get("alt"), "image without a text alternative"
        if tag == "a":
            assert attrs.get("href", "").startswith("#"), attrs
        for k in ("href", "src", "srcset", "action", "formaction", "data", "poster",
                  "xlink:href", "style", "srcdoc"):
            assert "javascript:" not in (attrs.get(k) or "").lower(), (tag, k)
    ids = doc.attrs("id")
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1][:5]
    for ref in ([h[1:] for h in doc.attrs("href")] + doc.attrs("for")
                + doc.attrs("aria-labelledby") + doc.attrs("aria-describedby")
                + doc.attrs("data-layer") + doc.attrs("data-box")):
        assert ref in ids, f"dangling reference to #{ref}"
    return doc


def minimal(media: str, verdict: str, level: int, **kw) -> Report:
    info = {"image": {"format": "JPEG", "width": 64, "height": 48},
            "audio": {"duration": 4.0, "sample_rate": 16000},
            "video": {"duration": 4.0, "width": 64, "height": 48, "fps": 25.0}}[media]
    ev = [Evidence("x.fired", "Fired one", Category.MANIPULATION, 0.9, 0.8, "It fired.",
                   segments=[Segment(1.0, 1.5, 0.9, "cut")] if media != "image" else [],
                   regions=[Region(8, 8, 16, 16, 0.9, "blob")] if media != "audio" else [],
                   heatmap=np.pad(np.ones((16, 16), np.float32), 16) if media != "audio" else None,
                   series={"curve": {"x": [0, 1, 2, 3], "y": [0.0, float("nan"), 2.0, 1.0],
                                     "xlabel": "time (s)", "ylabel": "level",
                                     "markers": [1.0]}}),
          Evidence("x.quiet", "Quiet one", Category.AI_GENERATION, 0.1, 0.6, "Nothing."),
          Evidence.not_applicable("x.na", "Not applicable one", Category.PROVENANCE, "no data"),
          Evidence.failed("x.failed", "Failed one", Category.MANIPULATION, RuntimeError("boom"))]
    r = Report(path=f"/nonexistent/sample.{media}", media_type=media, info=info, evidences=ev,
               scores={"manipulation": 0.8, "ai_generation": 0.0, "provenance": 0.0,
                       "overall": 0.8},
               verdict=verdict, verdict_level=level, explanation=["[Fired one] It fired."],
               preview=np.full((48, 64, 3), 120, np.uint8), version="9.9", **kw)
    if media != "audio":
        r.fused_heatmap = ev[0].heatmap
    if media != "image":
        r.fused_segments = [Segment(1.0, 1.5, 0.9, "cut")]
    return r


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    return str(tmp_path_factory.mktemp("report"))


@pytest.fixture(scope="module")
def real(out):
    """Reports produced by the engines themselves, one per media type and verdict kind."""
    reports = {}
    reports["splice"] = basira.analyze(S.make_splice(out, 2)[0])            # likely manipulated
    reports["copymove"] = basira.analyze(S.make_copymove(out, 3)[0])        # boxes + visuals
    reports["clean"] = basira.analyze(S.make_authentic(out, 3)[0])          # no traces
    reports["sd"] = basira.analyze(S.make_sd_png(out, 12))                  # Q02: unsigned metadata is capped at level 3
    garbage = os.path.join(out, "garbage.mp4")
    with open(garbage, "wb") as f:
        f.write(b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 12)
    reports["garbage"] = basira.analyze(garbage)                            # could not analyse
    clean = A.recording(12.0, seed=0)
    wav = os.path.join(out, "insert.wav")
    sf.write(wav, A.quantize16(A.forge_insert(clean, at=5.0)[0]), A.SR, subtype="PCM_16")
    reports["audio"] = basira.analyze(wav)
    silent = V.encode(V.make_frames(75, 1), os.path.join(out, "silent.mp4"))
    video = os.path.join(out, "with_audio.mp4")
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", silent,
                    "-i", wav, "-c:v", "copy", "-c:a", "aac", "-shortest", video], check=True)
    reports["video"] = basira.analyze(video)
    return reports


# --------------------------------------------------------------------------- #
# Every verdict and media type renders, in both languages
# --------------------------------------------------------------------------- #
def test_real_reports_of_every_kind_render(real):
    assert real["splice"].verdict_level == 3 and real["clean"].verdict_level == 0
    assert real["sd"].verdict_level == 3 and "could not be analysed" in real["garbage"].verdict
    assert real["video"].children, "the video should carry an audio-track child report"
    for name, r in real.items():
        for lang in RH.LANGS:
            doc = check(render_html(r, lang, generated=WHEN))
            text = " ".join(doc.text)
            assert r.evidences[0].summary[:40] in text, (name, lang)
    en = render_html(real["splice"])
    assert "<html" in en and "Evidence" in en and "Likely manipulated" in en


@pytest.mark.parametrize("media", ["image", "audio", "video"])
def test_all_verdict_levels_and_media_types(media):
    for verdict, key in RH.VERDICT_KEYS.items():
        level = int(key[1])
        for lang in RH.LANGS:
            html = render_html(minimal(media, verdict, level), lang, generated=WHEN)
            doc = check(html)
            text = " ".join(doc.text)
            label, means, not_means = RH.STRINGS[lang]["v"][key]
            assert (label or verdict) in text and means in text and not_means in text
            assert f'class="card verdict lv{level}"' in html
            # the four outcome groups, with the quiet and not-applicable ones collapsed
            groups = [a for t, a in doc.tags if t == "details" and a.get("class") == "grp"]
            assert len(groups) == 4 and sum("open" in g for g in groups) == 2
            for state in ("fired", "quiet", "na", "failed"):
                assert RH.STRINGS[lang]["state"][state] in text
    # wording the report does not know still renders, by level
    for level in range(5):
        check(render_html(minimal(media, "Some new wording", level), "ar"))


def test_fusion_wording_is_known_to_the_report():
    """If the fusion step changes a verdict string, the translations must follow."""
    made = {fusion.LEVELS[0], fusion.LEVELS[1], fusion.LEVELS[2],
            fusion.LEVELS[3].format(what="manipulated"),
            fusion.LEVELS[3].format(what="AI-generated"),
            fusion.LEVELS[4].format(What="AI-generated"), fusion.LEVELS[4].format(What="Edited")}
    assert made <= set(RH.VERDICT_KEYS), made - set(RH.VERDICT_KEYS)


def test_media_specific_parts(real):
    image = check(render_html(real["copymove"]))
    classes = " ".join(image.attrs("class"))
    assert "stage" in classes and "layer l0 dflt" in classes and "boxes" in classes
    assert any(t == "input" and a.get("type") == "checkbox" for t, a in image.tags)
    assert any(t == "input" and a.get("class") == "split" for t, a in image.tags)
    assert "clone source" in " ".join(image.text) and "of the image" in " ".join(image.text)
    audio = check(render_html(real["audio"]))
    classes = " ".join(audio.attrs("class"))
    assert "spec zoomable" in classes and "lane all" in classes and "psvg" in classes
    assert "seg" in audio.attrs("class") and "Suspicious time ranges" in " ".join(audio.text)
    video = check(render_html(real["video"]))
    assert "Embedded audio track" in " ".join(video.text)
    assert "stage" in " ".join(video.attrs("class")) and "lane all" in video.attrs("class")


def test_every_detector_has_an_explanation_in_both_languages(real):
    for det, d in RH.DETECTORS.items():
        assert len(d["en"]) == 2 and len(d["ar"]) == 3 and all(d["en"]) and all(d["ar"]), det
    seen = {e.detector for r in real.values() for e in r.evidences}
    seen |= {e.detector for r in real.values() for c in r.children for e in c.evidences}
    assert seen <= set(RH.DETECTORS), f"no explanation for {sorted(seen - set(RH.DETECTORS))}"
    doc = check(render_html(real["splice"]))
    assert RH.DETECTORS["image.jpeg_dq"]["en"][1] in " ".join(doc.text)


# --------------------------------------------------------------------------- #
# Self-contained
# --------------------------------------------------------------------------- #
def test_no_external_reference_and_strict_csp(real):
    for r in real.values():
        html = render_html(r)
        assert not EXTERNAL.search(html)
        head = html.split("</head>")[0]
        m = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)">', head)
        assert m and head.index(m.group(0)) < head.index("<style>") < html.index("<script")
        csp = m.group(1)
        assert "default-src 'none'" in csp and "img-src data:" in csp
        assert "base-uri 'none'" in csp and "form-action 'none'" in csp
        # the only script allowed to run is the static one, identified by its hash
        digest = base64.b64encode(hashlib.sha256(RH.JS.encode()).digest()).decode()
        assert f"script-src 'sha256-{digest}'" in csp
        assert "unsafe-eval" not in csp and "script-src 'unsafe-inline'" not in csp
    assert "</script" not in RH.JS and "</style" not in RH.CSS and not EXTERNAL.search(RH.CSS)


def test_content_is_in_the_markup_not_built_by_script(real):
    """With JavaScript disabled the verdict, map, boxes, findings and hash are all there."""
    html = render_html(real["splice"])
    static = html[:html.index("<script")]
    for needle in ("Likely manipulated", "What it does not mean", 'class="layer l0 dflt"',
                   "<rect", "single-compressed block cluster", "SHA-256", "image.jpeg_dq",
                   real["splice"].get("image.jpeg_dq").summary[:50]):
        assert needle in static, needle
    assert "document.write" not in RH.JS and "innerHTML" not in RH.JS and "eval(" not in RH.JS


# --------------------------------------------------------------------------- #
# "How the verdict was reached": the diagram is drawn from the report, nothing else
# --------------------------------------------------------------------------- #
def _flow_section(html: str) -> str:
    start = html.index('<section id="flow')
    return html[start:html.index("</section>", start)]


@pytest.mark.parametrize("name", ["splice", "audio", "video", "clean", "garbage"])
def test_flow_diagram_reflects_the_measurements(real, name):
    r = real[name]
    html = render_html(r)
    check(html)
    flow = _flow_section(html)
    assert flow in html[:html.index("<script")], "the diagram must not be built by script"
    assert RH.STRINGS["en"]["flow"] in flow and 'aria-hidden="true"' in flow
    # one node per detector and per embedded track, each in its outcome's class
    nodes = re.findall(r'<g class="nd (fired|quiet|na|failed)">', flow)
    assert len(nodes) == len(r.evidences) + len(r.children)
    states = [RH._state(e) for e in r.evidences]
    assert sorted(nodes[:len(r.evidences)]) == sorted(states)
    # every detector above its threshold shows its real weight; quiet ones add nothing
    labels = re.findall(r'class="elab"[^>]*>\+([\d.]+)<', flow)
    expected = [f"{RH._weight(e):.2f}" for e in r.evidences
                if RH._state(e) == "fired" and RH._weight(e) >= 0.005]
    if not r.children:
        assert sorted(labels) == sorted(expected)
    else:
        assert set(expected) <= set(labels)
    # category nodes print the fused scores, the verdict node the level
    for cat in ("manipulation", "ai_generation", "provenance"):
        assert f'>{r.scores.get(cat, 0) * 100:.0f}%</text>' in flow
    assert f'class="vd lv{r.verdict_level} an"' in flow
    # the same path in words, with the real counts
    counts = {s: states.count(s) for s in ("fired", "quiet", "na", "failed")}
    assert RH.STRINGS["en"]["flow_s2"].format(total=len(r.evidences), **counts) in flow
    # nodes link to their detector cards and to the embedded track
    targets = re.findall(r'<a href="#([^"]+)" tabindex="-1">', flow)
    assert len(targets) == len(r.evidences) + len(r.children)


def test_flow_replay_is_wired_and_reading_order_follows_the_language(real):
    html = render_html(real["video"])
    section = _flow_section(html)
    # the script finds the button through the card's parent: both sit in the same section
    assert 'class="btn js-only replay"' in section and 'class="card flow"' in section
    assert "f.parentNode.querySelector('.replay')" in RH.JS
    assert "prefers-reduced-motion:no-preference){\n.flow.play" in RH.CSS
    en = _flow_section(html)
    ar = _flow_section(render_html(real["video"], "ar"))
    assert 'direction="rtl"' not in en and 'direction="rtl"' in ar
    # numbers stay left-to-right inside the Arabic diagram ("+0.80", never "0.80+")
    assert not re.search(r'<text class="(n|big n|elab)"[^>]*direction="rtl"', ar)
    name = RH._esc(os.path.basename(real["video"].path))
    assert re.search(r'<text class="sm" [^>]*text-anchor="end">' + re.escape(name), ar)
    # the diagram is mirrored, not merely translated: the file column is on the right
    x_en = float(re.search(r'<g class="an" style="--d:0s"><rect class="box" x="([\d.]+)"',
                           en).group(1))
    x_ar = float(re.search(r'<g class="an" style="--d:0s"><rect class="box" x="([\d.]+)"',
                           ar).group(1))
    assert x_en < RH.FLOW_W / 2 < x_ar


# --------------------------------------------------------------------------- #
# Hostile input
# --------------------------------------------------------------------------- #
def nasty(tag: str) -> str:
    return (f'{tag}"><img src=x onerror=alert(1)></script><script>alert("{tag}")</script>'
            f"<!--\u2028'\u202e https://evil.example/{tag}.js javascript:alert(1) &amp; \x00\x1b")


def hostile_report(nasty=nasty, media: str = "video") -> Report:
    ev = [
        Evidence(nasty("DETID"), nasty("TITLE"), Category.MANIPULATION, 0.95, 0.9,
                 nasty("SUMMARY"), details={nasty("DKEY"): nasty("DVAL"), "n": [nasty("DLIST")]},
                 heatmap=np.pad(np.ones((8, 8), np.float32), 12),
                 segments=[Segment(0.5, 1.0, 0.9, nasty("SEGLABEL"))],
                 regions=[Region(2, 2, 9, 9, 0.9, nasty("REGLABEL"))],
                 visuals={nasty("VISUAL"): np.zeros((8, 8, 3), np.uint8)},
                 series={nasty("SERIES"): {"x": [0, 1, 2], "y": [1, 3, 2],
                                           "xlabel": nasty("XLABEL"), "ylabel": nasty("YLABEL"),
                                           "markers": [1, "x", None]}}),
        Evidence("q", nasty("QTITLE"), Category.PROVENANCE, 0.1, 0.5, nasty("QSUMMARY")),
        Evidence("e", nasty("ETITLE"), Category.AI_GENERATION, 0.0, 0.0, nasty("ESUMMARY"),
                 error=nasty("ERROR")),
    ]
    child = Report(path=nasty("CHILDPATH"), media_type="audio",
                   info={"duration": 2.0, nasty("CKEY"): nasty("CVAL")}, evidences=ev[1:],
                   verdict=nasty("CVERDICT"), verdict_level=2,
                   explanation=[nasty("CEXPL")],
                   fused_segments=[Segment(0.1, 0.2, 0.5, nasty("CSEG"))])
    return Report(path="/nonexistent/" + nasty("PATH").replace("/", "\u2215") + ".jpg",
                  media_type=media,
                  info={"duration": 2.0, "width": 32, "height": 32, nasty("IKEY"): nasty("IVAL"),
                        "codec_hints": [nasty("ILIST")], "format": nasty("FORMAT")},
                  evidences=ev, scores={"manipulation": 0.9}, verdict=nasty("VERDICT"),
                  verdict_level=3,
                  explanation=[f"[{nasty('EXTAG')}] {nasty('EXPL')}", nasty("EXPLAIN2")],
                  fused_heatmap=ev[0].heatmap, fused_segments=ev[0].segments,
                  preview=np.zeros((32, 32, 3), np.uint8), children=[child],
                  version=nasty("VERSION"))


@pytest.mark.parametrize("lang", RH.LANGS)
def test_hostile_strings_are_inert_in_every_context(lang):
    r = hostile_report()
    html = render_html(r, lang)
    doc = check(html)                       # no new tags, handlers, scripts or URLs
    # nothing hostile became markup: the element and attribute structure is exactly that
    # of the same report filled with harmless strings
    plain = check(render_html(hostile_report(lambda tag: tag), lang))
    assert [(t, sorted(a)) for t, a in doc.tags] == [(t, sorted(a)) for t, a in plain.tags]
    assert not any(t in ("img", "script") and "x" == a.get("src") for t, a in doc.tags)
    for ch in "\u2028\u2029\u202e\x00\x1b":
        assert ch not in html
    assert "[U+202E]" in html               # the bidi override is shown, not obeyed
    # ...and every field is still displayed, as text
    text = " ".join(doc.text) + " ".join(v or "" for _, a in doc.tags for v in a.values())
    for tag in ("PATH", "VERDICT", "EXTAG", "EXPL", "EXPLAIN2", "DETID", "TITLE", "SUMMARY",
                "DKEY", "DVAL", "DLIST", "SEGLABEL", "REGLABEL", "VISUAL", "SERIES", "XLABEL",
                "YLABEL", "QTITLE", "QSUMMARY", "ETITLE", "ERROR", "IKEY", "IVAL", "ILIST",
                "FORMAT", "CVERDICT", "CEXPL", "CSEG", "CKEY", "CVAL", "VERSION"):
        q = '\\"' if tag in ("DKEY", "DVAL", "DLIST") else '"'    # raw measurements are JSON
        assert f"{tag}{q}><img src=x onerror=alert(1)>" in text, f"{tag} is not shown"
        assert f'{tag}"><img' not in html and f'{tag}\\"><img' not in html
        assert f"{tag}{q[:-1]}&quot;&gt;&lt;img" in html
    # JSON context: the block ends where it should and round-trips the hostile values
    summary = json.loads(doc.scripts[0])
    assert summary["verdict"].startswith('VERDICT"><img src=x onerror=alert(1)></script>')
    assert summary["verdict"] == r.verdict and "https://evil.example" in summary["verdict"]
    assert summary["file"].startswith('PATH"><img')
    raw = html[html.index('id="basira-summary">'):]
    raw = raw[:raw.index("</script>")]
    assert "<" not in raw.split(">", 1)[1] and "\\u003c/script\\u003e" in raw


def test_hostile_file_name_and_metadata_from_a_real_file(out):
    name = '"><img src=x onerror=alert(1)>.jpg'
    exif = Image.Exif()
    exif[0x0131] = "</script><script>alert(2)</script>"          # Software
    exif[0x010F] = '"onmouseover="alert(3)'                      # Make
    exif[0x0110] = "x\u2028y https://evil.example/a.js"          # Model
    path = os.path.join(out, name)
    Image.fromarray(S.camera_photo(1, 240, 320)).save(path, "JPEG", quality=90, exif=exif)
    r = basira.analyze(path)
    assert "alert(2)" in json.dumps(r.to_dict()), "the metadata should reach the report"
    meta = PngInfo()
    meta.add_text("parameters", "</script><script>alert(4)</script>\nNegative prompt: "
                  "<img src=x onerror=alert(5)>\nSteps: 30, Sampler: Euler a, CFG scale: 7, "
                  'Seed: 1, Size: 512x512, Model: "><svg onload=alert(6)>')
    png = os.path.join(out, "'><svg onload=alert(7)><script>.png")
    Image.fromarray(S.camera_photo(2, 240, 320)).save(png, pnginfo=meta)
    r2 = basira.analyze(png)
    assert r2.verdict_level == 3 and "alert(4)" in json.dumps(r2.to_dict())
    for rep in (r, r2):
        for lang in RH.LANGS:
            html = render_html(rep, lang)
            doc = check(html)
            assert os.path.basename(rep.path) in " ".join(doc.text)
            assert json.loads(doc.scripts[0])["file"] == os.path.basename(rep.path)
            assert not re.search(r"<(img|svg|script)[^>]*(onerror|onload|alert)", html)
            assert "<title>" in html and "<img src=x" not in html
            assert "&lt;/script&gt;&lt;script&gt;alert(" in html


def test_undecodable_and_bidi_file_names(tmp_path):
    r = minimal("image", "Suspicious", 2)
    r.path = "/nonexistent/photo\u202egpj.exe \udcff\ud83d.jpg"
    html = render_html(r)
    check(html)
    assert "photo[U+202E]gpj.exe" in html and "\u202e" not in html
    p = save_html(r, str(tmp_path / "r.html"), "ar")           # must be writable as UTF-8
    assert 'dir="rtl"' in open(p, encoding="utf-8").read()


# --------------------------------------------------------------------------- #
# Arabic
# --------------------------------------------------------------------------- #
def _same_shape(a, b, where="STRINGS"):
    assert type(a) is type(b) or {type(a), type(b)} <= {str, type(None)}, where
    if isinstance(a, dict) and where != "STRINGS.fusion":
        assert set(a) == set(b), (where, set(a) ^ set(b))
        for k in a:
            _same_shape(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _same_shape(x, y, f"{where}[{i}]")


def test_arabic_dictionary_is_complete():
    _same_shape(RH.STRINGS["en"], RH.STRINGS["ar"])
    arabic = re.compile("[\u0600-\u06ff]")

    def walk(v, where):
        if isinstance(v, str):
            assert arabic.search(v), f"{where} is not translated: {v!r}"
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{where}.{k}")
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                walk(x, f"{where}[{i}]")

    walk({k: v for k, v in RH.STRINGS["ar"].items() if k != "fusion"}, "ar")
    walk(list(RH.STRINGS["ar"]["fusion"].values()), "ar.fusion")
    # placeholders survive translation
    for k, v in RH.STRINGS["en"].items():
        if isinstance(v, str):
            assert set(re.findall(r"{(\w+)}", v)) == set(re.findall(r"{(\w+)}",
                                                                    RH.STRINGS["ar"][k])), k


def test_arabic_report_is_right_to_left_with_arabic_labels(real):
    for name in ("splice", "audio", "video", "clean"):
        r = real[name]
        html = render_html(r, "ar")
        doc = check(html)
        assert '<html lang="ar" dir="rtl"' in html
        text = " ".join(doc.ui)
        ar = RH.STRINGS["ar"]
        for key in ("verdict", "evidence", "means", "not_means", "details", "howto",
                    "disclaimer", "overview", "measures", "blind"):
            assert ar[key] in text, key
        assert ar["v"][RH.VERDICT_KEYS[r.verdict]][0] in text
        for key, english in RH.STRINGS["en"].items():       # no English interface text left
            if isinstance(english, str) and len(english) > 3 and "{" not in english \
                    and english not in ar[key]:                 # (terms such as "SHA-256" stay)
                assert english not in text, english
        # engine-written English stays as it is, isolated and left-to-right
        fired = [e for e in r.evidences if e.score >= 0.5 and e.confidence > 0 and not e.error]
        for e in fired[:2] or r.evidences[:1]:
            assert f'<span class="say" dir="ltr" lang="en">{RH._esc(e.summary)}</span>' in html
        # file name, hash and numbers are isolated left-to-right inside the RTL page
        assert f'<bdi dir="ltr">{RH._esc(os.path.basename(r.path))}</bdi>' in html
        assert re.search(r'<code class="hash mono" dir="ltr">[0-9a-f]{64}</code>', html)
        assert re.search(r'<bdi dir="ltr">\d+%</bdi>', html)
        # the image, timeline and plots are not mirrored
        for cls in ("viewer", "tl", "pbox"):
            for t, a in doc.tags:
                if a.get("class") == cls:
                    assert a.get("dir") == "ltr", cls
    assert RH.DETECTORS["image.jpeg_dq"]["ar"][0] in render_html(real["splice"], "ar")


def test_language_argument():
    r = minimal("image", "Suspicious", 2)
    assert 'lang="ar"' in render_html(r, lang="ar-SA") and 'lang="en"' in render_html(r, "EN")
    with pytest.raises(ValueError, match="unsupported report language"):
        render_html(r, lang="xx")


# --------------------------------------------------------------------------- #
# Header: hash, size, date, version
# --------------------------------------------------------------------------- #
def test_header_identifies_the_file(real, out, tmp_path):
    r = real["splice"]
    digest = hashlib.sha256(open(r.path, "rb").read()).hexdigest()
    html = render_html(r, generated=WHEN)
    doc = check(html)
    assert digest in html and f"{os.path.getsize(r.path):,}" in html
    assert "2026-01-02 03:04:05 UTC+0000" in html and f"Basira {r.version}" in html
    summary = json.loads(doc.scripts[0])
    assert summary["sha256"] == digest and summary["verdict_level"] == 3
    assert summary["size"] == os.path.getsize(r.path) and summary["tool"] == "basira"

    # the analysed file is gone (server uploads are deleted): say so, or accept the digest
    gone = str(tmp_path / "gone.jpg")
    os.replace(S.make_authentic(str(tmp_path), 5)[0], gone)
    r2 = basira.analyze(gone)
    os.remove(gone)
    html = render_html(r2)
    check(html)
    assert RH.STRINGS["en"]["sha_missing"] in html and json.loads(
        check(html).scripts[0])["sha256"] is None
    assert f'{r2.info["bytes"]:,}' in html                    # size falls back to the engine's
    html = render_html(r2, sha256="ab" * 32, file_size=1234)
    assert "ab" * 32 in html and "1,234" in html and RH.STRINGS["en"]["sha_missing"] not in html
    with pytest.raises(ValueError):
        render_html(r2, sha256='"><script>')


# --------------------------------------------------------------------------- #
# Accessibility and print
# --------------------------------------------------------------------------- #
def test_structure_is_accessible(real):
    for name in ("copymove", "video"):
        html = render_html(real[name])
        doc = check(html)                  # includes: unique ids, no dangling label/aria refs
        tags = [t for t, _ in doc.tags]
        assert tags.count("h1") == 1 and tags.count("main") == 1 and "nav" in tags
        assert "header" in tags and "footer" in tags
        levels = [int(t[1]) for t in tags if re.fullmatch(r"h[1-6]", t)]
        assert all(b - a <= 1 for a, b in zip(levels, levels[1:])), "heading level skipped"
        # every control has a name
        for t, a in doc.tags:
            if t == "input":
                assert a.get("id") in doc.attrs("for") or a.get("class", "").startswith("lt"), a
            if a.get("role") in ("meter", "img", "group"):
                assert a.get("aria-label"), a
        for t, a in doc.tags:
            if t == "th":
                assert a.get("scope") in ("col", "row")
    css = RH.CSS
    assert ":focus-visible" in css and "prefers-reduced-motion" in css
    assert "prefers-color-scheme:dark" in css and "forced-colors" in css
    assert "@media print" in css and "print-color-adjust:exact" in css


def test_text_alternative_for_the_map(real):
    doc = check(render_html(real["splice"]))
    alts = [a.get("alt", "") for t, a in doc.tags if t == "img"]
    assert any(re.search(r"Fused localisation map: the highlighted area covers about "
                         r"[\d.]+% of the image, centred on the bottom right", a) for a in alts)
    assert re.search(r"bottom right of the image, \d+×\d+ px at \(\d+, \d+\)",
                     " ".join(doc.text))


def _contrast(fg: str, bg: str) -> float:
    def lum(c):
        v = [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        v = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
    a, b = sorted((lum(fg), lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def test_text_colours_meet_wcag_aa_in_light_and_dark():
    blocks = re.findall(r":root\{([^}]*)\}", RH.CSS)
    light, dark = blocks[0], blocks[1]
    for block in (light, dark):
        v = dict(re.findall(r"--([\w-]+):(#[0-9a-fA-F]{6})", block))
        for text in ("ink", "ink2", "accent", "l0", "l1", "l2", "l3", "l4"):
            for surface in ("surface", "bg", "sunk"):
                assert _contrast(v[text], v[surface]) >= 4.5, (text, surface)
        for i in range(5):                                   # verdict text on its tint
            assert _contrast(v["ink"], v[f"t{i}"]) >= 7, i


# --------------------------------------------------------------------------- #
# Size and speed
# --------------------------------------------------------------------------- #
def test_size_and_speed_budget(real):
    for name in ("splice", "copymove", "clean", "sd", "audio", "video"):
        size = len(render_html(real[name]).encode("utf-8"))
        assert size < 1_500_000, (name, size)
    # a large, detailed photograph with several maps and visuals stays within budget too
    rng = np.random.RandomState(0)
    big = rng.randint(0, 255, (1200, 1600, 3)).astype(np.uint8)
    heat = np.zeros((2400, 3200), np.float32)
    heat[600:1200, 800:1800] = 1.0
    r = minimal("image", "Likely manipulated", 3)
    r.preview, r.fused_heatmap, r.info = big, heat, {"width": 3200, "height": 2400}
    for i in range(4):
        r.evidences.append(Evidence(f"x.map{i}", f"Map {i}", Category.MANIPULATION, 0.9, 0.9,
                                    "fired", heatmap=heat, visuals={"visual": big}))
    html = render_html(r)
    check(html)
    assert len(html.encode("utf-8")) < 1_500_000
    # the analysed image is embedded once; maps are small indexed PNGs, not copies of it
    assert html.count("data:image/jpeg") == 1 + 4 and html.count("data:image/png") == 1 + 5
    best = min(_timed(render_html, real["splice"]) for _ in range(3))
    assert best < 1.0, f"rendering took {best:.2f} s"


def _timed(fn, *args) -> float:
    """CPU seconds, so that a busy machine does not fail the budget."""
    t = time.process_time()
    fn(*args)
    return time.process_time() - t


# --------------------------------------------------------------------------- #
# Upload page
# --------------------------------------------------------------------------- #
def test_upload_page_template():
    from basira.server import PAGE

    # exactly three substitutions (CSS, version, token); every other percent sign is doubled
    assert PAGE.count("%s") == 3 and "%" not in PAGE.replace("%%", "").replace("%s", "")
    page = PAGE % (RH.CSS, "1.2.3", "the-token")
    doc = Doc()
    doc.feed(page)
    assert doc.problems == [] and doc.stack == []
    assert not EXTERNAL.search(page)
    assert page.index(RH.CSS) < page.index("Basira 1.2.3") < page.index("the-token")
    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', page).group(1)
    assert "default-src 'none'" in csp and "connect-src 'self'" in csp
    assert "Drop an image" in page and "Nothing leaves this computer" in page
    assert "لا شيء يغادر هذا الحاسوب" in page and 'data-lang="ar"' in page
    assert 'role="alert"' in page and 'role="progressbar"' in page
    # the report is shown in a sandbox that cannot reach the server or the upload token
    frame = [a for t, a in doc.tags if t == "iframe"]
    assert len(frame) == 1 and "allow-same-origin" not in frame[0]["sandbox"]
