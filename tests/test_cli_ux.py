"""Command-line behaviour people script against: inputs, output formats, exit status,
errors, the worker pool, ``basira doctor`` and the stable JSON layout (docs/JSON.md).

Most tests replace the analysis with a stub so they run in milliseconds; the ones marked
"real" run the actual engines on generated files.
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import synth_image as S

import basira
from basira import cli
from basira.core.types import SCHEMA_VERSION, Category, Evidence, Report, Segment, _jsonable

VERDICTS = {0: "No significant traces found", 3: "Likely manipulated"}


def fake_analyze(path, **_):
    """Stand-in for basira.analyze: 'bad' in the name means a level-3 verdict."""
    name = os.path.basename(path)
    if "boom" in name:
        raise RuntimeError("decoder exploded\nwith a second line")
    with open(path, "rb") as f:
        if f.read(5) == b"hello":
            raise ValueError(f"Unrecognised media type: {name}")
    level = 3 if "bad" in name else 0
    score = 0.8 if level else 0.0
    return Report(path=path, media_type="image", info={"detectors_failed": 0},
                  scores={"manipulation": score, "ai_generation": 0.0, "provenance": 0.0,
                          "overall": score},
                  verdict=VERDICTS[level], verdict_level=level, explanation=["stub"],
                  elapsed=0.01, version=basira.__version__)


def fake_save_html(report, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("<html>stub</html>")
    return path


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(cli, "analyze", fake_analyze)
    monkeypatch.setattr(cli, "save_html", fake_save_html)
    monkeypatch.delenv("NO_COLOR", raising=False)


@pytest.fixture
def tree(tmp_path):
    """folder/ with two media files, two non-media files and a sub-folder."""
    d = tmp_path / "folder"
    (d / "sub").mkdir(parents=True)
    (d / "clean.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 64)
    (d / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
    (d / "notes.txt").write_text("hello")
    (d / "data.csv").write_text("a,b\n")
    (d / ".hidden.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 64)
    (d / "sub" / "bad_deep.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 64)
    return d


# ----------------------------------------------------------------- stable JSON (real)

def _assert_strict_json(obj):
    """Only JSON types, str keys, finite numbers."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert type(k) is str
            _assert_strict_json(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_strict_json(v)
    elif isinstance(obj, float):
        assert np.isfinite(obj)
    else:
        assert obj is None or type(obj) in (str, int, bool), type(obj)


def _media_files(folder):
    import cv2
    import soundfile as sf

    img, _ = S.make_authentic(str(folder), 4)
    rs = np.random.RandomState(0)
    t = np.arange(3 * 16000) / 16000.0
    wav = str(folder / "tone.wav")
    sf.write(wav, (0.1 * np.sin(2 * np.pi * 220 * t) + 0.01 * rs.randn(t.size)).astype("float32"),
             16000)
    vid = str(folder / "clip.avi")
    vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 120))
    base = rs.randint(0, 255, (120, 160, 3)).astype(np.uint8)
    for i in range(50):
        vw.write(np.roll(base, i, axis=1))
    vw.release()
    return [("image", img), ("audio", wav), ("video", vid)]


def test_every_report_is_strict_json_with_schema_version(tmp_path):
    for kind, path in _media_files(tmp_path):
        if kind == "video" and not (os.path.exists(path) and os.path.getsize(path) > 1000):
            continue          # this OpenCV build cannot write MJPG/AVI
        d = basira.analyze(path).to_dict()
        text = json.dumps(d, allow_nan=False)            # the documented promise
        assert json.loads(text) == d
        _assert_strict_json(d)
        assert d["tool"] == "basira" and d["schema_version"] == SCHEMA_VERSION == 1
        assert d["media_type"] == kind and d["version"] == basira.__version__
        assert isinstance(d["info"]["detectors_failed"], int)
        assert isinstance(d["info"]["children_failed"], int)
        assert set(d) == {"tool", "schema_version", "version", "path", "media_type", "info",
                          "verdict", "verdict_level", "scores", "explanation", "segments",
                          "evidences", "children", "elapsed_sec"}
        for e in d["evidences"]:
            assert set(e) == {"detector", "title", "category", "score", "confidence",
                              "summary", "decisive", "details", "segments", "regions",
                              "has_heatmap", "error"}
            assert 0.0 <= e["score"] <= 1.0 and 0.0 <= e["confidence"] <= 1.0
            assert e["category"] in ("manipulation", "ai_generation", "provenance")
        for c in d["children"]:
            assert c["schema_version"] == 1


def test_non_finite_and_exotic_values_never_reach_json():
    from fractions import Fraction

    r = Report(path="x.jpg", media_type="image", info={"ratio": float("inf")},
               scores={"manipulation": 0.25}, verdict="v",
               fused_segments=[Segment(0.0, float("nan"), 0.5, "s")])
    r.evidences.append(Evidence("d.nan", "t", Category.MANIPULATION, float("nan"), 1.0, "s"))
    r.evidences.append(Evidence(
        "d.odd", "t", Category.PROVENANCE, 0.1, 1.0, "s",
        details={"nan": np.float32("nan"), "arr": np.array([1.0, np.inf]), "set": {2, 1},
                 "raw": b"\xff\x00", "frac": Fraction(1, 4), "np_int": np.int64(7),
                 "np_bool": np.bool_(True), 5: "int key", "obj": object(),
                 "big": np.zeros((40, 40)), "cat": Category.AI_GENERATION}))
    d = r.to_dict()
    json.dumps(d, allow_nan=False)
    _assert_strict_json(d)
    assert d["info"] == {"ratio": None, "detectors_failed": 1, "children_failed": 0}
    assert d["segments"][0]["end"] is None
    assert d["evidences"][0]["score"] == 0.0 and d["evidences"][0]["error"]
    det = d["evidences"][1]["details"]
    assert det["nan"] is None and det["arr"] == [1.0, None] and det["set"] == [1, 2]
    assert det["frac"] == 0.25 and det["np_int"] == 7 and det["np_bool"] is True
    assert det["5"] == "int key" and det["big"] == "<array (40, 40)>"
    assert det["cat"] == "ai_generation" and type(det["cat"]) is str
    assert _jsonable(float("-inf")) is None
    # "0.5 is the threshold" must survive rounding: just below stays below
    quiet = Evidence("d", "t", Category.MANIPULATION, 0.49999999999999994, 1.0, "s")
    assert quiet.to_dict()["score"] < 0.5
    assert Evidence("d", "t", Category.MANIPULATION, 0.5, 1.0, "s").to_dict()["score"] == 0.5


def test_json_reference_matches_the_code(tmp_path):
    """docs/JSON.md documents every field and carries a generated example: both must
    have exactly the fields the code produces."""
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "docs", "JSON.md"), encoding="utf-8") as f:
        doc = f.read()
    m = re.search(r"<!-- example:begin -->.*?```json\n(.*?)\n```.*?<!-- example:end -->",
                  doc, re.S)
    assert m, "docs/JSON.md has no generated example"
    example = json.loads(m.group(1))
    path, _ = S.make_authentic(str(tmp_path), 6)
    live = basira.analyze(path).to_dict()
    assert set(example) == set(live)
    assert example["schema_version"] == live["schema_version"] == SCHEMA_VERSION
    assert {k for e in example["evidences"] for k in e} == {k for e in live["evidences"] for k in e}
    assert set(example["scores"]) == set(live["scores"])
    tables = "\n".join(ln for ln in doc.splitlines() if ln.startswith("|"))
    for key in list(live) + list(live["evidences"][0]) + list(live["scores"]) + \
            ["detectors_failed", "children_failed", "start", "end", "label", "x", "y", "w", "h",
             "frame"]:
        assert f"`{key}`" in tables, f"{key} is not documented in docs/JSON.md"
    for key in live["info"]:
        assert f"`{key}`" in tables, f"info.{key} is not documented in docs/JSON.md"
    for kind in ("not_found", "unreadable", "unsupported", "too_large", "failed", "timeout",
                 "crashed"):
        assert f"`{kind}`" in tables
    with open(os.path.join(root, "basira", "cli.py"), encoding="utf-8") as f:
        used = set(re.findall(r'_fail\(\s*item,\s*"([a-z_]+)"', f.read()))
    assert used <= {"not_found", "unreadable", "unsupported", "too_large", "failed", "timeout",
                    "crashed"}, used


# ----------------------------------------------------------------- inputs

def test_folder_scan_skips_other_files_with_one_note(stub, tree, capsys):
    code = cli.main(["scan", str(tree), "--no-html", "--json"])
    cap = capsys.readouterr()
    out = json.loads(cap.out)
    assert code == 0
    assert sorted(os.path.basename(r["path"]) for r in out) == ["bad.png", "clean.jpg"]
    notes = [ln for ln in cap.err.splitlines() if ln]
    assert len(notes) == 1 and "skipped 2 unsupported files" in notes[0]


def test_recursive_and_report_dir_keep_the_folder_layout(stub, tree, tmp_path, capsys):
    rep = tmp_path / "reports"
    assert cli.main(["scan", str(tree), "--recursive", "--report-dir", str(rep), "-q"]) == 0
    assert capsys.readouterr().out == ""
    made = sorted(os.path.relpath(os.path.join(d, f), rep).replace(os.sep, "/")
                  for d, _, fs in os.walk(rep) for f in fs)
    assert made == ["bad.png.basira.html", "clean.jpg.basira.html",
                    "sub/bad_deep.jpg.basira.html"]


def test_pattern_is_expanded_when_the_shell_did_not(stub, tree, capsys):
    code = cli.main(["scan", os.path.join(str(tree), "*"), "--no-html", "--jsonl"])
    lines = capsys.readouterr().out.splitlines()
    # like a shell would: the two media files, plus the files of the matched sub-folder
    assert code == 0 and len(lines) == 3 and all(json.loads(x)["tool"] == "basira" for x in lines)
    assert cli.main(["scan", os.path.join(str(tree), "*.nope"), "--no-html", "-q"]) == 1
    assert "no files match" in capsys.readouterr().err


def test_empty_folder_is_a_note_not_an_error(stub, tmp_path, capsys):
    (tmp_path / "empty" / "sub").mkdir(parents=True)
    assert cli.main(["scan", str(tmp_path / "empty"), "--no-html"]) == 0
    err = capsys.readouterr().err
    assert "no media files found" in err and "--recursive" in err


def test_same_file_given_twice_is_analysed_once(stub, tree, capsys):
    p = str(tree / "clean.jpg")
    cli.main(["scan", p, p, str(tree), "--no-html", "--json"])
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_reports_of_same_named_files_do_not_overwrite_each_other(stub, tree, tmp_path):
    (tree / "sub" / "clean.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 64)
    rep = tmp_path / "r"
    cli.main(["scan", str(tree / "clean.jpg"), str(tree / "sub" / "clean.jpg"),
              "--report-dir", str(rep), "-q"])
    assert sorted(os.listdir(rep)) == ["clean.jpg-2.basira.html", "clean.jpg.basira.html"]


# ----------------------------------------------------------------- output formats

def test_json_is_one_object_for_one_file_and_an_array_otherwise(stub, tree, capsys):
    one = str(tree / "clean.jpg")
    cli.main(["scan", one, "--json", "--no-html"])
    assert json.loads(capsys.readouterr().out)["path"] == one
    code = cli.main(["scan", one, str(tree / "notes.txt"), str(tree / "missing.jpg"),
                     "--json", "--no-html"])
    cap = capsys.readouterr()
    out = json.loads(cap.out)                      # nothing but JSON on stdout
    assert code == 1 and [("error" in r) for r in out] == [False, True, True]
    assert out[1]["error"]["kind"] == "unsupported" and out[2]["error"]["kind"] == "not_found"
    assert all(r["schema_version"] == 1 and r["tool"] == "basira" for r in out)
    # a single input that fails is still a single (error) object, and still valid JSON
    assert cli.main(["scan", str(tree / "missing.jpg"), "--json", "--no-html"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["kind"] == "not_found"


def test_jsonl_has_one_compact_object_per_input(stub, tree, capsys):
    code = cli.main(["scan", str(tree), "-r", str(tree / "nope.jpg"), "--jsonl", "--no-html"])
    lines = capsys.readouterr().out.splitlines()
    assert code == 1 and len(lines) == 4
    recs = [json.loads(x) for x in lines]
    assert sum("error" in r for r in recs) == 1 and all("\n" not in x for x in lines)


def test_csv_summary(stub, tree, tmp_path, monkeypatch):
    evil = tree / "=1+1.jpg"
    evil.write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 64)
    monkeypatch.chdir(tree)
    out = tmp_path / "out" / "summary.csv"
    code = cli.main(["scan", str(tree / "bad.png"), "=1+1.jpg", str(tree / "notes.txt"),
                     "--no-html", "--csv", str(out), "-q"])
    assert code == 1
    assert out.read_bytes()[:3] == b"\xef\xbb\xbf"          # BOM: Excel reads UTF-8 names
    with open(out, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == cli.CSV_COLUMNS and len(rows) == 3
    assert rows[0]["verdict_level"] == "3" and rows[0]["manipulation"] == "0.8"
    assert rows[1]["path"] == "'=1+1.jpg"                   # not runnable as a formula
    assert rows[2]["verdict"] == "" and "unsupported file type" in rows[2]["error"]


def test_summary_table_is_sorted_worst_first_and_stays_aligned(stub, tree, capsys):
    arabic = tree / "صُورَة اختبار 📷.jpg"
    arabic.write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 64)
    code = cli.main(["scan", str(tree), "-r", str(tree / "notes.txt"), "--no-html"])
    out = capsys.readouterr().out
    assert code == 1 and "\033" not in out
    table = out[out.index("lvl  verdict"):].splitlines()
    head, rows, total = table[0], table[1:-1], table[-1]
    col = head.index("file")
    assert [r.split()[0] for r in rows] == ["3", "3", "0", "0", "-"]
    names = [r[col:] for r in rows]
    # every column before the name is ASCII, so the name starts at the same offset
    assert all(r[col - 1] == " " and not n.startswith(" ") for r, n in zip(rows, names))
    assert any(n.endswith("📷.jpg") for n in names) and names[-1].endswith("notes.txt")
    assert "not analysed (unsupported)" in rows[-1]
    assert total.startswith("5 files in") and "2 flagged" in total and "1 not analysed" in total


def test_display_width_of_arabic_cjk_and_emoji():
    w = cli.display_width
    assert w("photo.jpg") == 9
    assert w("صورة") == 4 and w("صُورَة") == 4          # harakat take no column
    assert w("写真") == 4 and w("📷") == 2
    assert w("👍🏽") == 2 and w("👨‍👩‍👧") == 2 and w("❤️") == 2
    assert w("é") == 1
    assert cli.display_width(cli._clip("📷" * 30, 20)) <= 20
    assert cli._pad("صورة", 8) == "صورة    "


def test_hostile_file_names_cannot_repaint_the_terminal():
    assert "\x1b" not in cli._show("a\x1b[2J\x1b[31m.jpg")
    assert "‮" not in cli._show("invoice‮gpj.exe")       # right-to-left override
    assert "\n" not in cli._show("two\nlines.jpg")
    assert cli._show("صورة 📷.jpg") == "صورة 📷.jpg"               # ordinary names untouched
    text = cli._json_text({"path": "bad\udcff.jpg"})               # undecodable byte in a name
    assert json.loads(text)["path"] == "bad\udcff.jpg"
    text.encode("utf-8")


def test_ascii_fallback_when_the_stream_cannot_show_blocks():
    legacy = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    g = cli._glyphs(legacy)
    assert g["full"] == "#" and all(ord(c) < 128 for c in "".join(g.values()))
    assert cli._glyphs(io.TextIOWrapper(io.BytesIO(), encoding="utf-8"))["full"] == "█"


class _Tty(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.skipif(sys.platform == "win32", reason="colour needs a real console on Windows")
def test_colour_only_on_a_terminal_and_never_with_no_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    assert cli._want_color(False, _Tty()) is True
    assert cli._want_color(True, _Tty()) is False                   # --no-color
    assert cli._want_color(False, io.StringIO()) is False           # a pipe
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli._want_color(False, _Tty()) is False


@pytest.mark.skipif(sys.platform == "win32", reason="colour needs a real console on Windows")
def test_no_color_flag_and_env_reach_the_output(stub, tree, monkeypatch):
    monkeypatch.setenv("TERM", "xterm")
    def run(extra):
        fake = _Tty()
        with monkeypatch.context() as m:
            m.setattr(sys, "stdout", fake)
            cli.main(["scan", str(tree / "bad.png"), "--no-html"] + extra)
        return fake.getvalue()

    assert "\033[" in run([])
    assert "\033[" not in run(["--no-color"])
    monkeypatch.setenv("NO_COLOR", "1")
    assert "\033[" not in run([])


def test_progress_only_when_stderr_is_a_terminal(stub, tree, monkeypatch, capsys):
    cli.main(["scan", str(tree / "clean.jpg"), "--no-html"])
    assert capsys.readouterr().err == ""                            # not a TTY: nothing extra
    fake = _Tty()
    monkeypatch.setattr(sys, "stderr", fake)
    cli.main(["scan", str(tree / "clean.jpg"), str(tree / "bad.png"), "--no-html", "--json"])
    assert "[0/2] analysing clean.jpg" in fake.getvalue()
    assert "[1/2] analysing bad.png" in fake.getvalue()
    json.loads(capsys.readouterr().out)                             # stdout is still pure JSON
    quiet = _Tty()
    monkeypatch.setattr(sys, "stderr", quiet)
    cli.main(["scan", str(tree / "clean.jpg"), "--no-html", "--quiet"])
    assert quiet.getvalue() == ""


def test_quiet_prints_errors_only(stub, tree, capsys):
    code = cli.main(["scan", str(tree), str(tree / "nope.jpg"), "--no-html", "--quiet"])
    cap = capsys.readouterr()
    assert code == 1 and cap.out == ""
    assert cap.err.splitlines() == [f"basira: no such file: {tree / 'nope.jpg'}"]


# ----------------------------------------------------------------- errors and exit status

def test_errors_are_one_line_without_traceback_unless_debug(stub, tree, capsys):
    boom = tree / "boom.jpg"
    boom.write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 64)
    (tree / "empty.jpg").write_bytes(b"")
    args = ["scan", str(boom), str(tree / "notes.txt"), str(tree / "empty.jpg"), "--no-html"]
    assert cli.main(args) == 1
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 3 and "Traceback" not in "".join(err)
    assert err[0] == f"basira: failed on {boom}: RuntimeError: decoder exploded with a second line"
    assert err[1].startswith("basira: unsupported file type:") and "notes.txt" in err[1]
    assert err[2] == f"basira: empty file: {tree / 'empty.jpg'}"
    assert cli.main(args + ["--debug"]) == 1
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_too_large_image_is_reported_as_such(tmp_path, capsys):          # real loader
    from PIL import Image

    p = tmp_path / "huge.png"
    Image.new("1", (8000, 8000)).save(p)                                  # 64 MP, tiny file
    assert cli.main(["scan", str(p), "--no-html"]) == 1
    err = capsys.readouterr().err.strip()
    assert err.startswith("basira: file too large:") and "60 MP" in err
    assert "\n" not in err


@pytest.mark.skipif(sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs POSIX permissions and a non-root user")
def test_unreadable_file(stub, tree, capsys):
    p = tree / "clean.jpg"
    p.chmod(0)
    try:
        assert cli.main(["scan", str(p), "--no-html"]) == 1
    finally:
        p.chmod(0o644)
    assert capsys.readouterr().err.strip() == f"basira: cannot read {p}: Permission denied"


def test_exit_status(stub, tree, capsys):
    clean, bad = str(tree / "clean.jpg"), str(tree / "bad.png")
    assert cli.main(["scan", clean, bad, "--no-html", "-q"]) == 0
    assert cli.main(["scan", clean, bad, "--no-html", "-q", "--fail-on", "3"]) == 2
    assert cli.main(["scan", clean, bad, "--no-html", "-q", "--fail-on", "4"]) == 0
    # --fail-on wins over an unreadable input: 2, not 1
    assert cli.main(["scan", "nope.jpg", bad, "--no-html", "-q", "--fail-on", "2"]) == 2
    capsys.readouterr()
    for bad_args in (["scan"], ["scan", clean, "--json", "--jsonl"], ["scan", clean, "--jobs", "-1"],
                     ["scan", clean, bad, "--html", "x.html"], ["scan", clean, "--timeout", "0"],
                     ["nonsense"]):
        with pytest.raises(SystemExit) as e:                # usage errors are 1: 2 means --fail-on
            cli.main(bad_args)
        assert e.value.code == 1, bad_args
    assert "usage:" in capsys.readouterr().err


def test_report_that_cannot_be_written_is_an_error_but_the_result_is_kept(stub, tree, capsys):
    target = tree / "notes.txt" / "x.html"                  # parent is a file
    code = cli.main(["scan", str(tree / "clean.jpg"), "--html", str(target), "--json"])
    cap = capsys.readouterr()
    assert code == 1 and json.loads(cap.out)["verdict_level"] == 0
    assert cap.err.startswith("basira: could not write report") and "Traceback" not in cap.err


def test_lang_is_passed_only_if_save_html_accepts_it(stub, tree, monkeypatch, capsys):
    seen = {}

    def with_lang(report, path, lang="en"):
        seen["lang"] = lang
        return fake_save_html(report, path)

    monkeypatch.setattr(cli, "save_html", with_lang)
    assert cli.main(["scan", str(tree / "clean.jpg"), "--lang", "ar", "-q"]) == 0
    assert seen == {"lang": "ar"} and capsys.readouterr().err == ""
    monkeypatch.setattr(cli, "save_html", fake_save_html)           # no ``lang`` parameter
    assert cli.main(["scan", str(tree / "bad.png"), "--lang", "ar"]) == 0
    assert "--lang ar ignored" in capsys.readouterr().err
    assert os.path.exists(str(tree / "bad.png") + ".basira.html")


# ----------------------------------------------------------------- worker pool

def faulty_worker(item, opts):
    """Runs in a child process: crash, hang or succeed depending on the file name."""
    name = item["path"]
    if "crash" in name:
        os._exit(7)                      # what a segfault in a decoder looks like to the parent
    if "hang" in name:
        time.sleep(600)
    out = cli._outcome(item)
    out["report"] = {"path": name, "pid": os.getpid()}
    return out


def test_a_crash_or_a_hang_costs_one_file_not_the_batch():
    names = ["ok1", "crash", "ok2", "hang", "ok3", "ok4"]
    items = [{"index": i, "path": n, "display": n, "html": None, "error": None}
             for i, n in enumerate(names)]
    t0 = time.time()
    events = list(cli.run_pool(items, jobs=2, timeout=2.0, func=faulty_worker, warm=None))
    done = {o["path"]: o for kind, o in events if kind == "done"}
    assert sorted(done) == sorted(names)
    assert all(done[n]["report"] and not done[n]["error"] for n in names if n.startswith("ok"))
    assert done["crash"]["error"]["kind"] == "crashed" and "exit status 7" in \
        done["crash"]["error"]["message"]
    assert done["hang"]["error"]["kind"] == "timeout" and "timed out after 2 s" in \
        done["hang"]["error"]["message"]
    assert time.time() - t0 < 120
    assert len({o["report"]["pid"] for o in done.values() if o["report"]} - {os.getpid()}) >= 1


def test_parallel_scan_end_to_end(tmp_path, capsys):                     # real
    clean, _ = S.make_authentic(str(tmp_path), 0)
    forged, _ = S.make_splice(str(tmp_path), 1)
    code = cli.main(["scan", clean, forged, str(tmp_path / "nope.jpg"), "--jobs", "2",
                     "--json", "--report-dir", str(tmp_path / "rep"), "--fail-on", "2"])
    out = json.loads(capsys.readouterr().out)
    assert code == 2 and [r["path"] for r in out] == [clean, forged, str(tmp_path / "nope.jpg")]
    assert out[0]["verdict_level"] == 0 and out[1]["verdict_level"] >= 2 and "error" in out[2]
    assert len(os.listdir(tmp_path / "rep")) == 2
    json.dumps(out, allow_nan=False)


def test_timeout_end_to_end(tmp_path, capsys):                           # real
    clean, _ = S.make_authentic(str(tmp_path), 0)
    assert cli.main(["scan", clean, "--no-html", "--timeout", "0.001"]) == 1
    assert "timed out after 0.001 s" in capsys.readouterr().err


def test_module_entry_point_prints_pure_json(tmp_path):                  # real, subprocess
    clean, _ = S.make_authentic(str(tmp_path), 2)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=root, PYTHONIOENCODING="utf-8")
    v = subprocess.run([sys.executable, "-m", "basira", "--version"], capture_output=True,
                       text=True, env=env, cwd=str(tmp_path))
    assert v.returncode == 0 and v.stdout.strip() == f"basira {basira.__version__}"
    r = subprocess.run([sys.executable, "-m", "basira", "scan", clean, "--json", "--no-html",
                        "--timeout", "600"], capture_output=True, env=env,
                       cwd=str(tmp_path), encoding="utf-8")       # --timeout: worker process
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["schema_version"] == 1


# ----------------------------------------------------------------- doctor

def test_doctor_passes_here_and_reports_what_it_checked(capsys):         # real
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    for word in ("Python", "NumPy", "SciPy", "OpenCV", "Pillow", "soundfile", "ffmpeg",
                 "self-test", "Everything Basira needs is present"):
        assert word in out, word
    assert "MISSING" not in out and "FAILED" not in out


def test_doctor_fails_when_something_needed_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_self_test", lambda: (True, "stubbed"))
    monkeypatch.setattr(cli, "_ffmpeg_facts", lambda: {"exe": None, "source": None,
                                                       "version": None, "decoders": None})
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "MISSING" in out and "1 problem found" in out
    facts = {"exe": "/x/ffmpeg", "source": "system PATH", "version": "7.0",
             "decoders": {"h264": True, "hevc": True, "aac": True, "mp3": True, "opus": False}}
    monkeypatch.setattr(cli, "_ffmpeg_facts", lambda: facts)
    assert cli.main(["doctor"]) == 0                       # a missing decoder is a warning
    assert "missing: opus" in capsys.readouterr().out
    assert cli.main(["doctor", "--strict"]) == 1
    monkeypatch.setattr(cli, "_self_test", lambda: (False, "detector(s) crashed: image.x"))
    assert cli.main(["doctor"]) == 1 and "FAILED" in capsys.readouterr().out
