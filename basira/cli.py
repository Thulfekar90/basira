"""Command line interface.

    basira scan photo.jpg                 # terminal summary + photo.jpg.basira.html
    basira scan clip.mp4 --json           # machine-readable JSON on stdout
    basira scan *.jpg --report-dir out/   # batch
    basira serve                          # local drag-and-drop web UI (127.0.0.1 only)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from . import __version__, analyze
from .core.report_html import save_html
from .core.types import Report

_COL = {0: "\033[32m", 1: "\033[33m", 2: "\033[38;5;208m", 3: "\033[31m", 4: "\033[31;1m"}
_RST, _DIM, _BOLD = "\033[0m", "\033[2m", "\033[1m"


def _c(s: str, code: str, on: bool) -> str:
    return f"{code}{s}{_RST}" if on else s


def _bar(v: float, width: int = 20) -> str:
    n = int(round(v * width))
    return "█" * n + "░" * (width - n)


def print_summary(r: Report, color: bool = True, indent: str = "") -> None:
    p = lambda s="": print(indent + s)  # noqa: E731
    p(_c(os.path.basename(r.path), _BOLD, color) + _c(f"  [{r.media_type}]", _DIM, color))
    p("  " + _c(r.verdict, _COL.get(r.verdict_level, ""), color))
    for key, label in (("manipulation", "manipulation "), ("ai_generation", "AI generation"),
                       ("provenance", "provenance   ")):
        v = r.scores.get(key, 0.0)
        p(f"  {label} {_bar(v)} {v * 100:5.1f}%")
    for line in r.explanation[:5]:
        p(_c("  · " + line, _DIM, color))
    for s in r.fused_segments[:8]:
        p(f"  ⏱ {s.start:7.2f}s – {s.end:7.2f}s  {s.label}")
    for c in r.children:
        print_summary(c, color, indent + "    ")


def _scan(args) -> int:
    worst = 0
    results = []
    color = sys.stdout.isatty() and not args.json
    for path in args.files:
        if not os.path.isfile(path):
            print(f"basira: no such file: {path}", file=sys.stderr)
            worst = max(worst, 1)
            continue
        try:
            r = analyze(path)
        except Exception as e:
            print(f"basira: failed on {path}: {e}", file=sys.stderr)
            worst = max(worst, 1)
            continue
        results.append(r)
        html_path = None
        if not args.no_html:
            if args.html and len(args.files) == 1:
                html_path = args.html
            else:
                base = os.path.basename(path) + ".basira.html"
                d = args.report_dir or os.path.dirname(os.path.abspath(path))
                os.makedirs(d, exist_ok=True)
                html_path = os.path.join(d, base)
            save_html(r, html_path)
        if not args.json:
            print_summary(r, color)
            if html_path:
                print(_c(f"  report: {html_path}", _DIM, color))
            print()
        if args.fail_on is not None and r.verdict_level >= args.fail_on:
            worst = max(worst, 2)
    if args.json:
        out = [r.to_dict() for r in results]
        json.dump(out[0] if len(out) == 1 else out, sys.stdout, indent=2, ensure_ascii=False)
        print()
    return worst


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="basira", description="Explainable forensics for AI-generated and manipulated "
        "images, audio and video. Reports evidence, not proof.")
    ap.add_argument("--version", action="version", version=f"basira {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    sc = sub.add_parser("scan", help="analyse one or more media files")
    sc.add_argument("files", nargs="+")
    sc.add_argument("--json", action="store_true", help="print JSON to stdout")
    sc.add_argument("--html", help="path of the HTML report (single file)")
    sc.add_argument("--report-dir", help="directory for HTML reports")
    sc.add_argument("--no-html", action="store_true", help="do not write an HTML report")
    sc.add_argument("--fail-on", type=int, choices=[1, 2, 3, 4], default=None,
                    help="exit with status 2 if any verdict level is >= this (for CI / pipelines)")
    sv = sub.add_parser("serve", help="local web UI (binds to 127.0.0.1)")
    sv.add_argument("--port", type=int, default=8642)
    sv.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "scan":
        return _scan(args)
    if args.cmd == "serve":
        from .server import serve

        return serve(args.port, not args.no_browser)
    ap.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
