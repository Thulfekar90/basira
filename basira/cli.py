"""Command line interface.

    basira scan photo.jpg                 # terminal summary + photo.jpg.basira.html
    basira scan clip.mp4 --json           # machine-readable JSON on stdout
    basira scan *.jpg --report-dir out/   # batch
    basira scan folder/ --recursive --jobs 4 --csv summary.csv
    basira doctor                         # check the installation
    basira serve                          # local drag-and-drop web UI (127.0.0.1 only)

Exit status of ``scan``: 0 = every input was analysed, 1 = at least one input could not be
analysed (or the command line was wrong), 2 = ``--fail-on`` was triggered. Details and the
JSON layout: docs/USAGE.md and docs/JSON.md.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import inspect
import json
import os
import re
import shutil
import signal
import sys
import time
import traceback
import unicodedata
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from . import __version__, analyze
from .core.report_html import save_html
from .core.types import SCHEMA_VERSION, Report

_COL = {0: "\033[32m", 1: "\033[33m", 2: "\033[38;5;208m", 3: "\033[31m", 4: "\033[31;1m"}
_RST, _DIM, _BOLD = "\033[0m", "\033[2m", "\033[1m"

# Seconds a worker process may take to start (import NumPy, SciPy, OpenCV) before it is
# considered broken. The per-file ``--timeout`` clock only starts once it reports ready.
STARTUP_LIMIT = 300.0

CSV_COLUMNS = ["path", "media_type", "verdict_level", "verdict", "manipulation",
               "ai_generation", "provenance", "overall", "detectors_failed", "elapsed_sec",
               "report", "error"]


# --------------------------------------------------------------------------- text helpers

def _c(s: str, code: str, on: bool) -> str:
    return f"{code}{s}{_RST}" if on else s


def _can_encode(stream, text: str) -> bool:
    try:
        text.encode(getattr(stream, "encoding", None) or "utf-8")
        return True
    except (UnicodeError, LookupError):
        return False


def _glyphs(stream=None) -> Dict[str, str]:
    """Block characters when the stream can show them, plain ASCII otherwise (cp1252 pipes)."""
    stream = stream or sys.stdout
    if _can_encode(stream, "█░·⏱–"):
        return {"full": "█", "empty": "░", "dot": "·", "clock": "⏱", "dash": "–"}
    return {"full": "#", "empty": "-", "dot": "*", "clock": "@", "dash": "-"}


def _bar(v: float, width: int = 20, glyphs: Optional[Dict[str, str]] = None) -> str:
    g = glyphs or _glyphs()
    n = int(round(max(0.0, min(1.0, float(v or 0.0))) * width))
    return g["full"] * n + g["empty"] * (width - n)


_BIDI_OVERRIDES = set("‪‫‬‭‮⁦⁧⁨⁩")


def _show(s: Any) -> str:
    """A file name made safe to print: control characters, bidi overrides (the trick that
    makes ``exe.jpg`` out of ``gpj.exe``) and undecodable bytes are shown as escapes, so a
    hostile name cannot repaint the terminal or hide part of a line."""
    out = []
    for ch in str(s):
        if ch in _BIDI_OVERRIDES or unicodedata.category(ch) in ("Cc", "Cs", "Zl", "Zp"):
            out.append(ch.encode("unicode_escape").decode("ascii"))
        else:
            out.append(ch)
    return "".join(out)


def display_width(s: str) -> int:
    """Number of terminal columns ``s`` occupies: combining marks (Arabic harakat) count 0,
    wide CJK characters and emoji count 2, joined emoji sequences count once."""
    width, last, after_zwj = 0, 0, False
    for ch in s:
        o = ord(ch)
        if o == 0x200D:                       # zero-width joiner
            after_zwj = True
            continue
        if o == 0xFE0F:                       # emoji presentation: narrow symbol turns wide
            if last == 1:
                width, last = width + 1, 2
            continue
        cat = unicodedata.category(ch)
        if cat in ("Mn", "Me", "Cf", "Cc") or unicodedata.combining(ch):
            cw = 0
        elif unicodedata.east_asian_width(ch) in ("W", "F"):
            cw = 2
        else:
            cw = 1
        joined = after_zwj and cw == 2 and last == 2          # second half of a ZWJ emoji
        modifier = 0x1F3FB <= o <= 0x1F3FF and last == 2      # skin tone on an emoji
        after_zwj = False
        if joined or modifier:
            continue
        width += cw
        if cw:
            last = cw
    return width


def _clip(s: str, width: int) -> str:
    """Cut ``s`` to at most ``width`` terminal columns, marking the cut with ``...``."""
    if display_width(s) <= width:
        return s
    out, used = [], 0
    for ch in s:
        w = display_width(ch)
        if used + w > max(0, width - 3):
            break
        out.append(ch)
        used += w
    return "".join(out) + "..."


def _pad(s: str, width: int, right: bool = False) -> str:
    fill = " " * max(0, width - display_width(s))
    return fill + s if right else s + fill


def _one_line(s: Any, limit: int = 300) -> str:
    text = " ".join(str(s).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _want_color(no_color_flag: bool, stream) -> bool:
    """Colour only for an interactive terminal, and never with --no-color or NO_COLOR."""
    if no_color_flag or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    try:
        if not stream.isatty():
            return False
    except Exception:
        return False
    if sys.platform == "win32":
        return _enable_windows_vt()
    return True


def _enable_windows_vt() -> bool:  # pragma: no cover - Windows consoles only
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        handle = k32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not k32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(k32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _prepare_streams(machine: bool) -> None:
    """Never die on a character the console cannot show; JSON on stdout is always UTF-8."""
    for name in ("stdout", "stderr"):
        reconfigure = getattr(getattr(sys, name), "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if machine and name == "stdout":
                reconfigure(encoding="utf-8", errors="strict")
            else:
                reconfigure(errors="backslashreplace")
        except Exception:
            pass


def _json_text(obj: Any, indent: Optional[int] = None) -> str:
    """Strict JSON (no NaN). Non-ASCII is written as is; a file name with undecodable bytes
    falls back to ``\\uXXXX`` escapes instead of failing."""
    seps = None if indent else (",", ":")
    text = json.dumps(obj, indent=indent, ensure_ascii=False, allow_nan=False, separators=seps)
    try:
        text.encode("utf-8")
    except UnicodeError:
        text = json.dumps(obj, indent=indent, ensure_ascii=True, allow_nan=False,
                          separators=seps)
    return text


def _columns(stream) -> int:
    """Width of the terminal behind ``stream`` (80 when it cannot be asked or says 0)."""
    try:
        cols = os.get_terminal_size(stream.fileno()).columns
    except Exception:
        cols = shutil.get_terminal_size((80, 24)).columns
    return cols if cols > 0 else 80


class _Console:
    """Messages and progress on stderr. The progress line exists only on a terminal and is
    wiped before anything else is printed, so pipes and logs never see it."""

    def __init__(self, quiet: bool, debug: bool) -> None:
        self.quiet, self.debug = quiet, debug
        try:
            tty = sys.stderr.isatty()
        except Exception:
            tty = False
        self.live = tty and not quiet
        self._shown = 0

    def clear(self) -> None:
        if self._shown:
            sys.stderr.write("\r" + " " * self._shown + "\r")
            sys.stderr.flush()
            self._shown = 0

    def progress(self, text: str) -> None:
        if not self.live:
            return
        text = _clip(text, max(20, _columns(sys.stderr) - 1))
        self.clear()
        sys.stderr.write(text)
        sys.stderr.flush()
        self._shown = display_width(text)

    def error(self, msg: str, tb: Optional[str] = None) -> None:
        self.clear()
        print(f"basira: {msg}", file=sys.stderr)
        if tb and self.debug:
            print(tb.rstrip(), file=sys.stderr)

    def note(self, msg: str) -> None:
        if not self.quiet:
            self.clear()
            print(f"basira: {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- human output

def print_summary(r: Any, color: bool = True, indent: str = "") -> None:
    """Terminal summary of one report (a :class:`Report` or its ``to_dict()``)."""
    d = r.to_dict() if isinstance(r, Report) else r
    g = _glyphs()
    p = lambda s="": print(indent + s)  # noqa: E731
    p(_c(_show(os.path.basename(d["path"])), _BOLD, color)
      + _c(f"  [{d['media_type']}]", _DIM, color))
    p("  " + _c(d["verdict"], _COL.get(d["verdict_level"], ""), color))
    scores = d.get("scores") or {}
    for key, label in (("manipulation", "manipulation "), ("ai_generation", "AI generation"),
                       ("provenance", "provenance   ")):
        v = scores.get(key) or 0.0
        p(f"  {label} {_bar(v, glyphs=g)} {v * 100:5.1f}%")
    for line in (d.get("explanation") or [])[:5]:
        p(_c(f"  {g['dot']} " + line, _DIM, color))
    for s in (d.get("segments") or [])[:8]:
        p(f"  {g['clock']} {s['start'] or 0:7.2f}s {g['dash']} {s['end'] or 0:7.2f}s  {s['label']}")
    for c in d.get("children") or []:
        print_summary(c, color, indent + "    ")


def _pct(v: Any) -> str:
    return f"{float(v) * 100:5.1f}%" if isinstance(v, (int, float)) else "     -"


def _sort_key(o: Dict[str, Any]) -> Tuple:
    """Worst first; inputs that could not be analysed go last."""
    rep = o.get("report")
    if not rep:
        return (1, 0, 0.0, o["index"])
    s = rep.get("scores") or {}
    top = max(s.get("manipulation") or 0.0, s.get("ai_generation") or 0.0)
    return (0, -int(rep.get("verdict_level") or 0), -top, o["index"])


def print_table(outcomes: List[Dict[str, Any]], color: bool = False) -> None:
    """One line per input, worst first. The file name is the LAST column on purpose: every
    other column is ASCII, so Arabic, CJK or emoji names cannot push a column out of line
    (and a right-to-left name cannot flip the row in a terminal that reorders text)."""
    rows = []
    for o in sorted(outcomes, key=_sort_key):
        rep = o.get("report")
        if rep:
            s = rep.get("scores") or {}
            rows.append((str(rep["verdict_level"]), rep["verdict"], _pct(s.get("manipulation")),
                         _pct(s.get("ai_generation")), f"{rep.get('elapsed_sec') or 0:6.1f}s",
                         rep["media_type"], _show(o["display"]), rep["verdict_level"]))
        else:
            kind = (o.get("error") or {}).get("kind", "failed").replace("_", " ")
            rows.append(("-", f"not analysed ({kind})", "     -", "     -",
                         f"{o.get('elapsed') or 0:6.1f}s", "-", _show(o["display"]), None))
    head = ("lvl", "verdict", "manip.", "AI", "time", "type", "file")
    wv = max([len(head[1])] + [display_width(r[1]) for r in rows])
    wt = max([len(head[5])] + [len(r[5]) for r in rows])

    def line(lvl, verdict, manip, ai, tm, typ, name) -> str:
        return (f"{lvl:>3}  {_pad(verdict, wv)}  {manip:>6}  {ai:>6}  {tm:>7}  "
                f"{_pad(typ, wt)}  {name}")

    print(_c(line(*head), _DIM, color))
    for r in rows:
        text = line(*r[:7])
        print(_c(text, _COL.get(r[7], ""), color) if r[7] else text)


def _totals(outcomes: List[Dict[str, Any]], skipped: int, seconds: float) -> str:
    done = [o for o in outcomes if o.get("report")]
    flagged = sum(1 for o in done if o["report"]["verdict_level"] >= 2)
    failed = len(outcomes) - len(done)
    n = len(outcomes)
    parts = [f"{flagged} flagged (level 2 or higher)", f"{len(done) - flagged} not flagged"]
    if failed:
        parts.append(f"{failed} not analysed")
    if skipped:
        parts.append(f"{skipped} unsupported skipped")
    return f"{n} file{'s' if n != 1 else ''} in {seconds:.1f} s: " + ", ".join(parts)


# --------------------------------------------------------------------------- inputs

_IGNORED_NAMES = {"thumbs.db", "desktop.ini"}


def _known_extensions() -> set:
    from .core import io as _io

    return _io.IMAGE_EXT | _io.AUDIO_EXT | _io.VIDEO_EXT


def _looks_like_media(path: str, known: set) -> bool:
    if os.path.splitext(path)[1].lower() in known:
        return True
    from .core.io import sniff

    try:
        sniff(path)
        return True
    except Exception:
        return False


def _ignorable(name: str) -> bool:
    low = name.lower()
    return name.startswith(".") or low.endswith(".basira.html") or low in _IGNORED_NAMES


def expand_inputs(args_files: List[str], recursive: bool
                  ) -> Tuple[List[Dict[str, Any]], List[str], List[str], bool]:
    """Turn command-line arguments into the list of files to analyse.

    * a file is taken as is (an unsupported or missing one is reported as an error later);
    * a directory contributes its media files (sub-directories only with ``recursive``);
    * a pattern the shell did not expand (Windows, or a quoted ``'*.jpg'``) is expanded here.

    Files found through a directory or a pattern that are not media are *skipped*, not
    errors. Returns ``(items, skipped_paths, notes, batch)`` where ``batch`` is False only
    for exactly one plain file argument."""
    known = _known_extensions()
    items: List[Dict[str, Any]] = []
    skipped: List[str] = []
    notes: List[str] = []
    seen = set()
    batch = len(args_files) != 1

    def add(path: str, rel: str, error: Optional[Tuple[str, str]] = None) -> None:
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            return
        seen.add(key)
        items.append({"index": len(items), "path": path, "display": path, "rel": rel,
                      "html": None, "error": error})

    def add_found(path: str, rel: str) -> None:
        if _ignorable(os.path.basename(path)):
            return
        if _looks_like_media(path, known):
            add(path, rel)
        else:
            skipped.append(path)

    def add_dir(root: str) -> None:
        before = len(items)
        subdirs = 0
        if recursive:
            for cur, dirs, files in os.walk(root):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                for name in sorted(files):
                    full = os.path.join(cur, name)
                    add_found(full, os.path.relpath(full, root))
        else:
            for name in sorted(os.listdir(root)):
                full = os.path.join(root, name)
                if os.path.isdir(full):
                    subdirs += not name.startswith(".")
                elif os.path.isfile(full):
                    add_found(full, name)
        if len(items) == before:
            more = f" ({subdirs} sub-folder(s) not searched: add --recursive)" if subdirs else ""
            notes.append(f"no media files found in {_show(root)}{more}")

    for arg in args_files:
        if os.path.isdir(arg):
            batch = True
            try:
                add_dir(arg)
            except OSError as e:
                add(arg, os.path.basename(arg),
                    ("unreadable", f"cannot read folder {_show(arg)}: {e.strerror or e}"))
        elif os.path.exists(arg) or not glob.has_magic(arg):
            add(arg, os.path.basename(arg))
        else:
            batch = True
            matches = sorted(glob.glob(arg, recursive=True))
            if not matches:
                add(arg, arg, ("not_found", f"no files match: {_show(arg)}"))
            for m in matches:
                if os.path.isdir(m):
                    try:
                        add_dir(m)
                    except OSError as e:
                        add(m, os.path.basename(m),
                            ("unreadable", f"cannot read folder {_show(m)}: {e.strerror or e}"))
                else:
                    add_found(m, os.path.basename(m))
    return items, skipped, notes, batch


def _assign_reports(items: List[Dict[str, Any]], args) -> None:
    """Decide where each HTML report goes. With ``--report-dir`` a scanned folder keeps its
    layout, and two inputs can never be given the same report file."""
    if args.no_html:
        return
    used = set()
    for it in items:
        if args.html and len(items) == 1:
            it["html"] = args.html
            continue
        if args.report_dir:
            base = os.path.join(args.report_dir, it["rel"])
        else:
            base = it["path"]
        path, n = base + ".basira.html", 1
        while os.path.normcase(os.path.abspath(path)) in used:
            n += 1
            path = f"{base}-{n}.basira.html"
        used.add(os.path.normcase(os.path.abspath(path)))
        it["html"] = path


# --------------------------------------------------------------------------- one file

def _html_kwargs(lang: Optional[str]) -> Tuple[Dict[str, Any], bool]:
    """``save_html`` grew a ``lang`` parameter after 0.1.0 was drafted; pass it only when
    the installed function has one. Returns (kwargs, supported)."""
    if not lang:
        return {}, True
    try:
        params = inspect.signature(save_html).parameters
    except (TypeError, ValueError):
        return {}, False
    if "lang" in params or any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return {"lang": lang}, True
    return {}, False


def _outcome(item: Dict[str, Any]) -> Dict[str, Any]:
    return {"index": item["index"], "path": item["path"], "display": item["display"],
            "report": None, "error": None, "traceback": None, "html": None,
            "html_error": None, "elapsed": 0.0}


def _fail(item: Dict[str, Any], kind: str, message: str, tb: Optional[str] = None,
          elapsed: float = 0.0) -> Dict[str, Any]:
    out = _outcome(item)
    out.update(error={"kind": kind, "message": message}, traceback=tb, elapsed=elapsed)
    return out


_TOO_LARGE = re.compile(r"limit (is )?\d+ ?M[BP]\b|not enough memory|pixels per byte"
                        r"|outside the supported range|channels; the limit is")


def _classify(path: str, exc: BaseException) -> Tuple[str, str]:
    """One line for a failed analysis. ``basira.analyze`` raises ValueError for "not media",
    "over a size limit" and "cannot be decoded" alike, so the message text decides."""
    shown, msg = _show(path), _one_line(exc) or type(exc).__name__
    lead = os.path.basename(path) + ": "
    if msg.startswith(lead):                     # some messages already name the file
        msg = msg[len(lead):]
    if isinstance(exc, MemoryError) or (isinstance(exc, ValueError) and _TOO_LARGE.search(msg)):
        return "too_large", f"file too large: {shown}: {msg or 'not enough memory'}"
    if isinstance(exc, ValueError) and msg.startswith("Cannot read"):
        return "unreadable", f"cannot read {shown}: {msg.split(': ', 1)[-1]}"
    if isinstance(exc, ValueError) and (msg.startswith("Unrecognised media type")
                                        or msg.startswith("Unsupported file")):
        why = "not an image, audio or video format Basira reads"
        if " is " in msg:                        # "Unsupported file: x is a playlist / ..."
            why = msg.split(" is ", 1)[1]
        return "unsupported", f"unsupported file type: {shown} ({why})"
    if isinstance(exc, ValueError):
        return "failed", f"failed on {shown}: {msg}"
    return "failed", f"failed on {shown}: {type(exc).__name__}: {msg}"


def analyse_one(item: Dict[str, Any], opts: Dict[str, Any]) -> Dict[str, Any]:
    """Analyse one input and write its HTML report. Never raises: every problem becomes a
    one-line error in the returned dict. Runs in this process or in a worker process."""
    if item.get("error"):
        return _fail(item, *item["error"])
    path, shown = item["path"], _show(item["path"])
    t0 = time.time()
    if os.path.isdir(path):
        return _fail(item, "unsupported", f"is a folder, not a file: {shown}")
    if not os.path.isfile(path):
        return _fail(item, "not_found", f"no such file: {shown}")
    try:
        with open(path, "rb") as f:
            empty = not f.read(1)
    except OSError as e:
        return _fail(item, "unreadable", f"cannot read {shown}: {e.strerror or e}")
    if empty:
        return _fail(item, "unsupported", f"empty file: {shown}")
    try:
        report = analyze(path)
        data = report.to_dict()
    except Exception as e:  # noqa: BLE001 - the batch must go on
        kind, msg = _classify(path, e)
        return _fail(item, kind, msg, traceback.format_exc(), time.time() - t0)
    out = _outcome(item)
    out["report"] = data
    if item.get("html"):
        try:
            folder = os.path.dirname(os.path.abspath(item["html"]))
            os.makedirs(folder, exist_ok=True)
            save_html(report, item["html"], **opts.get("html_kwargs", {}))
            out["html"] = item["html"]
        except Exception as e:  # noqa: BLE001
            out["html_error"] = (f"could not write report {_show(item['html'])}: "
                                 f"{_one_line(getattr(e, 'strerror', None) or e)}")
            out["traceback"] = traceback.format_exc()
    out["elapsed"] = time.time() - t0
    return out


# --------------------------------------------------------------------------- worker pool

def _warm_up(opts: Dict[str, Any]) -> None:
    """Import the heavy modules before the first file, so that the first file's timeout
    does not pay for them; share the CPU between workers."""
    from . import audio, image, video  # noqa: F401

    try:
        import cv2

        cv2.setNumThreads(max(1, int(opts.get("threads") or 1)))
    except Exception:
        pass


def _worker_loop(conn, func: Callable, opts: Dict[str, Any],
                 warm: Optional[Callable]) -> None:  # pragma: no cover - runs in a child
    try:
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)   # the parent handles Ctrl+C
    except Exception:
        pass
    if hasattr(os, "setsid"):
        try:
            # Own process group, so a timed-out worker is killed together with any FFmpeg
            # it started (``_Worker.stop`` kills the whole group).
            os.setsid()
        except OSError:
            pass
    try:
        os.dup2(2, 1)       # nothing a decoder prints may reach the parent's stdout (JSON)
    except Exception:
        pass
    try:
        if warm is not None:
            warm(opts)
        while True:
            try:
                item = conn.recv()
            except (EOFError, OSError):
                break
            if item is None:
                break
            conn.send(("start", item["index"]))
            try:
                result = func(item, opts)
            except Exception as e:  # noqa: BLE001
                result = _fail(item, "failed", f"failed on {_show(item['path'])}: "
                               f"{type(e).__name__}: {_one_line(e)}", traceback.format_exc())
            conn.send(("done", result))
    finally:
        try:
            conn.close()
        except Exception:
            pass


class _Slot:
    def __init__(self, ctx, func, opts, warm) -> None:
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_loop, args=(child, func, opts, warm),
                                daemon=True)
        self.proc.start()
        child.close()
        self.item: Optional[Dict[str, Any]] = None
        self.pgid: Optional[int] = None
        self.given = 0.0
        self.started: Optional[float] = None

    def give(self, item: Dict[str, Any]) -> None:
        self.conn.send(item)
        self.item, self.given, self.started = item, time.monotonic(), None

    def _kill_group(self) -> None:
        _kill_process_group(self.pgid or _own_group(self.proc.pid))

    def stop(self, kill: bool = False) -> Optional[int]:
        if not kill:
            try:
                self.conn.send(None)
            except Exception:
                pass
            self.proc.join(2.0)
        if self.proc.is_alive() or kill:
            self._kill_group()
            if self.proc.is_alive():
                self.proc.kill()
            self.proc.join(10.0)
        try:
            self.conn.close()
        except Exception:
            pass
        return self.proc.exitcode


def _is_an_input(out: str, items: List[Dict[str, Any]]) -> bool:
    """True if writing ``out`` would overwrite a file being scanned, including through a
    symlink or a hard link (same device and inode)."""
    try:
        st = os.stat(out)
    except OSError:
        return False                                    # does not exist yet: cannot be an input
    for it in items:
        try:
            si = os.stat(it["path"])
        except OSError:
            continue
        if (si.st_dev, si.st_ino) == (st.st_dev, st.st_ino):
            return True
    return False


def _own_group(pid: Optional[int]) -> Optional[int]:
    """The worker's process group id, if the worker leads its own group (it calls setsid)."""
    if pid is None or not hasattr(os, "getpgid"):
        return None
    try:
        return pid if os.getpgid(pid) == pid else None
    except OSError:
        return None


def _kill_process_group(pgid: Optional[int]) -> None:
    """Kill a worker's process group: the worker and every decoder it spawned, even after
    the worker itself has died (POSIX does not reuse a pid while its group still exists).
    POSIX only; on Windows the worker alone is killed and a decoder it started can
    outlive it until the decoder's own time limit (see SECURITY.md)."""
    if pgid is None or not hasattr(os, "killpg") or pgid == os.getpgrp():
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


def _exit_text(code: Optional[int]) -> str:
    if code is None:
        return "no exit status"
    if code < 0:
        try:
            import signal

            return f"killed by {signal.Signals(-code).name}"
        except Exception:
            return f"killed by signal {-code}"
    return f"exit status {code}"


def run_pool(items: List[Dict[str, Any]], jobs: int, timeout: Optional[float],
             func: Callable = analyse_one, opts: Optional[Dict[str, Any]] = None,
             warm: Optional[Callable] = _warm_up) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Analyse ``items`` in up to ``jobs`` worker processes, yielding ``("start", item)``
    and ``("done", outcome)`` as they happen.

    A worker that crashes (segfault in a decoder) or exceeds ``timeout`` seconds on one
    file costs exactly that file: it is reported as an error, the worker is replaced and
    the batch goes on. ``func`` must be importable (it is sent to a fresh interpreter)."""
    import multiprocessing as mp
    from multiprocessing.connection import wait

    ctx = mp.get_context("spawn")        # same behaviour on Linux, macOS and Windows
    opts = dict(opts or {})
    queue = collections.deque(items)
    slots: List[_Slot] = []
    try:
        while queue or any(s.item is not None for s in slots):
            for s in slots:
                if s.item is None and queue:
                    s.give(queue.popleft())
            while queue and len(slots) < jobs:
                s = _Slot(ctx, func, opts, warm)
                slots.append(s)
                s.give(queue.popleft())
            busy = [s for s in slots if s.item is not None]
            ready = wait([s.conn for s in busy], timeout=0.1)
            for s in busy:
                item = s.item
                if s.conn in ready:
                    try:
                        kind, payload = s.conn.recv()
                    except Exception:  # noqa: BLE001 - EOF: the worker died
                        slots.remove(s)
                        code = s.stop(kill=True)
                        yield "done", _fail(
                            item, "crashed", f"analysis process crashed ({_exit_text(code)}) "
                            f"on {_show(item['path'])}", elapsed=time.monotonic() - s.given)
                        continue
                    if kind == "start":
                        s.started = time.monotonic()
                        if s.pgid is None:      # the worker has called setsid by now
                            s.pgid = _own_group(s.proc.pid)
                        yield "start", item
                    else:
                        s.item = None
                        yield "done", payload
                    continue
                now = time.monotonic()
                if s.started is None and now - s.given > STARTUP_LIMIT:
                    slots.remove(s)
                    s.stop(kill=True)
                    yield "done", _fail(item, "crashed", "analysis process did not start "
                                        f"within {STARTUP_LIMIT:.0f} s for "
                                        f"{_show(item['path'])}")
                elif s.started is not None and timeout and now - s.started > timeout:
                    slots.remove(s)
                    s.stop(kill=True)
                    yield "done", _fail(item, "timeout", f"timed out after {timeout:g} s: "
                                        f"{_show(item['path'])}", elapsed=now - s.started)
    finally:
        for s in slots:
            s.stop(kill=s.item is not None)


def _run_inline(items: List[Dict[str, Any]], opts: Dict[str, Any]
                ) -> Iterator[Tuple[str, Dict[str, Any]]]:
    for it in items:
        yield "start", it
        yield "done", analyse_one(it, opts)


# --------------------------------------------------------------------------- scan

class _UsageError(Exception):
    pass


def _csv_cell(v: Any) -> Any:
    """Spreadsheets run a cell that starts with = + - @ as a formula; a file name is
    attacker-controlled text, so such a cell is written with a leading apostrophe."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def _csv_row(o: Dict[str, Any]) -> List[Any]:
    rep = o.get("report") or {}
    s = rep.get("scores") or {}
    err = o.get("error") or {}
    row = [o["path"], rep.get("media_type", ""), rep.get("verdict_level", ""),
           rep.get("verdict", ""), s.get("manipulation", ""), s.get("ai_generation", ""),
           s.get("provenance", ""), s.get("overall", ""),
           (rep.get("info") or {}).get("detectors_failed", ""), rep.get("elapsed_sec", ""),
           o.get("html") or "", err.get("message") or o.get("html_error") or ""]
    return [_csv_cell(v) for v in row]


def _record(o: Dict[str, Any]) -> Dict[str, Any]:
    """What ``--json`` / ``--jsonl`` print for one input: the report, or an error record."""
    if o.get("report"):
        return o["report"]
    return {"tool": "basira", "schema_version": SCHEMA_VERSION, "version": __version__,
            "path": o["path"], "error": o["error"]}


def _scan(args) -> int:
    machine = bool(args.json or args.jsonl)
    _prepare_streams(machine)
    con = _Console(args.quiet, args.debug)
    color = (not machine) and _want_color(args.no_color, sys.stdout)
    human = not machine and not args.quiet

    items, skipped, notes, batch = expand_inputs(args.files, args.recursive)
    if args.html and not args.no_html and (batch or len(items) != 1):
        raise _UsageError("--html takes exactly one input file; use --report-dir for several")
    _assign_reports(items, args)
    html_kwargs, lang_ok = _html_kwargs(args.lang)
    if args.lang and not lang_ok and not args.no_html:
        con.note(f"--lang {args.lang} ignored: this version writes the HTML report in "
                 "English only")
    for n in notes:
        con.note(n)
    if skipped:
        con.note(f"skipped {len(skipped)} unsupported file{'s' if len(skipped) != 1 else ''}"
                 + (f" (not image, audio or video), e.g. {_show(skipped[0])}"
                    if not args.debug else ": " + ", ".join(_show(s) for s in skipped)))

    if args.csv and _is_an_input(args.csv, items):
        con.error(f"refusing to write --csv {_show(args.csv)}: it is one of the files to scan")
        return 1

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, len(items) or 1))
    opts = {"html_kwargs": html_kwargs, "threads": max(1, (os.cpu_count() or 1) // jobs)}
    use_pool = jobs > 1 or args.timeout is not None
    events = run_pool(items, jobs, args.timeout, analyse_one, opts) if use_pool \
        else _run_inline(items, opts)

    csv_file = csv_out = None
    if args.csv:
        try:
            folder = os.path.dirname(os.path.abspath(args.csv))
            os.makedirs(folder, exist_ok=True)
            # utf-8 with a BOM: Excel then reads Arabic file names correctly.
            csv_file = open(args.csv, "w", newline="", encoding="utf-8-sig",
                            errors="backslashreplace")   # undecodable POSIX names
        except OSError as e:
            con.error(f"cannot write {_show(args.csv)}: {e.strerror or e}")
            return 1
        csv_out = csv.writer(csv_file)
        csv_out.writerow(CSV_COLUMNS)

    outcomes: List[Dict[str, Any]] = []
    running: Dict[int, str] = {}
    had_error = triggered = interrupted = False
    t_start = time.time()

    def show_progress() -> None:
        if not con.live:
            return
        if not running:
            con.clear()
            return
        names = ", ".join(_show(os.path.basename(p)) for p in running.values())
        con.progress(f"[{len(outcomes)}/{len(items)}] analysing {names}")

    try:
        for kind, payload in events:
            if kind == "start":
                running[payload["index"]] = payload["path"]
                show_progress()
                continue
            o = payload
            running.pop(o["index"], None)
            outcomes.append(o)
            rep = o.get("report")
            if o.get("error"):
                had_error = True
                con.error(o["error"]["message"], o.get("traceback"))
            if o.get("html_error"):
                had_error = True
                con.error(o["html_error"], o.get("traceback"))
            if rep and args.fail_on is not None and rep["verdict_level"] >= args.fail_on:
                triggered = True
            if csv_out is not None:
                csv_out.writerow(_csv_row(o))
                csv_file.flush()
            if args.jsonl:
                con.clear()
                print(_json_text(_record(o)), flush=True)
            elif human and rep:
                con.clear()
                print_summary(rep, color)
                if o.get("html"):
                    print(_c(f"  report: {_show(o['html'])}", _DIM, color))
                print()
            show_progress()
    except KeyboardInterrupt:
        interrupted = True
        if hasattr(events, "close"):
            events.close()                      # stops the worker processes
        con.error("interrupted")
    finally:
        con.clear()
        if csv_file is not None:
            csv_file.close()

    if args.json:
        ordered = [_record(o) for o in sorted(outcomes, key=lambda o: o["index"])]
        doc: Any = ordered if batch else (ordered[0] if ordered else None)
        print(_json_text(doc, indent=2))
    elif human and len(outcomes) > 1:
        print_table(outcomes, color)
        print(_totals(outcomes, len(skipped), time.time() - t_start))
    if args.csv and not args.quiet and not machine:
        print(f"summary: {_show(args.csv)}")
    try:
        sys.stdout.flush()
    except Exception:
        pass
    if interrupted:
        return 130
    return 2 if triggered else (1 if had_error else 0)


# --------------------------------------------------------------------------- doctor

def _self_test() -> Tuple[bool, str]:
    """Analyse a generated photograph-like JPEG end to end (decode, every image detector,
    JSON, HTML). Takes about two seconds; nothing is downloaded or left on disk."""
    import tempfile

    import numpy as np
    from PIL import Image

    from . import image  # noqa: F401 - import the engine before the clock starts
    from .core.report_html import render_html

    rs = np.random.RandomState(7)
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
    base = 110 + 60 * np.sin(xx / 47.0) * np.cos(yy / 31.0) + 0.08 * xx
    rgb = np.stack([base, base * 0.92 + 9, base * 0.85 + 18], axis=-1)
    rgb = np.clip(rgb + rs.normal(0, 4.0, rgb.shape), 0, 255).astype(np.uint8)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="basira_doctor_") as td:
        path = os.path.join(td, "selftest.jpg")
        Image.fromarray(rgb).save(path, quality=90)
        report = analyze(path)
    data = report.to_dict()
    json.dumps(data, allow_nan=False)
    page = render_html(report)
    took = time.time() - t0
    broken = [e["detector"] for e in data["evidences"] if e.get("error")]
    if broken:
        return False, "detector(s) crashed on a generated 640x480 JPEG: " + ", ".join(broken)
    if not data["verdict"] or "<html" not in page.lower():
        return False, "analysis ran but produced an empty verdict or report"
    return True, (f"640x480 JPEG analysed in {took:.1f} s, {len(data['evidences'])} detectors "
                  f"ran, verdict \"{data['verdict']}\"")


FFMPEG_DECODERS = (("h264", ("h264",)), ("hevc", ("hevc",)), ("aac", ("aac", "aac_at")),
                   ("mp3", ("mp3", "mp3float", "mp3_at")), ("opus", ("opus", "libopus")))


def _ffmpeg_facts() -> Dict[str, Any]:
    import subprocess

    from .core.io import ffmpeg_exe

    exe = ffmpeg_exe()
    facts: Dict[str, Any] = {"exe": exe, "source": None, "version": None, "decoders": None}
    if not exe:
        return facts
    system = shutil.which("ffmpeg")
    facts["source"] = "system PATH" if system and os.path.abspath(system) == \
        os.path.abspath(exe) else "bundled with imageio-ffmpeg"
    try:
        r = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, text=True,
                           timeout=30, errors="replace")
        first = (r.stdout or r.stderr).splitlines()[0]
        facts["version"] = first.replace("ffmpeg version ", "").split(" Copyright")[0].strip()
        r = subprocess.run([exe, "-hide_banner", "-decoders"], capture_output=True, text=True,
                           timeout=30, errors="replace")
        names = set()
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
                names.add(parts[1])
        facts["decoders"] = {label: any(a in names for a in alts)
                             for label, alts in FFMPEG_DECODERS}
    except Exception as e:  # noqa: BLE001
        facts["error"] = _one_line(e)
    return facts


def _doctor(args) -> int:
    """Print what is installed and whether it works. Exit status 1 if something Basira
    needs is missing or the self-test fails (with ``--strict`` also on a warning)."""
    import importlib
    import platform

    _prepare_streams(False)
    color = _want_color(args.no_color, sys.stdout)
    problems = warnings = 0

    def row(status: str, label: str, text: str) -> None:
        nonlocal problems, warnings
        problems += status == "MISSING" or status == "FAILED"
        warnings += status == "warn"
        code = {"ok": _COL[0], "warn": _COL[1]}.get(status, _COL[3])
        print(f"  {_c(f'{status:<7}', code, color)} {label:<15} {text}")

    print(f"basira {__version__}  ({os.path.dirname(os.path.abspath(__file__))})")
    py_ok = sys.version_info >= (3, 9)
    row("ok" if py_ok else "MISSING", "Python",
        f"{platform.python_version()} ({platform.python_implementation()}, "
        f"{platform.system()} {platform.machine()})" + ("" if py_ok else " - needs 3.9+"))
    row("ok", "executable", sys.executable or "?")

    mods: Dict[str, Any] = {}
    for label, name in (("NumPy", "numpy"), ("SciPy", "scipy"), ("OpenCV", "cv2"),
                        ("Pillow", "PIL"), ("soundfile", "soundfile")):
        try:
            m = importlib.import_module(name)
            mods[name] = m
            extra = ""
            if name == "cv2":
                try:
                    has = m.videoio_registry.hasBackend(m.CAP_FFMPEG)
                except Exception:
                    has = "FFMPEG:                      YES" in m.getBuildInformation()
                extra = "  (video decoding through its own FFmpeg: " \
                        f"{'yes' if has else 'NO'})"
                if not has:
                    row("MISSING", label, f"{m.__version__}{extra}")
                    continue
            elif name == "PIL":
                from PIL import features

                fmts = []
                for f in ("jpg", "zlib", "libtiff", "webp", "avif"):
                    try:
                        if features.check(f):
                            fmts.append({"jpg": "jpeg", "zlib": "png", "libtiff": "tiff"}
                                        .get(f, f))
                    except Exception:
                        pass
                extra = "  (" + " ".join(fmts) + ")"
                if "jpeg" not in fmts or "png" not in fmts:
                    row("MISSING", label, f"{m.__version__}{extra} - JPEG and PNG are needed")
                    continue
            elif name == "soundfile":
                fm = m.available_formats()
                have = [k.lower() for k in ("WAV", "FLAC", "OGG", "MP3") if k in fm]
                extra = f"  ({m.__libsndfile_version__}: {' '.join(have)})"
            row("ok", label, f"{getattr(m, '__version__', '?')}{extra}")
        except Exception as e:  # noqa: BLE001
            row("MISSING", label, f"cannot import {name}: {_one_line(e)}")

    ff = _ffmpeg_facts()
    if not ff["exe"]:
        row("MISSING", "ffmpeg", "no ffmpeg binary: sound tracks of videos and MP4/AAC/Opus "
            "audio cannot be read (pip install imageio-ffmpeg, or put ffmpeg on PATH)")
    elif ff.get("error") or ff["decoders"] is None:
        row("FAILED", "ffmpeg", f"{ff['exe']} did not run: {ff.get('error', 'no output')}")
    else:
        row("ok", "ffmpeg", f"{ff['version']}  ({ff['source']}: {ff['exe']})")
        missing = [k for k, v in ff["decoders"].items() if not v]
        present = [k for k, v in ff["decoders"].items() if v]
        if missing:
            row("warn", "ffmpeg decoders", "missing: " + " ".join(missing) + "; present: "
                + (" ".join(present) or "none") + ". Sound in those formats cannot be read.")
        else:
            row("ok", "ffmpeg decoders", " ".join(present)
                + "  (used for sound tracks and compressed audio)")

    if all(k in mods for k in ("numpy", "scipy", "cv2", "PIL")):
        try:
            ok, text = _self_test()
        except Exception as e:  # noqa: BLE001
            ok, text = False, f"{type(e).__name__}: {_one_line(e)}"
            if args.debug:
                traceback.print_exc()
        row("ok" if ok else "FAILED", "self-test", text)
    else:
        row("FAILED", "self-test", "not run: a required library is missing")

    if problems:
        print(f"{problems} problem{'s' if problems != 1 else ''} found.")
        return 1
    if warnings:
        print(f"Working, with {warnings} warning{'s' if warnings != 1 else ''}.")
        return 1 if args.strict else 0
    print("Everything Basira needs is present and working.")
    return 0


# --------------------------------------------------------------------------- entry point

class _Parser(argparse.ArgumentParser):
    """Exit status 2 is reserved for --fail-on, so a bad command line exits with 1."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def _positive(text: str) -> float:
    v = float(text)
    if not v > 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return v


def _jobs(text: str) -> int:
    v = int(text)
    if v < 0:
        raise argparse.ArgumentTypeError("must be 0 (one per CPU core) or more")
    return v


def build_parser() -> argparse.ArgumentParser:
    ap = _Parser(
        prog="basira", description="Explainable forensics for AI-generated and manipulated "
        "images, audio and video. Reports evidence, not proof.",
        epilog="Exit status of scan: 0 analysed, 1 an input could not be analysed, "
               "2 --fail-on triggered. Guide: docs/USAGE.md")
    ap.add_argument("--version", action="version", version=f"basira {__version__}")
    sub = ap.add_subparsers(dest="cmd", parser_class=_Parser)

    sc = sub.add_parser("scan", help="analyse media files or folders",
                        description="Analyse images, audio and video files. A folder "
                        "contributes the media files in it; other files in it are skipped.")
    sc.add_argument("files", nargs="+", metavar="PATH", help="file, folder or pattern")
    sc.add_argument("-r", "--recursive", action="store_true",
                    help="also search sub-folders of the folders given")
    out = sc.add_argument_group("output")
    fmt = out.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true",
                     help="print JSON to stdout: one object for a single file, an array "
                          "for several (docs/JSON.md)")
    fmt.add_argument("--jsonl", action="store_true",
                     help="print one JSON object per line as each file finishes")
    out.add_argument("--csv", metavar="PATH", help="also write a one-line-per-file CSV summary")
    out.add_argument("--html", metavar="PATH", help="path of the HTML report (single file)")
    out.add_argument("--report-dir", metavar="DIR", help="directory for HTML reports")
    out.add_argument("--no-html", action="store_true", help="do not write an HTML report")
    out.add_argument("--lang", choices=["en", "ar"], default=None,
                     help="language of the HTML report")
    out.add_argument("-q", "--quiet", action="store_true",
                     help="print only errors (and --json / --jsonl output)")
    out.add_argument("--no-color", action="store_true",
                     help="no colours (also: set NO_COLOR, or pipe the output)")
    run = sc.add_argument_group("running")
    run.add_argument("--fail-on", type=int, choices=[1, 2, 3, 4], default=None, metavar="N",
                     help="exit with status 2 if any verdict level is >= N, 1..4 "
                          "(for CI / pipelines)")
    run.add_argument("-j", "--jobs", type=_jobs, default=1, metavar="N",
                     help="analyse N files at once in separate processes (default 1; "
                          "0 = one per CPU core)")
    run.add_argument("--timeout", type=_positive, default=None, metavar="SECONDS",
                     help="give up on a file after this long and carry on with the rest")
    run.add_argument("--debug", action="store_true",
                     help="show the Python traceback behind an error")

    sv = sub.add_parser("serve", help="local web UI (binds to 127.0.0.1)")
    sv.add_argument("--port", type=int, default=8642)
    sv.add_argument("--no-browser", action="store_true")

    dr = sub.add_parser("doctor", help="check the installation and run a self-test",
                        description="Print the versions Basira runs on, where its ffmpeg "
                        "comes from, and analyse a generated image. Exit status 1 if "
                        "something needed is missing or broken.")
    dr.add_argument("--strict", action="store_true", help="treat warnings as failures")
    dr.add_argument("--no-color", action="store_true")
    dr.add_argument("--debug", action="store_true")
    ap.scan_parser = sc
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "scan":
        # Options may come between the files: ``basira scan a.jpg --no-html b.jpg``.
        args = ap.scan_parser.parse_intermixed_args(argv[1:])
        args.cmd = "scan"
    else:
        args = ap.parse_args(argv)
    try:
        if args.cmd == "scan":
            return _scan(args)
        if args.cmd == "doctor":
            return _doctor(args)
        if args.cmd == "serve":
            from .server import serve

            return serve(args.port, not args.no_browser)
    except _UsageError as e:
        ap.error(str(e))
    except KeyboardInterrupt:
        print("basira: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # The reader went away (``basira scan ... | head``): leave quietly.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except Exception:
            pass
        return 1
    ap.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
