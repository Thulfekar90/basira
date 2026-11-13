from __future__ import annotations

import json
import os

import synth_image as S

from basira.cli import main
from basira.core.io import sniff


def test_scan_writes_html_and_json(tmp_path, capsys):
    path, _ = S.make_splice(str(tmp_path), 0)
    code = main(["scan", path, "--json", "--report-dir", str(tmp_path / "rep")])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["media_type"] == "image" and out["verdict_level"] >= 2
    assert os.path.exists(tmp_path / "rep" / (os.path.basename(path) + ".basira.html"))


def test_fail_on_sets_exit_status(tmp_path):
    path, _ = S.make_splice(str(tmp_path), 1)
    assert main(["scan", path, "--no-html", "--fail-on", "2"]) == 2
    clean, _ = S.make_authentic(str(tmp_path), 0)
    assert main(["scan", clean, "--no-html", "--fail-on", "2"]) == 0


def test_missing_file_is_an_error(tmp_path):
    assert main(["scan", str(tmp_path / "nope.jpg"), "--no-html"]) == 1


def test_sniff_by_magic_bytes(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 60)
    assert sniff(str(p)) == "image"
    p.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\0" * 48)
    assert sniff(str(p)) == "audio"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\0" * 52)
    assert sniff(str(p)) == "video"
