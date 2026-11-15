"""Keep shared measured image tables numerically identical in both languages."""
import re
from pathlib import Path


def _tables(text):
    groups, group = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            group.append(line)
        elif group:
            groups.append(group)
            group = []
    return groups


def _numbers(rows, columns=None):
    # Row labels may contain different descriptions, so compare result columns.
    return [re.findall(r"\d+(?:[.,]\d+)*", "|".join(row.split("|")[2:2 + columns] if columns else row.split("|")[2:]))
            for row in rows[2:]]


def test_shared_image_measurement_tables_match():
    root = Path(__file__).resolve().parents[1]
    en, ar = [list(_tables((root / name).read_text()))
              for name in ("README.md", "README.ar.md")]
    # Match stable corpus/row markers; translated result columns remain numeric.
    for en_marker, ar_marker in [
        ("**Columbia", "**Columbia"),
        ("Forged flagged (of 200)", "مزوّرة أُنذر عليها (من 200)"),
        ("**CASIA v1.0**", "**CASIA v1.0**"),
        ("Previous positive-only F1", "F1 السابق للقيم الموجبة فقط"),
        ("CASIA v2, held-out half", "CASIA v2، النصف المحجوز"),
        ("Clean, one save", "سليمة، حفظ واحد"),
        ("Clean (false positives)", "سليمة (إنذار كاذب)"),
        ("Current synthetic video", "الفيديو المصطنع الحالي"),
    ]:
        a = next(t for t in en if en_marker in "\n".join(t))
        b = next(t for t in ar if ar_marker in "\n".join(t))
        columns = 3 if en_marker == "**Columbia" else None
        assert _numbers(a, columns) == _numbers(b, columns), en_marker
