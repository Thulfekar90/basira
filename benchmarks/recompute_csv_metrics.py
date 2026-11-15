"""Correct historical localisation aggregates from saved CSV, without analysing media.

    python benchmarks/recompute_csv_metrics.py saved-run.csv [another-run.csv ...]

The old conditional mean selected positive F1 only, omitting flagged zero-overlap
forgeries. The corrected denominator is every forged row with verdict level >= 2.
Blank pixel scores are conservatively zero and counted explicitly; CSV values are
rounded historical measurements, not a recomputation of pixel maps. Input files are
read-only. This module imports no Basira engine and never opens dataset images.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean


def summarize_csv(path):
    path = Path(path)
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    required = {"file", "forged", "level", "pixel_f1"}
    if not required <= set(reader.fieldnames or []):
        raise ValueError("CSV lacks required fields: " + ", ".join(sorted(required)))
    total = clean = flagged = missing = flagged_missing = 0
    scores, selected, historical = [], [], []
    for line, row in enumerate(reader, start=2):
        total += 1
        if row["forged"] not in ("0", "1"):
            raise ValueError(f"Invalid forged value at row {line}")
        level = int(row["level"])
        if level not in range(5):
            raise ValueError(f"Invalid verdict level at row {line}")
        if row["forged"] == "0":
            clean += 1
            continue
        field = row["pixel_f1"]
        value = float(field) if field else 0.0
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid pixel_f1 at row {line}: {field}")
        missing += not bool(field)
        scores.append(value)
        if value > 0:
            historical.append(value)
        if level >= 2:
            flagged += 1
            flagged_missing += not bool(field)
            selected.append(value)
    return {"csv": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": total, "clean": clean, "forged": len(scores),
            "flagged_forged": flagged, "missing_mask_scores": missing,
            "flagged_missing_mask_scores": flagged_missing,
            "flagged_zero_f1": sum(x == 0 for x in selected),
            "historical_positive_only_f1": mean(historical) if historical else None,
            "corrected_flagged_f1": mean(selected) if selected else None,
            "all_forged_f1": mean(scores) if scores else None,
            "measurement": "saved rounded CSV only; no detector or held-out media rerun"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+")
    args = parser.parse_args(argv)
    for path in args.csv:
        print(json.dumps(summarize_csv(path), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
