"""Offline sanity check of system artwork, NOT a clean-camera forensic dataset.

Example (assets remain outside this repository)::

    python benchmarks/system_images.py --root /System/Library --limit 600 \
        --output /tmp/system-images.json

Supported paths are sorted, filtered by headers and the loader's existing limits,
then frozen with content hashes before inference. Every candidate exclusion and
analysis failure is recorded. Flags are not called false positives: system artwork
can be edited, rendered, duplicated, or generated and has no forensic ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import basira  # noqa: E402
from basira.image.load import (DENSITY_CHECK_PIXELS, MAX_IMAGE_BYTES,  # noqa: E402
                               MAX_IMAGE_PIXELS, MAX_PIXELS_PER_BYTE, _limit)

EXTENSIONS = {".png", ".jpg", ".jpeg", ".jpe", ".tif", ".tiff", ".bmp", ".webp"}
FORMATS = ("PNG", "JPEG", "TIFF", "BMP", "WEBP")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    return {str(path.relative_to(REPO)): sha256(path)
            for path in sorted((REPO / "basira").rglob("*.py"))}


def write_json(path, value, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def select(root, limit):
    candidates, walk_errors = [], []

    def onerror(exc):
        walk_errors.append({"path": exc.filename, "error": str(exc)})

    # Do not follow directory symlinks; record them explicitly, as they can lead outside
    # the requested system tree. File symlinks are excluded below for the same reason.
    directory_symlinks = []
    for folder, directories, files in os.walk(root, followlinks=False, onerror=onerror):
        for name in directories:
            path = Path(folder) / name
            if path.is_symlink():
                directory_symlinks.append(str(path))
        candidates.extend(Path(folder) / name for name in files
                          if Path(name).suffix.lower() in EXTENSIONS)
    max_pixels = int(_limit("BASIRA_MAX_IMAGE_PIXELS", MAX_IMAGE_PIXELS))
    max_density = _limit("BASIRA_MAX_PIXELS_PER_BYTE", MAX_PIXELS_PER_BYTE)
    selected, excluded = [], []
    Image.init()
    for path in sorted(candidates):
        item = {"path": str(path)}
        if len(selected) >= limit:
            excluded.append({**item, "reason": "selection_limit_reached_header_not_read"})
            continue
        try:
            if path.is_symlink():
                excluded.append({**item, "reason": "symlink"})
                continue
            if not path.is_file():
                excluded.append({**item, "reason": "not_regular_file"})
                continue
            size = path.stat().st_size
            item["bytes"] = size
            if size > MAX_IMAGE_BYTES:
                excluded.append({**item, "reason": "file_byte_cap"})
                continue
            with warnings.catch_warnings(record=True) as caught:
                with Image.open(path, formats=FORMATS) as image:
                    width, height = image.size
                    item.update(width=width, height=height, format=image.format)
                if caught:
                    item["header_warnings"] = [str(w.message) for w in caught]
            if min(width, height) < 64:
                excluded.append({**item, "reason": "side_below_64"})
                continue
            pixels = width * height
            if pixels > max_pixels:
                excluded.append({**item, "reason": "image_pixel_cap"})
                continue
            if max_density and pixels > DENSITY_CHECK_PIXELS and pixels > max_density * max(size, 1):
                excluded.append({**item, "reason": "image_density_cap"})
                continue
            item["sha256"] = sha256(path)
            selected.append(item)
        except Exception as exc:
            excluded.append({**item, "reason": "header_or_read_error",
                             "error": f"{type(exc).__name__}: {exc}"})
    return {"selected": selected, "excluded": excluded, "walk_errors": walk_errors,
            "directory_symlinks_not_followed": sorted(directory_symlinks),
            "candidate_count": len(candidates),
            "limits": {"minimum_side": 64, "file_bytes": MAX_IMAGE_BYTES,
                       "image_pixels": max_pixels, "density_check_pixels": DENSITY_CHECK_PIXELS,
                       "pixels_per_byte": max_density}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", default="/System/Library")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.limit < 1 or not Path(args.root).is_dir():
        parser.error("--limit must be positive and --root an existing directory")
    output = args.output
    manifest_path = output.with_name(output.stem + ".inputs.json")
    rows_path = output.with_name(output.stem + ".rows.jsonl")
    if any(path.exists() for path in (output, manifest_path, rows_path)):
        parser.error("evidence output already exists; use a new --output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source_before = source_hashes()
    manifest = {"root": str(Path(args.root).resolve()), "requested_limit": args.limit,
                "corpus": "System artwork; no forensic ground truth; not clean camera originals",
                "ordering": "lexicographic path; first eligible files; not a random sample",
                "started_utc": started, "source_sha256": source_before,
                **select(args.root, args.limit)}
    write_json(manifest_path, manifest, exclusive=True)
    manifest_sha = sha256(manifest_path)
    selected_sha = hashlib.sha256(json.dumps(manifest["selected"], sort_keys=True,
                                             ensure_ascii=False).encode()).hexdigest()
    print(f"FROZEN before inference: {manifest_path}", flush=True)
    print(f"manifest_sha256={manifest_sha}\nselected_list_sha256={selected_sha}", flush=True)
    counts = Counter(selected=len(manifest["selected"]), candidates=manifest["candidate_count"],
                     exclusions=len(manifest["excluded"]), walk_errors=len(manifest["walk_errors"]))
    exclusion_counts = dict(Counter(item["reason"] for item in manifest["excluded"]))
    print("selection=" + json.dumps(dict(counts), sort_keys=True), flush=True)
    print("exclusions=" + json.dumps(exclusion_counts, sort_keys=True), flush=True)
    # Initialize zero counters so absence is explicit in the result schema.
    for key in ("completed", "analysis_failures", "input_hash_changed", "verdict_ge_2",
                "manipulation_ge_0_45", "reports_with_detector_failures", "jpeg_trace_missing",
                "jpeg_trace_error", "jpeg_trace_not_applicable_summary", "jpeg_trace_returned_evidence",
                "jpeg_trace_positive_confidence", "jpeg_trace_has_statistic", "jpeg_trace_score_ge_0_5"):
        counts[key] = 0
    statistics, rows = [], []
    with rows_path.open("x", encoding="utf-8") as stream:
        for index, item in enumerate(manifest["selected"], 1):
            t0 = time.monotonic()
            row = {"path": item["path"], "input_sha256": item["sha256"]}
            try:
                if sha256(item["path"]) != item["sha256"]:
                    counts["input_hash_changed"] += 1
                    raise ValueError("input bytes changed after selection was frozen")
                report = basira.analyze(item["path"])
                counts["completed"] += 1
                counts["verdict_ge_2"] += int(report.verdict_level >= 2)
                counts["manipulation_ge_0_45"] += int(report.scores["manipulation"] >= 0.45)
                errors = [{"detector": e.detector, "error": e.error} for e in report.evidences if e.error]
                counts["reports_with_detector_failures"] += bool(errors)
                trace = report.get("image.jpeg_trace")
                if trace is None:
                    state = "missing"
                elif trace.error:
                    state = "error"
                elif trace.summary.startswith("Not applicable"):
                    state = "not_applicable_summary"
                else:
                    state = "returned_evidence"
                counts["jpeg_trace_" + state] += 1
                if trace is not None:
                    counts["jpeg_trace_positive_confidence"] += int(trace.confidence > 0)
                    counts["jpeg_trace_has_statistic"] += int("statistic" in trace.details)
                    counts["jpeg_trace_score_ge_0_5"] += int(not trace.error and trace.score >= 0.5)
                    if "statistic" in trace.details:
                        statistics.append(float(trace.details["statistic"]))
                row.update(status="report", verdict_level=report.verdict_level,
                           verdict=report.verdict, scores=report.scores,
                           detector_failures=errors,
                           fired=[e.detector for e in report.evidences
                                  if not e.error and e.score >= 0.5 and e.confidence > 0],
                           jpeg_trace_state=state,
                           jpeg_trace=trace.to_dict() if trace else None)
            except Exception as exc:
                counts["analysis_failures"] += 1
                row.update(status="analysis_failure", error=f"{type(exc).__name__}: {exc}")
            row["elapsed_seconds"] = round(time.monotonic() - t0, 6)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if index % 25 == 0 or index == len(manifest["selected"]):
                print(f"analysed_attempts={index}/{len(manifest['selected'])}", flush=True)
    source_unchanged = source_hashes() == source_before
    summary = {"corpus": manifest["corpus"], "started_utc": started,
               "input_manifest": str(manifest_path), "input_manifest_sha256": manifest_sha,
               "selected_list_sha256": selected_sha, "rows": str(rows_path),
               "rows_sha256": sha256(rows_path), "source_unchanged": source_unchanged,
               "counts": dict(counts), "exclusion_counts": exclusion_counts,
               "jpeg_trace_max_statistic": max(statistics) if statistics else None,
               "applicability_definition": "summary starts Not applicable; raw confidence and statistic availability also counted",
               "per_file": rows}
    write_json(output, summary, exclusive=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_file"}, indent=2), flush=True)
    return int(not counts["selected"] or counts["analysis_failures"] > 0 or
               counts["reports_with_detector_failures"] > 0 or not source_unchanged)


if __name__ == "__main__":
    raise SystemExit(main())
