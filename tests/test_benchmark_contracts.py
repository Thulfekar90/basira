"""Benchmark denominators, dataset isolation, and incomplete-run exit contracts."""
import csv
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
import public_datasets as public
import audio_robustness as audio
import video_robustness as video


def test_flagged_localisation_mean_includes_zero_overlap(capsys):
    public.summarise("fixture", [(0, 0.0)], [(2, 0.6), (3, 0.9), (0, 0.0)],
                     [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                     {"authentic": {}, "forged": {}}, time.time())
    output = capsys.readouterr().out
    assert "pixel F1, flagged forged only       0.500" in output


def test_flagged_localisation_mean_prints_all_zero_case(capsys):
    public.summarise("fixture", [(0, 0.0)], [(2, 0.6)], [0.0], [0.0],
                     {"authentic": {}, "forged": {}}, time.time())
    assert "pixel F1, flagged forged only       0.000" in capsys.readouterr().out


def test_missing_mask_keeps_forged_denominator_and_is_disclosed(monkeypatch, capsys, tmp_path):
    def report(path):
        return SimpleNamespace(verdict_level=0 if path == "clean" else 2,
                               scores={"manipulation": 0.0 if path == "clean" else 0.6},
                               evidences=[], fused_heatmap=np.ones((2, 2)))
    monkeypatch.setattr(public.basira, "analyze", report)
    out = tmp_path / "scores.csv"
    public.run([("clean", False, None), ("missing", True, None),
                ("located", True, np.ones((2, 2), dtype=bool))], "fixture", str(out))
    output = capsys.readouterr().out
    assert "1 forged file(s) have no usable mask; localisation scored as zero" in output
    assert "pixel F1, flagged forged only       0.500" in output
    rows = list(csv.DictReader(out.open()))
    assert rows[1]["pixel_f1"] == "0.0000"


@pytest.fixture
def coverage_tree(tmp_path):
    (tmp_path / "image").mkdir()
    (tmp_path / "mask").mkdir()
    for target in (1, 2, 3, 4):
        for suffix in ("", "t"):
            (tmp_path / "image" / (str(target) + suffix + ".tif")).touch()
        (tmp_path / "mask" / (str(target) + "forged.tif")).touch()
    return tmp_path


def test_coverage_parity_split_keeps_target_pairs(monkeypatch, coverage_tree):
    captured = {}
    monkeypatch.setattr(public, "run", lambda pairs, name, csv_path: captured.update(pairs=pairs, name=name))
    ids = {}
    for half, wanted in (("dev", {"1", "3"}), ("test", {"2", "4"})):
        public.coverage(str(coverage_tree), 0, None, half=half)
        stems = {Path(p[0]).stem for p in captured["pairs"]}
        assert stems == wanted | {x + "t" for x in wanted}
        ids[half] = stems
    assert ids["dev"].isdisjoint(ids["test"])
    assert len(ids["dev"] | ids["test"]) == 8


def test_coverage_limit_preserves_original_and_forgery(monkeypatch, coverage_tree):
    captured = []
    monkeypatch.setattr(public, "run", lambda pairs, *args: captured.extend(pairs))
    public.coverage(str(coverage_tree), 1, None, half="dev")
    assert len(captured) == 2
    assert {Path(p[0]).stem.rstrip("t") for p in captured} == {"1"}


def test_coverage_half_reaches_dataset_dispatch(monkeypatch, coverage_tree):
    captured = []
    monkeypatch.setattr(public, "coverage", lambda *args: captured.append(args))
    assert public.main(["coverage", "--root", str(coverage_tree), "--half", "dev"]) == 0
    assert captured[0][-1] == "dev"


@pytest.mark.parametrize("dataset", ["columbia", "comofod", "casia1"])
def test_unsupported_half_rejected_before_dataset_access(dataset, capsys):
    with pytest.raises(SystemExit) as err:
        public.main([dataset, "--root", "absent", "--half", "dev"])
    assert err.value.code == 2
    assert "--half applies to casia2 or coverage only" in capsys.readouterr().err


def test_coverage_rejects_non_numeric_ids_before_split(monkeypatch, coverage_tree):
    (coverage_tree / "image" / "unexpected.tif").touch()
    monkeypatch.setattr(public, "run", lambda *args: pytest.fail("must not analyze an ambiguous split"))
    with pytest.raises(ValueError, match="COVERAGE target id"):
        public.coverage(str(coverage_tree), 0, None, half="dev")


def test_video_missing_ffmpeg_is_not_success(monkeypatch, capsys):
    monkeypatch.setattr(video, "ffmpeg_exe", lambda: None)
    assert video.main(["--quick"]) != 0
    assert "ffmpeg" in capsys.readouterr().out


@pytest.mark.parametrize("include_existing", [False, True])
def test_video_missing_requested_source_never_falls_back(monkeypatch, tmp_path, include_existing):
    monkeypatch.setattr(video, "ffmpeg_exe", lambda: "fake-ffmpeg")
    monkeypatch.setattr(video, "SynthSource", lambda *args: pytest.fail("silently fell back to synthetic"))
    monkeypatch.setattr(video, "RealSource", lambda *args: pytest.fail("partial corpus must not run"))
    args = ["--source", str(tmp_path / "missing.mov")]
    if include_existing:
        existing = tmp_path / "exists.mov"
        existing.touch()
        args += ["--source", str(existing)]
    assert video.main(args) != 0


def test_video_failed_clip_does_not_report_success(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "ffmpeg_exe", lambda: "fake-ffmpeg")
    monkeypatch.setattr(video, "build_jobs", lambda *args: [{"fixture": True}])
    monkeypatch.setattr(video, "run_item", lambda job: [{"cond": "ERROR", "error": "decode failed"}])
    monkeypatch.setattr(video, "table", lambda *args: "fixture error")
    assert video.main(["--quick", "--jobs", "1", "--work", str(tmp_path)]) != 0


def test_audio_missing_ffmpeg_stops_before_any_section(monkeypatch):
    monkeypatch.setattr(audio, "ffmpeg_exe", lambda: None)
    monkeypatch.setattr(audio, "run_matrix", lambda *args: pytest.fail("ran without prerequisite"))
    monkeypatch.setattr(sys, "argv", ["audio_robustness.py", "--only", "matrix"])
    assert audio.main() != 0


def test_audio_missing_codec_cannot_shrink_matrix(monkeypatch):
    monkeypatch.setattr(audio, "_encoders", lambda: " A..... aac AAC encoder\n")
    with pytest.raises(RuntimeError, match="required audio encoders"):
        audio.conditions()


def test_audio_tts_requires_available_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "available_voices", lambda: [])
    monkeypatch.setattr(audio, "conditions", lambda: pytest.fail("must reject no voices first"))
    with pytest.raises(RuntimeError, match="No usable.*voices"):
        audio.run_tts(SimpleNamespace(set="dev"), str(tmp_path))


def test_audio_missing_nonspeech_source_is_not_success(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "NONSPEECH_SOURCES", [("missing", str(tmp_path / "*.wav"))])
    with pytest.raises(RuntimeError, match="missing"):
        audio.run_nonspeech(SimpleNamespace(max_files=1, verbose=False), str(tmp_path))


def test_video_refuses_unidentified_cached_corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "ffmpeg_exe", lambda: "fake-ffmpeg")
    (tmp_path / "jobs.json").write_text('[{"source":"some earlier corpus"}]')
    monkeypatch.setattr(video, "run_item", lambda job: pytest.fail("used unverifiable cached corpus"))
    assert video.main(["--quick", "--jobs", "1", "--work", str(tmp_path)]) != 0


def _csv_metrics_module():
    spec = importlib.util.spec_from_file_location("recompute_csv_metrics", BENCHMARKS / "recompute_csv_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_csv_correction_uses_flags_and_retains_zero_missing_scores(tmp_path):
    path = tmp_path / "historical.csv"
    path.write_text("file,forged,level,pixel_f1\nclean,0,0,\nzero,1,2,0.0000\n"
                    "hit,1,3,1.0000\nmiss,1,0,0.0000\nmissing,1,2,\n")
    result = _csv_metrics_module().summarize_csv(path)
    assert result["flagged_forged"] == 3
    assert result["flagged_missing_mask_scores"] == 1
    assert result["historical_positive_only_f1"] == 1.0
    assert result["corrected_flagged_f1"] == pytest.approx(1 / 3)
    assert result["all_forged_f1"] == 0.25
    assert path.read_text().endswith("missing,1,2,\n")


@pytest.mark.parametrize("value", ["nan", "inf", "-0.01", "1.01"])
def test_csv_correction_rejects_invalid_scores(tmp_path, value):
    path = tmp_path / "invalid.csv"
    path.write_text("file,forged,level,pixel_f1\nimage,1,2," + value + "\n")
    with pytest.raises(ValueError, match="pixel_f1"):
        _csv_metrics_module().summarize_csv(path)
