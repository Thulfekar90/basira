"""Video engine under realistic conditions (synthetic clips; the real-footage numbers come
from ``benchmarks/video_robustness.py``): natural events must stay unflagged, frame-rate
conversion is not an edit, and the forgeries the engine claims must survive delivery."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from basira.core.io import ffmpeg_exe
from basira.video import analyze_video

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks"))
import video_robustness as B  # noqa: E402

pytestmark = pytest.mark.skipif(ffmpeg_exe() is None, reason="ffmpeg not available")

FPS = 30
SHOT_EVENTS = ("cuts", "cuts-similar", "dissolve", "fade-black", "flashes", "exposure",
               "dup-slowmo", "pulldown")
CAMERA_EVENTS = ("whip", "panstart", "bump")


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """One camera clip (decoded again, as an editor would see it) and a few shots."""
    work = str(tmp_path_factory.mktemp("robust"))
    src, oth = B.SynthSource(0), B.SynthSource(50)
    size = src.size(360)
    cam = os.path.join(work, "cam.mp4")
    a = B._camera_roundtrip(src.segment(3.0, 8.0, FPS, size), cam, FPS, 20, FPS, 0)
    o = B._camera_roundtrip(oth.segment(9.0, 2.0, FPS, size), os.path.join(work, "o.mp4"), FPS,
                            23, FPS, 0)
    shots = {"a": src.segment(1.0, 6.0, FPS, size), "b": oth.segment(2.0, 6.0, FPS, size),
             "c": src.segment(1.0, 6.0, FPS, size, "detail"),
             "d": src.segment(1.0, 6.0, FPS, size, "detail2")}
    return {"work": work, "cam": cam, "a": a, "o": o, "shots": shots, "src": src, "size": size}


def _quiet(path: str) -> None:
    r = analyze_video(path, analyze_audio_track=False)
    fired = {e.detector: round(e.score, 2) for e in r.evidences if e.score >= 0.5 and e.confidence > 0}
    assert r.verdict_level == 0 and not r.fused_segments, (r.verdict, fired,
                                                           [s.to_dict() for s in r.fused_segments])
    assert all(e.error is None for e in r.evidences)


def _forged(lab, what: str, cond: str = "orig"):
    frames, truth = B.forge(what, lab["a"], lab["o"], FPS)
    base = os.path.join(lab["work"], f"{what}.mp4")
    if not os.path.exists(base):
        B.encode(frames, base, FPS, crf=20)
    path = B.deliver(base, os.path.join(lab["work"], f"{what}__{cond}.mp4"), cond, FPS)
    r = analyze_video(path, analyze_audio_track=False)
    segs = [(s.start, s.end, s.label) for s in r.fused_segments]
    return r, B.located(segs, truth, r.info["duration"])


@pytest.mark.parametrize("what", SHOT_EVENTS)
def test_natural_events_between_shots_are_not_reported(lab, what):
    path = os.path.join(lab["work"], f"nat_{what}.mp4")
    B.encode(B.natural_from_shots(what, lab["shots"], FPS), path, FPS, crf=20, gop=FPS)
    _quiet(path)


@pytest.mark.parametrize("mode", CAMERA_EVENTS)
def test_camera_movement_is_not_reported(lab, mode):
    path = os.path.join(lab["work"], f"nat_{mode}.mp4")
    B.encode(lab["src"].segment(2.0, 6.0, FPS, lab["size"], mode), path, FPS, crf=20, gop=FPS)
    _quiet(path)


@pytest.mark.parametrize("what", ["tripod-object", "periodic", "screen-recording"])
def test_static_and_repetitive_content_is_not_reported(lab, what):
    still = lab["shots"]["a"][0]
    frames = {"tripod-object": lambda: B.tripod_object(still, FPS),
              "periodic": lambda: B.periodic_motion(still, FPS),
              "screen-recording": lambda: B.screen_recording(FPS, seconds=8.0, w=640, h=360)}[what]()
    path = os.path.join(lab["work"], f"nat_{what}.mp4")
    B.encode(frames, path, FPS, crf=20, gop=FPS)
    _quiet(path)
    _quiet(B.deliver(path, path.replace(".mp4", "__msg.mp4"), "msg", FPS))


@pytest.mark.parametrize("cond", ["crf35", "half", "fps-a", "fps-b", "bframes", "msg"])
def test_delivery_does_not_make_a_clean_clip_suspicious(lab, cond):
    """Re-encoding, downscaling and frame-rate conversion (which drops or duplicates frames
    by itself) are not edits."""
    _quiet(B.deliver(lab["cam"], os.path.join(lab["work"], f"clean__{cond}.mp4"), cond, FPS))


def test_frame_rate_upconversion_is_not_an_edit(lab):
    """25 -> 30 fps repeats every fifth frame."""
    src25 = os.path.join(lab["work"], "cam25.mp4")
    B.encode(lab["a"][:200], src25, 25, crf=20, gop=25)
    _quiet(B.deliver(src25, os.path.join(lab["work"], "cam25__30.mp4"), "fps-a", 25))


@pytest.mark.parametrize("what", ["del15", "replay", "freeze"])
@pytest.mark.parametrize("cond", ["orig", "crf28", "fps-a", "msg"])
def test_forgeries_survive_delivery(lab, what, cond):
    r, (hit, stray) = _forged(lab, what, cond)
    assert hit and stray == 0, (what, cond, [s.to_dict() for s in r.fused_segments])
    assert r.verdict_level >= 1


def test_what_the_engine_cannot_see_is_said_not_guessed(lab):
    """A speed change and footage cut in with hard cuts are not detectable from frame
    timing; the engine must stay quiet about them rather than guess, and say so."""
    r, _ = _forged(lab, "speed")
    assert r.get("video.jump").score < 0.5 and r.get("video.freeze").score < 0.5
    r, _ = _forged(lab, "insert")
    jump = r.get("video.jump")
    assert jump.score < 0.5 and jump.details["scene_cuts"] == 2
    assert "cannot be seen" in jump.summary


def test_located_requires_an_edge_near_the_truth():
    truth = [(4.0, 4.6)]
    assert B.located([(3.9, 4.6, "")], truth, 8.0) == (True, 0)
    assert B.located([(1.0, 1.2, "")], truth, 8.0) == (False, 1)
    assert B.located([(0.0, 8.0, "")], truth, 8.0)[0] is False      # covers everything: no credit


def test_benchmark_fails_without_ffmpeg(monkeypatch, capsys):
    monkeypatch.setattr(B, "ffmpeg_exe", lambda: None)
    # R-D002-04: an unmeasured benchmark must not appear to have passed.
    assert B.main(["--source", "/nonexistent/clip.mov"]) != 0
    assert "nothing to measure" in capsys.readouterr().out


def test_high_frame_rate_footage_is_subsampled(lab):
    """Frames beyond 60 per second are decoded but not converted (the costly part)."""
    from basira.video.temporal import scan_frames

    path = os.path.join(lab["work"], "fast.mp4")
    B.encode(np.repeat(lab["a"][:60], 2, axis=0), path, 120, crf=23)
    fs = scan_frames(path)
    assert fs.stride == 2 and len(fs.idx) == 60 and abs(fs.rate - 60.0) < 1.0


def test_scheme_like_relative_name_is_decoded_as_a_file(lab, tmp_path, monkeypatch):
    """``pipe:0.mp4`` (or ``concat:``, ``http:``) given as a relative name must be read as
    the local file, not handed to FFmpeg as a protocol URL."""
    import shutil

    shutil.copy(lab["cam"], tmp_path / "pipe:0.mp4")
    monkeypatch.chdir(tmp_path)
    r = analyze_video("pipe:0.mp4", analyze_audio_track=False)
    assert r.get("video.decode") is None and r.info.get("frames", 0) >= 200
    assert r.verdict_level == 0


def test_oversized_frames_are_refused_not_decoded(lab, monkeypatch):
    import basira.video.temporal as T

    monkeypatch.setattr(T, "MAX_FRAME_PIXELS", 1000)
    r = analyze_video(lab["cam"], analyze_audio_track=False)
    dec = r.get("video.decode")
    assert dec is not None and "MP limit" in dec.summary
    assert r.verdict_level >= 1 and "Inconclusive" in r.verdict
