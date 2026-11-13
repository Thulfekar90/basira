"""Video engine: temporal forgeries must be flagged at the right time; clean clips must not."""
from __future__ import annotations

import pytest

import basira
from basira.core.io import ffmpeg_exe
from basira.video.container import analyze_double_compression, parse_mp4

import synth_video as V

pytestmark = pytest.mark.skipif(ffmpeg_exe() is None, reason="ffmpeg not available")


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    return str(tmp_path_factory.mktemp("vid"))


def _near(segments, t, tol=0.25):
    return any(s.start - tol <= t <= s.end + tol for s in segments)


@pytest.mark.parametrize("seed", [0, 1])
def test_clean_clip_is_not_flagged(out, seed):
    path, _ = V.make_clean(out, seed)
    r = basira.analyze(path)
    assert r.media_type == "video" and r.verdict_level == 0, (r.verdict, r.scores)
    assert not r.fused_segments
    assert all(e.error is None for e in r.evidences)


def test_removed_frames_are_located(out):
    path, truth = V.make_deletion(out, 1)
    r = basira.analyze(path)
    ev = r.get("video.jump")
    assert ev.score > 0.5 and _near(ev.segments, truth[0])
    assert r.verdict_level >= 2


def test_replayed_footage_is_located(out):
    path, truth = V.make_repeat(out, 2)
    r = basira.analyze(path)
    ev = r.get("video.repeat")
    assert ev.score > 0.5 and _near(ev.segments, truth[0])
    assert abs(ev.details["repeats"][0]["source_s"] - 80 / V.FPS) < 0.2
    assert r.verdict_level >= 3


def test_frozen_frames_are_located(out):
    path, truth = V.make_freeze(out, 3)
    r = basira.analyze(path)
    ev = r.get("video.freeze")
    assert ev.score > 0.5 and _near(ev.segments, truth[0])


def test_inserted_foreign_footage_is_located(out):
    path, truth = V.make_insert(out, 4)
    r = basira.analyze(path)
    assert r.verdict_level >= 2
    assert _near(r.fused_segments, truth[0]) and _near(r.fused_segments, truth[1])
    # video.source is deliberately not asserted: H.264 flattens most of the noise
    # difference between sources, so it is only weak corroborating evidence.
    assert r.get("video.source").error is None


@pytest.mark.parametrize("gop1", [12, 18])
def test_double_compression_reveals_previous_gop(out, gop1):
    fr = V.make_frames(300, 7)
    single = V.encode(fr, f"{out}/single_{gop1}.mp4", gop=30, bframes=0)
    first = V.encode(fr, f"{out}/first_{gop1}.mp4", gop=gop1, crf=23, bframes=0)
    double = V.reencode(first, f"{out}/double_{gop1}.mp4", gop=30, bframes=0)
    e1 = analyze_double_compression(parse_mp4(single))
    e2 = analyze_double_compression(parse_mp4(double))
    assert e1.score < 0.5, e1.details
    assert e2.score > 0.5 and e2.details["candidate_period"] == gop1, e2.details


def test_generator_tag_in_container_is_decisive(out):
    path, _ = V.make_tagged(out, 6)
    r = basira.analyze(path)
    assert r.verdict_level == 4 and r.scores["ai_generation"] > 0.9


def test_audio_track_becomes_child_report(out):
    """A video with an edited audio track: the child audio report must carry the finding."""
    import os
    import subprocess

    import numpy as np
    import soundfile as sf

    import synth_audio as A

    names = [n for n in dir(A) if n.startswith("make_") or n.startswith("forge_")]
    assert names, "synthetic audio generators missing"
    sr = 48000
    rng = np.random.RandomState(0)
    t = np.arange(8 * sr) / sr
    x = 0.2 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.01 * rng.randn(len(t))
    x[4 * sr:int(4.3 * sr)] = 0.0                      # 0.3 s of digital silence
    wav = os.path.join(out, "track.wav")
    sf.write(wav, x.astype("float32"), sr)
    vid, _ = V.make_clean(out, 9)
    muxed = os.path.join(out, "with_audio.mp4")
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", vid, "-i", wav,
                    "-c:v", "copy", "-c:a", "pcm_s16le", "-shortest", muxed.replace(".mp4", ".mov")],
                   check=True)
    r = basira.analyze(muxed.replace(".mp4", ".mov"))
    assert r.children and r.children[0].media_type == "audio"
    sil = r.children[0].get("audio.silence")
    assert sil is not None and sil.score > 0.5 and _near(sil.segments, 4.0, 0.4)
    assert r.scores["manipulation"] > 0.3
    # The finding must surface on the video report itself, not only in the child.
    assert any("audio track" in s.label and abs(s.start - 4.0) < 0.5 for s in r.fused_segments)
    assert any(x.startswith("[audio track]") for x in r.explanation)
