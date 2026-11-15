"""Unsigned declarations are strong leads, never verified provenance (D003/Q02)."""
import hashlib
import os

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import basira
from basira.core.fusion import finalize
from basira.core.report_html import render_html
from basira.core.types import Category, Evidence, Report
import synth_image as S

CLAIM = "Metadata declares AI generation (unverified)"
AR_CLAIM = "البيانات الوصفية تصرّح بأنه مولّد بالذكاء الاصطناعي (غير متحقق منه)"


def _check_tagged_pixels(tmp_path, pixels, key, value):
    path = tmp_path / "declared.png"
    meta = PngInfo()
    meta.add_text(key, value)
    Image.fromarray(pixels).save(path, pnginfo=meta)
    assert hashlib.sha256(Image.open(path).tobytes()).digest() == hashlib.sha256(pixels.tobytes()).digest()
    r = basira.analyze(str(path))
    e = r.get("image.generator_metadata")
    assert r.verdict_level == 3, r.verdict
    assert r.verdict == CLAIM
    assert not e.decisive and e.details["unverified_generator_claim"] is True
    assert "unverified" in e.summary.lower()
    assert CLAIM in render_html(r, "en")
    assert AR_CLAIM in render_html(r, "ar")


@pytest.mark.parametrize("key,value", [
    ("Software", "NovelAI"),
    ("prompt", '{"3":{"class_type":"KSampler","inputs":{"seed":1}}}'),
    ("parameters", "A landscape\nSteps: 30, Sampler: Euler, Seed: 1"),
])
def test_unsigned_image_tags_on_identical_pixels(tmp_path, key, value):
    _check_tagged_pixels(tmp_path, S.camera_photo(2, 240, 320), key, value)


def test_unsigned_tag_on_external_camera_photo(tmp_path):
    """Optional real-photo gate; input remains external and no dataset is shipped."""
    source = os.environ.get("BASIRA_REAL_PHOTO")
    if not source:
        pytest.skip("BASIRA_REAL_PHOTO not supplied; synthetic contract tests still run")
    with Image.open(source) as im:
        pixels = np.asarray(im.convert("RGB"))
    _check_tagged_pixels(tmp_path, pixels, "Software", "NovelAI")


@pytest.mark.parametrize("failed,confidence", [(False, 1.0), (True, 1.0), (False, 0.0)])
def test_legacy_decisive_flag_cannot_claim_a_verified_signature(failed, confidence):
    e = Evidence("custom.metadata", "Metadata", Category.AI_GENERATION, .97, confidence,
                 "unsigned", decisive=True, error="failed" if failed else None,
                 details={"signature_verified": True})  # user-supplied detail is not a verifier
    r = finalize(Report("fixture", "image", {}, [e]))
    assert r.verdict_level < 4
    if failed:
        assert r.verdict_level == 1 and r.info["detectors_failed"] == 1


def test_empty_content_is_not_a_clean_verdict():
    r = finalize(Report("empty.wav", "audio", {"content_empty": True}, [
        Evidence.not_applicable("audio.empty", "Empty audio", Category.MANIPULATION, "empty audio")]))
    assert r.verdict_level == 1
    assert r.verdict == "No content to analyse"
    assert "لا يوجد محتوى قابل للتحليل" in render_html(r, "ar")


def test_child_unsigned_claim_is_visible_in_parent_verdict():
    e = Evidence("audio.metadata", "Metadata", Category.AI_GENERATION, .99, 1.0,
                 "Metadata declares AI generation (unverified)",
                 details={"unverified_generator_claim": True})
    child = finalize(Report("audio.wav", "audio", {}, [e]))
    r = finalize(Report("video.mp4", "video", {}, [], children=[child]))
    assert r.verdict_level == 3 and r.verdict == CLAIM


def test_video_c2pa_text_marker_is_an_unverified_claim(tmp_path, monkeypatch):
    from basira.video import container
    path = tmp_path / "marker.mp4"
    path.write_bytes(b"\x00\x00\x00\x20ftypisom\n c2pa.ai_generated \n")
    monkeypatch.setattr(container, "ffmpeg_probe_text", lambda _: "")
    monkeypatch.setattr(container, "parse_mp4", lambda _: None)
    evidences, info = container.analyze_container(str(path))
    report = finalize(Report(str(path), "video", info, evidences))
    assert report.verdict_level == 3 and report.verdict == CLAIM
    assert report.get("video.generator_metadata").details["unverified_generator_claim"]


def test_arabic_report_does_not_call_unsigned_metadata_decisive():
    report = finalize(Report("fixture", "image", {}, []))
    assert "ما لم تكن البيانات الوصفية قاطعة" not in render_html(report, "ar")
