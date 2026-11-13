"""Image engine: every detector must stay quiet on clean images and fire, in the right
place, on the forgery it is designed for."""
from __future__ import annotations

import numpy as np
import pytest

import basira
from basira.core.types import Category

import synth_image as S


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    return str(tmp_path_factory.mktemp("img"))


@pytest.mark.parametrize("maker", [S.make_authentic, S.make_resaved])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clean_images_are_not_flagged(out, maker, seed):
    path, _ = maker(out, seed)
    r = basira.analyze(path)
    assert r.verdict_level <= 1, (r.verdict, r.scores)
    assert r.scores["manipulation"] < 0.45
    for e in r.evidences:
        assert e.error is None, (e.detector, e.error)


@pytest.mark.parametrize("seed", [0, 1])
def test_splice_is_localised_by_double_quantisation(out, seed):
    path, mask = S.make_splice(out, seed)
    r = basira.analyze(path)
    dq = r.get("image.jpeg_dq")
    assert dq.score > 0.5 and r.verdict_level >= 2
    assert S.hit_ratio(dq.heatmap, mask) > 3
    assert S.hit_ratio(r.fused_heatmap, mask) > 3
    assert dq.regions


@pytest.mark.parametrize("seed", [0, 2])
def test_copy_move_is_found_with_accurate_mask(out, seed):
    path, mask = S.make_copymove(out, seed)
    r = basira.analyze(path)
    cm = r.get("image.copymove")
    assert cm.score > 0.5
    assert S.iou(cm.heatmap, mask, 0.3) > 0.6
    assert {reg.label for reg in cm.regions} == {"clone source", "clone target"}


@pytest.mark.parametrize("seed", [2, 4])
def test_smooth_inpainting_is_localised_by_noise(out, seed):
    path, mask = S.make_inpaint(out, seed)
    r = basira.analyze(path)
    nz = r.get("image.noise")
    assert nz.score > 0.5 and nz.details["direction"] == "smoother"
    assert S.hit_ratio(nz.heatmap, mask) > 5


def test_shifted_jpeg_grid_region_is_found(out):
    path, mask = S.make_grid_shift(out, 1)
    r = basira.analyze(path)
    assert r.get("image.jpeg_dq").score > 0.5
    assert S.hit_ratio(r.fused_heatmap, mask) > 3


def test_generator_metadata_is_decisive(out):
    r = basira.analyze(S.make_sd_png(out))
    ev = r.get("image.generator_metadata")
    assert ev.decisive and ev.category == Category.AI_GENERATION
    assert r.verdict_level == 4 and "AI-generated" in r.verdict
    assert r.scores["ai_generation"] > 0.9


def test_periodic_generator_fingerprint_in_lossless_image(out):
    path, _ = S.make_generated(out, 0)
    r = basira.analyze(path)
    assert r.get("image.spectral").score > 0.5
    clean, _ = S.make_authentic(out, 5)
    assert basira.analyze(clean).get("image.spectral").score < 0.5


def test_exif_thumbnail_mismatch(out):
    """An edited JPEG whose EXIF thumbnail still shows the original content."""
    import io
    import os
    import struct

    from PIL import Image

    orig = S.camera_photo(30)
    edited = orig.copy()
    edited[200:500, 300:700] = S.camera_photo(31)[200:500, 300:700][:, ::-1]
    tb = io.BytesIO()
    Image.fromarray(orig).resize((160, 120)).save(tb, "JPEG", quality=80)
    thumb = tb.getvalue()
    # Minimal EXIF: IFD0 (empty) -> IFD1 with JPEGInterchangeFormat / Length.
    ifd1_off = 8 + 2 + 4
    data_off = ifd1_off + 2 + 2 * 12 + 4
    tiff = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", 0) + struct.pack("<I", ifd1_off)
    tiff += struct.pack("<H", 2)
    tiff += struct.pack("<HHII", 0x0201, 4, 1, data_off) + struct.pack("<HHII", 0x0202, 4, 1, len(thumb))
    tiff += struct.pack("<I", 0) + thumb
    p = os.path.join(out, "thumb_mismatch.jpg")
    Image.fromarray(edited).save(p, "JPEG", quality=90, exif=b"Exif\x00\x00" + tiff)
    r = basira.analyze(p)
    th = r.get("image.thumbnail")
    assert th is not None and th.score > 0.5
    mask = np.zeros(orig.shape[:2], bool)
    mask[200:500, 300:700] = True
    assert S.hit_ratio(th.heatmap, mask) > 3
    # Same thumbnail on the unedited image must not be flagged.
    p2 = os.path.join(out, "thumb_match.jpg")
    Image.fromarray(orig).save(p2, "JPEG", quality=90, exif=b"Exif\x00\x00" + tiff)
    ok = basira.analyze(p2).get("image.thumbnail")
    assert ok is not None and ok.score < 0.5, ok.details


def test_report_serialises_and_renders(out):
    import json

    from basira.core.report_html import render_html

    path, _ = S.make_splice(out, 0)
    r = basira.analyze(path)
    json.dumps(r.to_dict())
    html = render_html(r)
    assert "<html" in html and "Evidence" in html and len(html) > 20_000
