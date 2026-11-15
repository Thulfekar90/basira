"""image.jpeg_trace: an earlier JPEG save, still visible in losslessly stored pixels,
localises content pasted in afterwards — and stays silent without such a history."""
from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from PIL import Image

import basira
from basira.image.jpeg_trace import analyze_jpeg_trace
from basira.image.load import from_array, load_image

import synth_image as S


def _png(rgb, out, name):
    p = os.path.join(out, name)
    Image.fromarray(rgb).save(p)
    return p


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    return str(tmp_path_factory.mktemp("trace"))


@pytest.mark.parametrize("seed", [0, 1])
def test_decoded_jpeg_stored_losslessly_is_clean(out, seed):
    img = S.jpeg(S.camera_photo(seed), [75, 85][seed])
    e = analyze_jpeg_trace(load_image(_png(img, out, f"clean_{seed}.png")))
    assert e.details["n_combs"] >= 4, e.details          # the earlier save is found ...
    assert e.score < 0.3, e.details                       # ... and it is consistent


def test_camera_original_without_jpeg_history_is_not_applicable(out):
    e = analyze_jpeg_trace(from_array(S.camera_photo(3), "camera"))
    assert e.score == 0.0 and e.confidence <= 0.1
    assert "no trace" in e.summary


@pytest.mark.parametrize("seed", [0, 1])
def test_scaled_paste_into_decoded_jpeg_is_localised(out, seed):
    rng = np.random.RandomState(40 + seed)
    host = S.jpeg(S.camera_photo(seed + 10), 80).copy()
    donor = S.jpeg(S.camera_photo(seed + 60), 90)
    h, w = host.shape[:2]
    s = 220
    y, x = int(rng.randint(40, h - s - 40)), int(rng.randint(40, w - s - 40))
    patch = cv2.resize(donor[100:100 + int(s / 1.3), 100:100 + int(s / 1.3)], (s, s),
                       interpolation=cv2.INTER_LINEAR)
    host[y:y + s, x:x + s] = patch
    mask = np.zeros((h, w), bool)
    mask[y:y + s, x:x + s] = True
    path = _png(host, out, f"splice_{seed}.png")
    r = basira.analyze(path)
    e = r.get("image.jpeg_trace")
    assert e.score > 0.5 and r.verdict_level >= 2, e.details
    assert S.hit_ratio(e.heatmap, mask) > 3
    assert e.regions


def test_saturated_colours_do_not_look_pasted(out):
    """Every false alarm on CASIA v2 originals came from colours clipped by the decoder."""
    img = S.camera_photo(5).astype(np.float32)
    img[200:420, 300:600] *= np.array([2.2, 0.5, 0.4], np.float32)   # vivid red object
    img = S.jpeg(np.clip(img, 0, 255).astype(np.uint8), 85)
    e = analyze_jpeg_trace(load_image(_png(img, out, "red.png")))
    assert e.score < 0.3, e.details


def test_jpeg_files_are_left_to_the_double_quantisation_map(out):
    p = os.path.join(out, "plain.jpg")
    S.save_jpeg(S.camera_photo(6), p, 85)
    e = analyze_jpeg_trace(load_image(p))
    assert e.score == 0.0 and "JPEG file" in e.summary
