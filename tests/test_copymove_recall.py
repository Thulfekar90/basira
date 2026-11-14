"""Copy-move recall: clones that keypoints alone miss or mis-judge, and the look-alikes
that must stay quiet. Everything is synthetic and seed-driven; the numbers on real data
are measured by benchmarks/public_datasets.py."""
from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from PIL import Image

import synth_image as S
from basira.image import copymove as CM
from basira.image.copymove import analyze_copymove
from basira.image.load import from_array, load_image

FLAG = 0.78      # a copy-move score from which the fused verdict is "Suspicious" on its own


def _png(rgb: np.ndarray, tmp_path, name: str):
    p = os.path.join(str(tmp_path), name)
    Image.fromarray(rgb).save(p)
    return load_image(p)


def _best(e, method):
    ts = [t for t in e.details.get("transforms", []) if t["method"] == method]
    return max(ts, key=lambda t: t["score"]) if ts else None


# ----------------------------------------------------------------------------- #
# Matching
# ----------------------------------------------------------------------------- #
def test_g2nn_by_matrix_products_equals_brute_force_knn():
    rs = np.random.RandomState(0)
    des = rs.rand(600, 128).astype(np.float32) * 60
    des[300:360] = des[:60] + rs.randn(60, 128).astype(np.float32) * 0.5     # 60 near-duplicates
    # 20 of them a second time: with two equally near neighbours the ratio test stops at
    # once and neither is matched (in both implementations).
    des[400:420] = des[:20] + rs.randn(20, 128).astype(np.float32) * 0.5
    pts = rs.rand(600, 2).astype(np.float32) * 1000
    want = set()
    for i, ms in enumerate(cv2.BFMatcher(cv2.NORM_L2).knnMatch(des, des, k=10)):
        ms = [m for m in ms if m.trainIdx != i]
        for a, b in zip(ms[:-1], ms[1:]):
            if a.distance >= CM.G2NN_RATIO * max(b.distance, 1e-6):
                break
            if np.linalg.norm(pts[i] - pts[a.trainIdx]) > CM.MIN_SHIFT:
                want.add((min(i, a.trainIdx), max(i, a.trainIdx)))
    got = set(map(tuple, CM._g2nn_pairs(des, pts, chunk=128).tolist()))
    assert len(want) == 40 and got == want


# ----------------------------------------------------------------------------- #
# Small images, resampled clones
# ----------------------------------------------------------------------------- #
def _detailed_photo(seed: int, h: int = 512, w: int = 512) -> np.ndarray:
    """A small 'photo' with busy detail (pebbles), which the plain scene lacks at this size."""
    rs = np.random.RandomState(seed)
    sc = S.scene(h, w, seed)
    for _ in range(900):
        c = tuple(int(v) for v in rs.randint(30, 230, 3))
        cv2.circle(sc, (int(rs.randint(w)), int(rs.randint(h))), int(rs.randint(2, 6)), c, -1, cv2.LINE_AA)
    return S.capture(cv2.GaussianBlur(sc, (0, 0), 0.6), noise=2.5, seed=seed + 100)


@pytest.mark.parametrize("seed", [2, 3])
def test_rotated_scaled_clone_in_a_small_lossless_image(tmp_path, seed):
    img = _detailed_photo(seed).copy()
    sy, sx, n = 60 + 40 * seed, 50 + 30 * seed, 110
    R = cv2.getRotationMatrix2D((n / 2, n / 2), 17 + 5 * seed, 1.15)
    rot = cv2.warpAffine(img[sy:sy + n, sx:sx + n], R, (n, n), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REFLECT)
    disc = np.zeros((n, n), np.uint8)
    cv2.circle(disc, (n // 2, n // 2), n // 2 - 4, 1, -1)
    ty, tx = 330, 300
    img[ty:ty + n, tx:tx + n][disc > 0] = rot[disc > 0]
    e = analyze_copymove(_png(img, tmp_path, "rot.png"))
    assert e.score > FLAG, e.details["transforms"]
    assert e.details["enlarged_for_keypoints"]
    t = _best(e, "keypoints")
    assert t["quiet_noise_corr"] is None or t["quiet_noise_corr"] > 0.6     # shared noise
    assert t["slide_corr"] < 0.3                                            # sharp alignment peak
    assert e.heatmap[ty + 25:ty + n - 25, tx + 25:tx + n - 25].mean() > 0.5
    clean = analyze_copymove(_png(_detailed_photo(seed), tmp_path, "clean.png"))
    assert clean.score < 0.5


# ----------------------------------------------------------------------------- #
# Clones without keypoints: pixel-identical blocks
# ----------------------------------------------------------------------------- #
def _smooth_photo(seed: int = 7) -> np.ndarray:
    """A noisy gradient: a sky or a wall. No keypoints at all."""
    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:512, 0:640].astype(np.float32)
    base = np.stack([150 + 40 * yy / 512 + 10 * xx / 640, 160 + 30 * yy / 512, 190 - 20 * yy / 512], -1)
    return np.clip(base + rs.randn(512, 640, 3) * 2.5, 0, 255).astype(np.uint8)


def test_clone_of_a_smooth_area_is_found_by_identical_blocks(tmp_path):
    clean = _smooth_photo()
    img = clean.copy()
    img[300:380, 400:500] = img[100:180, 120:220]
    e = analyze_copymove(_png(img, tmp_path, "smooth.png"))
    t = _best(e, "identical blocks")
    assert e.score > FLAG and t is not None, e.summary
    assert t["translation"] == [280.0, 200.0] and t["identical_blocks"] > 1000
    assert e.heatmap[310:370, 410:490].mean() > 0.9          # target
    assert e.heatmap[110:170, 130:210].mean() > 0.5          # source
    assert e.heatmap[200:280, 300:400].mean() < 0.01
    assert {r.label for r in e.regions} == {"clone source", "clone target"}
    assert "identical" in e.summary
    # The untouched picture has no identical blocks at all.
    assert analyze_copymove(_png(clean, tmp_path, "smooth_clean.png")).score < 0.2


def test_identical_blocks_are_claimed_only_for_stored_pixels(tmp_path):
    """After lossy coding, and for decoded video frames, nothing is pixel-identical by
    right: the test must not run there (a smooth JPEG repeats whole blocks by itself)."""
    img = _smooth_photo()
    img[300:380, 400:500] = img[100:180, 120:220]
    p = os.path.join(str(tmp_path), "smooth.jpg")
    Image.fromarray(img).save(p, quality=92)
    for d in (load_image(p), from_array(img)):
        e = analyze_copymove(d)
        assert _best(e, "identical blocks") is None and e.score < 0.5


def test_smooth_jpeg_history_is_not_an_identical_block_clone(tmp_path):
    """A PNG made from a JPEG: flat and smoothly shaded 8x8 blocks repeat exactly, but
    they carry no noise and must not count."""
    rs = np.random.RandomState(1)
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
    sky = np.stack([120 + 0 * xx + 60 * yy / 480] * 3, -1)
    sky[300:] += cv2.GaussianBlur(rs.randn(180, 640, 3).astype(np.float32) * 25, (0, 0), 3)
    rgb = S.jpeg(np.clip(sky, 0, 255).astype(np.uint8), 70)
    e = analyze_copymove(_png(rgb, tmp_path, "from_jpeg.png"))
    assert _best(e, "identical blocks") is None and e.score < 0.5


def test_drawn_content_is_not_searched_for_identical_blocks():
    """Screenshots and diagrams repeat glyphs and icons pixel for pixel; that is not
    evidence of anything."""
    img = np.full((480, 640, 3), 250, np.uint8)
    for row in range(12):
        cv2.putText(img, "Invoice item 0.00 EUR  qty 1", (20, 40 + 36 * row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    assert CM._exact_duplicates(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)) == []


# ----------------------------------------------------------------------------- #
# Printed duplicates: two real objects of one design
# ----------------------------------------------------------------------------- #
def _design(n: int, seed: int) -> np.ndarray:
    rs = np.random.RandomState(seed)
    d = np.full((n, n, 3), 225, np.uint8)
    cv2.circle(d, (n // 2, n // 2), n // 2 - 4, (200, 170, 60), 3, cv2.LINE_AA)
    for k, word in enumerate(("PAULA", "Brau 1634", "Oktober")):
        cv2.putText(d, word, (14, 45 + 34 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9 - 0.2 * k,
                    (60, 30, 140 - 40 * k), 2, cv2.LINE_AA)
    for _ in range(25):
        cv2.circle(d, tuple(int(v) for v in rs.randint(20, n - 20, 2)), int(rs.randint(2, 5)),
                   tuple(int(v) for v in rs.randint(40, 200, 3)), -1, cv2.LINE_AA)
    return d.astype(np.float32)


def _two_prints(seed: int) -> np.ndarray:
    """One printed design photographed twice in one frame (two beer coasters): slightly
    different pose, the same print, independent sensor noise."""
    rs = np.random.RandomState(seed)
    canvas = np.full((512, 512, 3), 205, np.float32) + \
        cv2.GaussianBlur(rs.randn(512, 512, 3).astype(np.float32) * 20, (0, 0), 25)
    d = _design(150, seed)
    for cx, cy, ang, sc in ((120, 130, 0.0, 1.0), (350, 330, 1.5, 0.985)):
        M = cv2.getRotationMatrix2D((75, 75), ang, sc)
        M[:, 2] += (cx - 75 + 0.37 * ang, cy - 75 + 0.21)
        warped = cv2.warpAffine(d, M, (512, 512), flags=cv2.INTER_CUBIC, borderValue=(-1, -1, -1))
        canvas[warped[..., 0] >= 0] = warped[warped[..., 0] >= 0]
    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.7)
    return np.clip(canvas + rs.randn(512, 512, 3) * 2.0, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("seed", [0, 1])
def test_printed_duplicates_are_not_a_clone(tmp_path, seed):
    e = analyze_copymove(_png(_two_prints(seed), tmp_path, "prints.png"))
    t = _best(e, "keypoints")
    assert t["inliers"] > 50 and t["fine_detail_corr"] > 0.8      # they do look alike...
    assert t["quiet_noise_corr"] < 0.4                            # ...but do not share noise
    assert e.score < 0.5, e.details["transforms"]
    assert "noise" in e.summary and not e.regions


@pytest.mark.parametrize("seed", [0, 1])
def test_real_clone_next_to_printed_duplicates_is_still_found(tmp_path, seed):
    img = _two_prints(seed)
    # A rotated digital copy of the first print: it does share the noise.
    R = cv2.getRotationMatrix2D((120, 130), 12, 1.0)
    R[:, 2] += (255, -45)
    moved = cv2.warpAffine(img, R, (512, 512), flags=cv2.INTER_CUBIC)
    disc = np.zeros((512, 512), np.uint8)
    cv2.circle(disc, (375, 85), 70, 1, -1)
    img[disc > 0] = moved[disc > 0]
    e = analyze_copymove(_png(img, tmp_path, "prints_clone.png"))
    assert e.score > FLAG, e.details["transforms"]
    hits = [t for t in e.details["transforms"] if t["score"] > 0.5]
    assert len(hits) == 1 and hits[0]["quiet_noise_corr"] > 0.6
    assert e.heatmap[45:125, 335:415].mean() > 0.5               # the digital copy
    assert e.heatmap[290:370, 310:390].mean() < 0.1              # the second real print


def _row_of_prints(seed: int, n: int) -> np.ndarray:
    """n copies of one printed design at an equal step (a row of windows or tiles), each with
    its own sensor noise."""
    rs = np.random.RandomState(seed)
    canvas = np.full((512, 512, 3), 205, np.float32) + \
        cv2.GaussianBlur(rs.randn(512, 512, 3).astype(np.float32) * 20, (0, 0), 25)
    d = _design(120, seed)
    for k in range(n):
        x, y = 30 + 150 * k, 40 + 115 * k
        canvas[y:y + 120, x:x + 120] = d
    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.7)
    return np.clip(canvas + rs.randn(512, 512, 3) * 2.0, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("seed", [0, 1])
def test_a_regular_row_of_look_alikes_in_a_jpeg_is_not_a_clone(tmp_path, seed):
    # After JPEG coding the flat-pixel noise cannot be compared, but a displacement that
    # recurs at twice its length is a lattice, not a copy.
    p = os.path.join(str(tmp_path), "row.jpg")
    Image.fromarray(_row_of_prints(seed, 3)).save(p, quality=90)
    e = analyze_copymove(load_image(p))
    ts = e.details["transforms"]
    assert len(ts) >= 2 and all(t["lattice_error"] < 0.02 for t in ts), ts
    assert e.score < 0.5 and "twice its length" in e.summary and not e.regions


# ----------------------------------------------------------------------------- #
# Contract
# ----------------------------------------------------------------------------- #
def test_evidence_keeps_its_documented_fields(tmp_path):
    img = _smooth_photo()
    img[300:380, 400:500] = img[100:180, 120:220]
    for e in (analyze_copymove(_png(img, tmp_path, "a.png")),
              analyze_copymove(_png(_two_prints(0), tmp_path, "b.png"))):
        assert {"transforms", "area_fraction", "fine_detail_corr", "verified_area"} <= set(e.details)
        for t in e.details["transforms"]:
            assert {"method", "inliers", "fine_detail_corr", "score", "translation", "linear"} <= set(t)
        assert e.to_dict()                                        # JSON-ready


def test_periodic_generator_texture_is_not_an_identical_block_clone(tmp_path):
    """A generator's 8-px upsampling imprint over smooth shading repeats exactly along one
    direction. The first identical-block search reported every such image as a clone."""
    import synth_image as S
    from basira.image.copymove import analyze_copymove
    from basira.image.load import load_image

    for seed in range(3):
        p, _ = S.make_generated(str(tmp_path), seed)
        assert analyze_copymove(load_image(p)).score < 0.5, seed
