"""Copy-move (clone-stamp) detection with dense verification.

1. SIFT keypoints matched against the image itself with the generalised 2NN
   test (Amerini et al., 2011) — robust to rotation and scaling of the clone.
2. Matches are grouped into geometric transforms with iterative RANSAC; each
   transform must be supported by enough spatially spread inliers.
3. Each transform is *verified densely*: the image is warped by the transform
   and the local zero-mean normalised correlation between the image and its
   warped self delineates the exact source and target regions.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from ..core.types import Category, Evidence, Region
from ..core.utils import calibrate
from .load import ImageData

MAX_SIDE = 1600
MIN_SHIFT = 24          # px: ignore matches closer than this (self-similarity)
G2NN_RATIO = 0.5
MIN_INLIERS = 8
INLIER_THRESHOLD = 10   # inliers at which the score reaches 0.5
INLIER_SOFTNESS = 1.5
CORR_THR = 0.92
# Correlation of the denoising residual between the two matched areas. Selected on 348
# real clones (CoMoFoD, plus this project's clones on real photos, fresh and re-compressed)
# against 92 look-alike matches in untouched images (Columbia, CoMoFoD originals): this rule
# keeps 231 clones and 3 look-alikes. Those thresholds were tuned on that same data.
FINE_THRESHOLD = 0.62
FINE_THRESHOLD_MANY = 0.52
MANY_INLIERS = 50
FINE_SOFTNESS = 0.02
IDENTITY_RADIUS = 10
MIN_VERIFIED_AREA = 0.0005


def _local_corr(a: np.ndarray, b: np.ndarray, valid: np.ndarray, k: int = 9) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    box = (k, k)
    ma, mb = cv2.blur(a, box), cv2.blur(b, box)
    va = cv2.blur(a * a, box) - ma * ma
    vb = cv2.blur(b * b, box) - mb * mb
    cov = cv2.blur(a * b, box) - ma * mb
    c = cov / np.sqrt(np.maximum(va * vb, 1e-6))
    textured = (va > 4.0) & (vb > 4.0)   # flat areas correlate trivially
    return np.where(valid & textured, c, 0.0)


def analyze_copymove(d: ImageData) -> Evidence:
    det, title = "image.copymove", "Copy-move (cloning)"
    rgb = d.rgb
    H, W = rgb.shape[:2]
    s = min(1.0, MAX_SIDE / max(H, W))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if s < 1:
        gray = cv2.resize(gray, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    h, w = gray.shape
    if min(h, w) < 64:
        return Evidence.not_applicable(det, title, Category.MANIPULATION, "image too small")
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02, edgeThreshold=12)
    kps, des = sift.detectAndCompute(gray, None)
    if des is None or len(kps) < 20:
        return Evidence(detector=det, title=title, category=Category.MANIPULATION, score=0.0,
                        confidence=0.2, summary="Too few keypoints to test for cloning "
                        "(flat image). Clones of flat areas cannot be detected this way.",
                        details={"keypoints": 0 if des is None else len(kps)})
    pts = np.float32([k.pt for k in kps])
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des, des, k=min(10, len(kps)))
    pairs = []
    for i, ms in enumerate(knn):
        ms = [m for m in ms if m.trainIdx != i]
        for a, b in zip(ms[:-1], ms[1:]):
            if a.distance >= G2NN_RATIO * max(b.distance, 1e-6):
                break
            j = a.trainIdx
            if np.linalg.norm(pts[i] - pts[j]) > MIN_SHIFT and i < j:
                pairs.append((i, j))
    pairs = list(dict.fromkeys(pairs))
    info = {"keypoints": len(kps), "candidate_pairs": len(pairs)}
    if len(pairs) < MIN_INLIERS:
        return Evidence(detector=det, title=title, category=Category.MANIPULATION,
                        score=calibrate(len(pairs), INLIER_THRESHOLD, INLIER_SOFTNESS) * 0.6,
                        confidence=0.7, summary=f"No cloned regions found ({len(pairs)} "
                        f"self-matching keypoint pairs).", details=info)

    src = np.float32([pts[i] for i, _ in pairs])
    dst = np.float32([pts[j] for _, j in pairs])
    # Orient every pair consistently (left-most point is the "source").
    flip = (src[:, 0] > dst[:, 0]) | ((src[:, 0] == dst[:, 0]) & (src[:, 1] > dst[:, 1]))
    src[flip], dst[flip] = dst[flip].copy(), src[flip].copy()

    transforms = []
    remaining = np.arange(len(src))
    for _ in range(4):
        if len(remaining) < MIN_INLIERS:
            break
        M, inl = cv2.estimateAffine2D(src[remaining], dst[remaining], method=cv2.RANSAC,
                                      ransacReprojThreshold=4.0, maxIters=4000, confidence=0.995)
        if M is None:
            break
        inl = inl.ravel().astype(bool)
        n_in = int(inl.sum())
        if n_in < MIN_INLIERS:
            break
        lin = M[:, :2]
        sv = np.linalg.svd(lin, compute_uv=False)
        spread = float(np.sqrt(np.prod(src[remaining][inl].std(axis=0) + 1)))
        if 0.33 < sv.min() and sv.max() < 3.0 and spread > 6 and \
                np.linalg.norm(M[:, 2] + (lin - np.eye(2)) @ src[remaining][inl].mean(0)) > MIN_SHIFT:
            transforms.append((M, src[remaining][inl], dst[remaining][inl]))
        remaining = remaining[~inl]

    if not transforms:
        return Evidence(detector=det, title=title, category=Category.MANIPULATION, score=0.1,
                        confidence=0.7, summary=f"Self-matching keypoints ({len(pairs)}) do not "
                        f"form a consistent geometric clone.", details=info)

    mask_src = np.zeros((h, w), np.uint8)
    mask_dst = np.zeros((h, w), np.uint8)
    best_in = 0
    descs = []
    identities = []
    accepted = []
    g = gray.astype(np.float32)
    for M, s_in, d_in in transforms:
        warped = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        valid = warped >= 0
        corr = _local_corr(g, warped, valid)       # high where I(x) ~ I(M^-1 x) -> target
        tgt = (corr > CORR_THR).astype(np.uint8)
        tgt = cv2.morphologyEx(tgt, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        hull_d = cv2.convexHull(d_in.astype(np.int32))
        near = np.zeros_like(tgt)
        cv2.fillConvexPoly(near, hull_d, 1)
        near = cv2.dilate(near, np.ones((41, 41), np.uint8))
        tgt &= near
        # Is the match a *digital copy*, or two real objects that look alike (keyboard
        # keys, identical windows, books of one series)? Keypoints match in both cases.
        # What only a copy shares is the fine detail: sensor noise and micro-texture. So
        # compare the denoising residual of the two areas, over small discs around the
        # matched keypoints (not their convex hull, which would include un-cloned
        # background between the parts of a non-convex object).
        discs = np.zeros_like(tgt)
        for x, y in np.round(d_in).astype(int):
            cv2.circle(discs, (int(x), int(y)), IDENTITY_RADIUS, 1, -1)
        dz = (discs > 0) & valid
        fine = 0.0
        if dz.sum() > 50:
            na = g - cv2.medianBlur(np.clip(g, 0, 255).astype(np.uint8), 3).astype(np.float32)
            nb = warped - cv2.medianBlur(np.clip(warped, 0, 255).astype(np.uint8), 3).astype(np.float32)
            a_, b_ = na[dz] - na[dz].mean(), nb[dz] - nb[dz].mean()
            fine = float((a_ * b_).sum() / (np.sqrt((a_ * a_).sum() * (b_ * b_).sum()) + 1e-6))
        verified = float(tgt.sum()) / (h * w)
        # Each transform is judged on its own: both conditions must hold - enough
        # geometrically consistent matches, and matching fine detail. With many matches a
        # lower detail correlation is accepted (re-compression rewrites the fine detail of
        # a real clone); with few it is not. A look-alike pair elsewhere in the image, even
        # one with more matches, must not hide a genuine clone.
        n_in = len(s_in)
        fine_thr = FINE_THRESHOLD_MANY if n_in >= MANY_INLIERS else FINE_THRESHOLD
        t_score = min(calibrate(n_in, INLIER_THRESHOLD, INLIER_SOFTNESS),
                      calibrate(fine, fine_thr, FINE_SOFTNESS))
        identities.append((t_score, n_in, fine, verified))
        shift = M[:, 2]
        descs.append({"inliers": n_in, "fine_detail_corr": round(fine, 3),
                      "score": round(t_score, 3),
                      "translation": [round(float(shift[0] / s), 1), round(float(shift[1] / s), 1)],
                      "linear": np.round(M[:, :2], 3).tolist()})
        if t_score >= 0.5:                       # only accepted clones are drawn
            Minv = cv2.invertAffineTransform(M)
            mask_dst |= tgt
            mask_src |= cv2.warpAffine(tgt, Minv, (w, h), flags=cv2.INTER_NEAREST)
            accepted.append((M, s_in, d_in))
    area = float((mask_src | mask_dst).mean())
    score, best_in, best_identity, best_verified = max(identities, key=lambda t: (t[0], t[1]))
    heat_small = np.maximum(mask_dst.astype(np.float32), 0.6 * mask_src.astype(np.float32))
    heat = cv2.resize(heat_small, (W, H), interpolation=cv2.INTER_LINEAR)
    heat = cv2.GaussianBlur(heat, (0, 0), 2)
    regions: List[Region] = []
    for m, label in ((mask_dst, "clone target"), (mask_src, "clone source")):
        n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        for k in range(1, n):
            x, y, ww, hh, a = st[k]
            if a > 0.0003 * h * w:
                regions.append(Region(int(x / s), int(y / s), int(ww / s), int(hh / s), score, label))
    vis = rgb.copy()
    if s < 1:
        vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_AREA)
    for M, s_in, d_in in (accepted or transforms)[:3]:
        for a, b in zip(s_in[:150], d_in[:150]):
            cv2.line(vis, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 120), 1, cv2.LINE_AA)
            cv2.circle(vis, tuple(map(int, a)), 2, (255, 220, 0), -1)
            cv2.circle(vis, tuple(map(int, b)), 2, (255, 60, 60), -1)
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION, score=score, confidence=0.8,
        summary=(f"Found {len(accepted)} cloned region pair(s); the strongest is supported by "
                 f"{best_in} geometrically consistent keypoint matches whose fine detail "
                 f"(noise, micro-texture) also matches (correlation {best_identity:.2f}); the "
                 f"pair covers {area * 100:.2f}% of the image." if score >= 0.5 else
                 (f"{best_in} keypoint matches line up, but the fine detail of the two areas "
                  f"does not match (correlation {best_identity:.2f}): this looks like genuinely "
                  f"repeated objects (keys, windows, tiles), not a digital clone."
                  if best_in >= INLIER_THRESHOLD else
                  f"Weak self-similarity only ({best_in} consistent matches); no convincing "
                  f"clone.")),
        details={**info, "transforms": descs, "area_fraction": area,
                 "fine_detail_corr": best_identity, "verified_area": best_verified,
                 "threshold_inliers": INLIER_THRESHOLD},
        heatmap=heat, regions=regions, visuals={"copymove_matches": vis})
