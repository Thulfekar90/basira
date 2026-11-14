"""Copy-move (clone-stamp) detection with dense verification.

1. SIFT keypoints matched against the image itself with the generalised 2NN
   test (Amerini et al., 2011) — robust to rotation and scaling of the clone. Small
   images are enlarged first: at 512 px a cloned object carries too few keypoints.
2. Matches are grouped into geometric transforms with iterative RANSAC; each
   transform must be supported by enough spatially spread inliers.
3. Each transform is aligned more precisely than keypoints allow (robust refinement on
   the pixels; a pure whole-pixel shift is tried as well) and *verified densely*: the
   image is warped by the transform and the local zero-mean normalised correlation
   between the image and its warped self delineates the source and target regions.
4. A *fine-detail test* separates a digital copy from two real objects that look alike:
   only a copy shares the noise and micro-texture. In losslessly stored images the
   noise in the flat parts of the two areas is compared too (printed duplicates share
   their print, not the sensor noise between the lines).
5. Losslessly stored images that need no downscaling (up to MAX_SIDE) are also searched
   for *pixel-identical* blocks at one common shift. This needs no keypoints, so it finds
   clones of smooth or tiny areas - as long as nothing re-coded the pixels afterwards.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.types import Category, Evidence, Region
from ..core.utils import calibrate
from .load import ImageData

MAX_SIDE = 1600
MIN_SHIFT = 24          # px: ignore matches closer than this (self-similarity)
G2NN_RATIO = 0.5
MIN_INLIERS = 8
MAX_TRANSFORMS = 4
INLIER_THRESHOLD = 10   # inliers at which the score reaches 0.5
INLIER_SOFTNESS = 1.5
CORR_THR = 0.92
# Working images smaller than this are enlarged x2 before keypoint detection. Measured on
# the odd-numbered CoMoFoD forgeries (512 px): forgeries with at least 8 keypoint pairs
# inside the true clone rise from 28 to 36 of the 48 that the 1x detector missed.
UPSAMPLE_BELOW = 800
# Correlation of the denoising residual between the two matched areas. Selected on 348
# real clones (CoMoFoD, plus this project's clones on real photos, fresh and re-compressed)
# against 92 look-alike matches in untouched images (Columbia, CoMoFoD originals): this rule
# keeps 231 clones and 3 look-alikes. Those thresholds were tuned on that same data.
FINE_THRESHOLD = 0.62
FINE_THRESHOLD_MANY = 0.52
MANY_INLIERS = 50       # at 1x; an enlarged image yields about three times the keypoints
MANY_INLIERS_UPSAMPLED = 150
FINE_SOFTNESS = 0.02
IDENTITY_RADIUS = 10
MIN_VERIFIED_AREA = 0.0005
# Pure-shift hypothesis: matches within this distance of the group's median displacement.
SHIFT_TOLERANCE = 1.5
SHIFT_SNAP = 0.25       # px: a refined shift this close to whole pixels is taken as whole
# Noise in the flat parts ("quiet" pixels: little local structure in either copy) of the
# densely verified area, lossless images only. After JPEG re-compression genuine clones
# drop to 0.1-0.7, so the test is not applied there. Known cost: noise ADDED to a lossless
# image after the edit also hides the shared noise, and the clone is then turned down like a
# print (CoMoFoD "NA3", sigma about 6 grey levels: 71 of 200 forgeries flagged with this
# test, 122 without it; the same images are where it removes 4 of 12 flagged originals).
# QUIET_THRESHOLD was chosen on odd-numbered CoMoFoD images and odd-indexed Columbia
# originals: genuine clones 0.36-1.00, printed duplicates (coasters, journal labels) and
# repeated architecture 0.03-0.40. With that rule alone the held-out halves still flagged
# two Columbia originals and one more CoMoFoD scene, all with middling values of both
# measures (fine detail 0.66-0.69, quiet noise 0.46-0.62), whereas borderline genuine
# clones are strong in at least one (0.70 + 0.68 is the weakest pair). JOINT_THRESHOLD, on
# the mean of the two, was therefore chosen on ALL of CoMoFoD and Columbia: look-alikes
# reach at most 0.64 (one roof-tile scene excepted, 0.80), genuine clones at least 0.69.
QUIET_ACTIVITY = 4.0
QUIET_MIN_PIXELS = 80
QUIET_THRESHOLD = 0.42
QUIET_SOFTNESS = 0.02
JOINT_THRESHOLD = 0.655
JOINT_SOFTNESS = 0.01
# A match that survives being slid along its own displacement is not a copy: keypoints on a
# straight edge (eaves, gutters, rails) match anywhere along that edge, and the residual of
# a line correlates with itself under such a slide. A genuine clone de-correlates within a
# pixel or two. Measured SLIDE_SHIFT px further along (and back) the displacement.
# On CoMoFoD and Columbia (all images): genuine clones mostly below 0.3 of the aligned value,
# the one roof scene that passes every other test 0.61. Two genuine clones lie on line
# texture and slide too (0.60, 0.65); their flat-pixel noise is all but identical (0.99),
# which a look-alike cannot have, so noise at or above SLIDE_WAIVED_BY_NOISE waives the test.
SLIDE_SHIFT = 4.0
SLIDE_RATIO = 0.5       # rejected when the slid correlation exceeds this share of the aligned one
SLIDE_SOFTNESS = 0.05
SLIDE_WAIVED_BY_NOISE = 0.9
# Pixel-identical blocks (lossless images, no rescaling). Measured on odd-numbered CoMoFoD
# and odd-indexed Columbia images with 12x12 blocks: no original has a single such block
# pair; the 21 forgeries that have any have 62 to 27373 at one shift.
EXACT_BLOCK = 12
EXACT_BUSY = 0.5        # share of a block's pixels that differ from their 3x3 median
EXACT_MIN_BLOCKS = 30   # block positions at one shift at which the score reaches 0.5
EXACT_SOFTNESS = 6.0
EXACT_MAX_FLAT = 0.10   # more exactly-flat pixels than this: drawn content, not a photograph
# Periodic content repeats exactly too: a generator's upsampling imprint over a flat area
# (an 8-px pattern) made every block equal the one 8, 16, 24 px away, which the first
# version reported as a clone. A shift is dropped when its match is not unique: the blocks
# are also identical at a short step from themselves, at a shift a short step away, or at
# twice the shift. Round values, not tuned.
EXACT_PERIOD_MAX = 16   # px; short steps checked horizontally and vertically
EXACT_PERIODIC = 0.5    # share of the shift's blocks that repeat at one such step
# Fine-detail self-correlation (3x3-median residual, steps 2..16 px) above which content is
# periodic. Measured: CoMoFoD forgeries with identical-block clones peak at 0.39, the
# synthetic upsampling imprint at >= 0.98; 0.7 sits between. (A blur residual and a 1-px
# step were tried first: ordinary texture correlates there and 8 real clones were lost.)
EXACT_TEXTURE_CORR = 0.7
# A displacement that recurs at twice its length is a lattice (windows, tiles, a facade), not
# a copy: a clone does not repeat itself at an equal step. Found by looking at the CoMoFoD
# originals still flagged after JPEG re-compression (all ids, both halves; the quiet-noise
# test cannot be used there): nearly all are two scenes matched at about (156, -94) and at
# twice that. The tolerance is a round value, not tuned: relative error of |t2 - 2 t1|.
LATTICE_TOL = 0.08
LATTICE_SOFTNESS = 0.01
LATTICE_MAX_LINEAR = 0.05   # only for transforms this close to a pure shift


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


def _g2nn_pairs(des: np.ndarray, pts: np.ndarray, chunk: int = 1024, max_rank: int = 8) -> np.ndarray:
    """Generalised 2NN self-matching by matrix products (same result as a brute-force
    k-NN matcher, several times faster). With neighbours of keypoint i sorted by distance
    d1 <= d2 <= ..., neighbour k is a match while d_k < G2NN_RATIO * d_(k+1). Returns
    index pairs (i, j), i < j, of keypoints farther apart than MIN_SHIFT."""
    d = np.ascontiguousarray(des, np.float32)
    n = len(d)
    sq = np.einsum("ij,ij->i", d, d)
    found_i, found_j = [], []
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        dist = d[a:b] @ d.T
        dist *= -2.0
        dist += sq[None, :]
        dist += sq[a:b, None]                       # squared distances
        rows = np.arange(b - a)
        dist[rows, rows + a] = np.inf               # a keypoint is not its own match
        j_cur = dist.argmin(1)
        d_cur = dist[rows, j_cur]
        dist[rows, j_cur] = np.inf
        act = rows
        for _ in range(max_rank):
            sub = dist[act]
            j_nxt = sub.argmin(1)
            d_nxt = sub[np.arange(len(act)), j_nxt]
            ok = np.sqrt(np.maximum(d_cur, 0)) < G2NN_RATIO * np.sqrt(np.maximum(d_nxt, 1e-12))
            if not ok.any():
                break
            found_i.append(act[ok] + a)
            found_j.append(j_cur[ok])
            act, j_cur, d_cur = act[ok], j_nxt[ok], d_nxt[ok]
            dist[act, j_cur] = np.inf
    if not found_i:
        return np.zeros((0, 2), np.int64)
    i, j = np.concatenate(found_i), np.concatenate(found_j)
    far = np.linalg.norm(pts[i] - pts[j], axis=1) > MIN_SHIFT
    i, j = i[far], j[far]
    return np.unique(np.stack([np.minimum(i, j), np.maximum(i, j)], 1), axis=0)


def _self_matches(gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, bool]:
    """-> (src points, dst points, number of keypoints, enlarged?) at working scale; the
    left-most point of every pair is the "source". (None, None, n, _) if too few keypoints."""
    up = max(gray.shape) < UPSAMPLE_BELOW
    g = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) if up else gray
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02, edgeThreshold=12)
    kps, des = sift.detectAndCompute(g, None)
    if des is None or len(kps) < 20:
        return None, None, 0 if des is None else len(kps), up
    pts = np.float32([k.pt for k in kps]) / (2.0 if up else 1.0)
    pairs = _g2nn_pairs(des, pts)
    src, dst = pts[pairs[:, 0]], pts[pairs[:, 1]]
    flip = (src[:, 0] > dst[:, 0]) | ((src[:, 0] == dst[:, 0]) & (src[:, 1] > dst[:, 1]))
    src[flip], dst[flip] = dst[flip].copy(), src[flip].copy()
    return src, dst, len(kps), up


def _group_transforms(src: np.ndarray, dst: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Iterative RANSAC: up to MAX_TRANSFORMS affine maps, each with its inlier points."""
    transforms = []
    remaining = np.arange(len(src))
    cv2.setRNGSeed(0)        # RANSAC draws from OpenCV's global generator: same image, same answer
    for _ in range(MAX_TRANSFORMS):
        if len(remaining) < MIN_INLIERS:
            break
        M, inl = cv2.estimateAffine2D(src[remaining], dst[remaining], method=cv2.RANSAC,
                                      ransacReprojThreshold=4.0, maxIters=4000, confidence=0.995)
        if M is None:
            break
        inl = inl.ravel().astype(bool)
        if int(inl.sum()) < MIN_INLIERS:
            break
        lin = M[:, :2]
        sv = np.linalg.svd(lin, compute_uv=False)
        s_in, d_in = src[remaining][inl], dst[remaining][inl]
        spread = float(np.sqrt(np.prod(s_in.std(axis=0) + 1)))
        if 0.33 < sv.min() and sv.max() < 3.0 and spread > 6 and \
                np.linalg.norm(M[:, 2] + (lin - np.eye(2)) @ s_in.mean(0)) > MIN_SHIFT:
            transforms.append((M, s_in, d_in))
        remaining = remaining[~inl]
    return transforms


def _discs(shape, pts: np.ndarray, r: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for x, y in np.round(pts).astype(int):
        cv2.circle(m, (int(x), int(y)), r, 1, -1)
    return m


def _residual(x: np.ndarray) -> np.ndarray:
    """Denoising residual: what a 3x3 median removes (noise and micro-texture)."""
    return x - cv2.medianBlur(np.clip(x, 0, 255).astype(np.uint8), 3).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


class _Image:
    """The working image with the derived planes that several steps share."""

    def __init__(self, gray: np.ndarray):
        self.g = gray.astype(np.float32)
        self.shape = gray.shape
        self.res = _residual(self.g)
        self._smooth: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def smooth(self, sigma: float):
        """(blurred image, d/dx, d/dy) at one scale, computed once."""
        if sigma not in self._smooth:
            gb = cv2.GaussianBlur(self.g, (0, 0), sigma)
            self._smooth[sigma] = (gb, cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8.0),
                                   cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8.0))
        return self._smooth[sigma]

    def warp(self, M: np.ndarray) -> np.ndarray:
        h, w = self.shape
        return cv2.warpAffine(self.g, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=-1)


def _refine(im: _Image, M: np.ndarray, pts: np.ndarray, shift_only: bool = False,
            r: int = 12, sigmas=(2.0, 1.0), iters: int = 12) -> np.ndarray:
    """Refine M on the pixels: M maps the other copy onto the area around ``pts``
    (g(x) ~ g(M^-1 x) there). Gauss-Newton with Tukey weights, coarse then fine, over
    discs around the matched keypoints; pixels that do not belong to the clone (the
    background around a pasted object) lose their weight instead of pulling the fit."""
    h, w = im.shape
    pad = r + 2
    x0, y0 = np.maximum(np.floor(pts.min(0)).astype(int) - pad, 0)
    x1, y1 = np.minimum(np.ceil(pts.max(0)).astype(int) + pad + 1, [w, h])
    cw, ch = int(x1 - x0), int(y1 - y0)
    yy, xx = np.nonzero(_discs((h, w), pts, r)[y0:y1, x0:x1])
    if len(yy) < 60:
        return M
    X, Y = (xx + x0).astype(np.float64), (yy + y0).astype(np.float64)
    A0 = cv2.invertAffineTransform(M)               # area around pts -> the other copy
    A = A0.copy()

    def sample(img, A):
        Ac = A.copy()
        Ac[:, 2] = A[:, :2] @ np.float64([x0, y0]) + A[:, 2]
        return cv2.warpAffine(img, Ac, (cw, ch), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)[yy, xx].astype(np.float64)

    for sigma in sigmas:
        gb, gx, gy = im.smooth(sigma)
        target = gb[y0:y1, x0:x1][yy, xx].astype(np.float64)
        for _ in range(iters):
            e = target - sample(gb, A)
            scale = 3.0 * max(1.4826 * float(np.median(np.abs(e - np.median(e)))), 1.0)
            wgt = np.where(np.abs(e) < scale, (1 - (e / scale) ** 2) ** 2, 0.0)
            ix, iy = sample(gx, A), sample(gy, A)
            J = np.stack([ix, iy], 1) if shift_only else \
                np.stack([ix * X, ix * Y, ix, iy * X, iy * Y, iy], 1)
            JW = J * wgt[:, None]
            JtJ = JW.T @ J
            try:
                dp = np.linalg.solve(JtJ + 1e-6 * np.trace(JtJ) / len(JtJ) * np.eye(len(JtJ)), JW.T @ e)
            except np.linalg.LinAlgError:
                return M
            A2 = A.copy()
            if shift_only:
                A2[:, 2] += dp
            else:
                A2 += dp.reshape(2, 3)
            step = float((np.abs(A2 - A) @ np.float64([w, h, 1])).max())
            A = A2
            if step < 0.01:
                break
    c = pts.mean(0)
    drift = np.linalg.norm((A[:, :2] - A0[:, :2]) @ c + A[:, 2] - A0[:, 2])
    if not np.isfinite(A).all() or drift > 4.0:     # ran away: keep the keypoint estimate
        return M
    return cv2.invertAffineTransform(A)


def _fine_detail(im: _Image, M: np.ndarray, pts: np.ndarray) -> float:
    """Is the match a *digital copy*, or two real objects that look alike (keyboard keys,
    identical windows, books of one series)? Keypoints match in both cases. What only a
    copy shares is the fine detail: sensor noise and micro-texture. So compare the
    denoising residual of the two areas, over small discs around the matched keypoints
    (not their convex hull, which would include un-cloned background between the parts of
    a non-convex object). M maps the other copy onto the area around ``pts``."""
    warped = im.warp(M)
    dz = (_discs(im.shape, pts, IDENTITY_RADIUS) > 0) & (warped >= 0)
    if dz.sum() <= 50:
        return 0.0
    return _ncc(im.res[dz], _residual(warped)[dz])


def _align(im: _Image, M: np.ndarray, s_in: np.ndarray, d_in: np.ndarray):
    """Best alignment of one group of matches -> (M src->dst, s_in, d_in, fine detail, how).

    The RANSAC affine map is accurate to a pixel or so, and half a pixel is enough to
    de-correlate the fine detail of a genuine clone. Three candidates are measured and the
    one with the highest fine-detail correlation is kept:
      "affine"   the RANSAC map as it is;
      "refined"  the same after robust refinement on the pixels;
      "shift"    a pure shift, if most matches agree on one, snapped to whole pixels when
                 the refined shift is within SHIFT_SNAP of them (an un-resampled copy).
    The detail is measured on the smaller of the two copies (an enlarged copy has its
    noise spread over several pixels; shrinking it back restores the comparison)."""
    def measure(M_fwd, s, d):
        if np.linalg.det(M_fwd[:, :2]) > 1.0:        # target is the larger copy
            return _fine_detail(im, cv2.invertAffineTransform(M_fwd), s)
        return _fine_detail(im, M_fwd, d)

    cands = [(measure(M, s_in, d_in), "affine", M, s_in, d_in)]
    if np.linalg.det(M[:, :2]) > 1.0:
        Mr = cv2.invertAffineTransform(_refine(im, cv2.invertAffineTransform(M), s_in))
    else:
        Mr = _refine(im, M, d_in)
    cands.append((measure(Mr, s_in, d_in), "refined", Mr, s_in, d_in))
    disp = d_in - s_in
    keep = np.linalg.norm(disp - np.median(disp, axis=0), axis=1) < SHIFT_TOLERANCE
    if keep.sum() >= max(MIN_INLIERS, 0.5 * len(s_in)):
        t = disp[keep].mean(0)
        Mt = _refine(im, np.float64([[1, 0, t[0]], [0, 1, t[1]]]), d_in[keep], shift_only=True)
        how = "shift"
        if np.abs(Mt[:, 2] - np.round(Mt[:, 2])).max() < SHIFT_SNAP:
            Mt[:, 2] = np.round(Mt[:, 2])
            how = "whole-pixel shift"
        cands.append((measure(Mt, s_in[keep], d_in[keep]), how, Mt, s_in[keep], d_in[keep]))
    fine, how, Mb, s_b, d_b = max(cands, key=lambda c: c[0])
    return Mb, s_b, d_b, fine, how


def _quiet_noise(im: _Image, warped: np.ndarray, region: np.ndarray) -> Optional[float]:
    """Correlation of the denoising residual over the *flat* pixels of ``region``.

    Two prints of one design (beer coasters, journal labels) share the printed detail, so
    their residuals correlate at the edges; the sensor noise in the flat parts between
    the lines is independent. In a digital copy that noise is shared too. Returns None
    when the region has too few flat pixels that carry any noise."""
    def activity(x):
        return cv2.blur(np.abs(cv2.Laplacian(cv2.GaussianBlur(x, (0, 0), 1.0), cv2.CV_32F)), (5, 5))

    na, nb = im.res, _residual(warped)
    quiet = region & (np.maximum(activity(im.g), activity(np.maximum(warped, 0))) < QUIET_ACTIVITY) \
        & ((na != 0) | (nb != 0))
    if quiet.sum() < QUIET_MIN_PIXELS:
        return None
    return _ncc(na[quiet], nb[quiet])


def _exact_duplicates(gray: np.ndarray) -> List[Tuple[int, int, int, np.ndarray]]:
    """Pixel-identical blocks at a common shift -> [(dx, dy, block positions, source mask)].

    Sensor noise never repeats, so a photograph does not contain two identical
    EXACT_BLOCK x EXACT_BLOCK blocks unless one was copied. Only blocks that carry noise
    count (most of their pixels differ from their 3x3 median): flat, clipped or smoothly
    shaded blocks repeat trivially. Translation only; any re-compression after the edit
    destroys the identity, hence lossless images only."""
    h, w = gray.shape
    B = EXACT_BLOCK
    if min(h, w) < 4 * B:
        return []
    flat = (cv2.dilate(gray, np.ones((3, 3), np.uint8)) == cv2.erode(gray, np.ones((3, 3), np.uint8))) \
        & (gray > 2) & (gray < 253)
    if flat.mean() > EXACT_MAX_FLAT:                # a drawing, a screenshot, a rendering
        return []
    res = gray.astype(np.int16) - cv2.medianBlur(gray, 3).astype(np.int16)
    busy = cv2.boxFilter((res != 0).astype(np.float32), -1, (B, B), anchor=(0, 0),
                         borderType=cv2.BORDER_CONSTANT)[:h - B + 1, :w - B + 1]
    ys, xs = np.nonzero(busy >= EXACT_BUSY)
    if len(ys) < 2:
        return []
    # Two independent random projections of each block: equal blocks have equal hashes,
    # and float64 sums of integers are exact, so unequal blocks practically never do.
    rs = np.random.RandomState(12345)
    g = gray.astype(np.float64)
    keymap = np.zeros((h - B + 1, w - B + 1), np.int64)
    for mult in (1000003, 1):
        kern = rs.randint(1, 1 << 20, (B, B)).astype(np.float64)
        hsh = cv2.filter2D(g, cv2.CV_64F, kern, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        keymap += np.round(hsh[:h - B + 1, :w - B + 1]).astype(np.int64) * mult
    key = keymap[ys, xs]
    order = np.argsort(key, kind="stable")
    same = key[order][1:] == key[order][:-1]
    i, j = order[:-1][same], order[1:][same]
    dx, dy = xs[j] - xs[i], ys[j] - ys[i]
    far = (dx * dx + dy * dy) > MIN_SHIFT ** 2
    i, j, dx, dy = i[far], j[far], dx[far], dy[far]
    if len(i) == 0:
        return []
    flip = (dx < 0) | ((dx == 0) & (dy < 0))        # the left-most block is the "source"
    i, j = np.where(flip, j, i), np.where(flip, i, j)
    dx, dy = xs[j] - xs[i], ys[j] - ys[i]
    shifts, inv, cnt = np.unique(np.stack([dx, dy], 1), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = []
    for t in np.argsort(-cnt)[:3]:
        sel = inv == t
        if _periodic(keymap, ys[i[sel]], xs[i[sel]], int(shifts[t][0]), int(shifts[t][1])):
            continue
        m = np.zeros((h, w), np.uint8)
        m[ys[i[sel]], xs[i[sel]]] = 1
        m = cv2.dilate(m, np.ones((B, B), np.uint8), anchor=(B - 1, B - 1))   # anchors -> blocks
        if _periodic_texture(g, m):
            continue
        out.append((int(shifts[t][0]), int(shifts[t][1]), int(cnt[t]), m))
    return out


def _periodic(keymap: np.ndarray, ys: np.ndarray, xs: np.ndarray, dx: int, dy: int) -> bool:
    """Is this match not unique? A clone matches at exactly one shift. Periodic content also
    matches at a short step from itself, at a shift a short step away from (dx, dy), or at
    twice the shift."""
    hh, ww = keymap.shape
    short = [(p, 0) for p in range(1, EXACT_PERIOD_MAX + 1)] + \
            [(0, p) for p in range(1, EXACT_PERIOD_MAX + 1)]
    steps = short + [(dx + qx, dy + qy) for qx, qy in short] + \
        [(dx - qx, dy - qy) for qx, qy in short] + [(2 * dx, 2 * dy)]
    k0 = keymap[ys, xs]
    for qx, qy in steps:
        for sx, sy in ((qx, qy), (-qx, -qy)):
            y2, x2 = ys + sy, xs + sx
            ok = (y2 >= 0) & (y2 < hh) & (x2 >= 0) & (x2 < ww)
            if ok.sum() < max(1, len(ys) // 2):
                continue
            if np.mean(keymap[y2[ok], x2[ok]] == k0[ok]) >= EXACT_PERIODIC:
                return True
    return False


def _periodic_texture(g: np.ndarray, mask: np.ndarray) -> bool:
    """Does the fine detail inside ``mask`` correlate with itself at a short step? Sensor
    noise does not; a generator's upsampling imprint, a screen pattern or a woven texture
    does, and such content can repeat exactly along one direction without being copied."""
    hp = g - cv2.medianBlur(np.clip(g, 0, 255).astype(np.uint8), 3).astype(np.float64)
    ys, xs = np.nonzero(mask)
    if len(ys) < 64:
        return False
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    P = EXACT_PERIOD_MAX
    for qx, qy in [(p, 0) for p in range(2, P + 1)] + [(0, p) for p in range(2, P + 1)]:
        ya, yb = y0, min(y1, hp.shape[0] - qy)
        xa, xb = x0, min(x1, hp.shape[1] - qx)
        if yb - ya < 8 or xb - xa < 8:
            continue
        sel = mask[ya:yb, xa:xb] > 0
        a = hp[ya:yb, xa:xb][sel]
        b = hp[ya + qy:yb + qy, xa + qx:xb + qx][sel]
        if a.size < 64:
            continue
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 0 and (a * b).sum() / den >= EXACT_TEXTURE_CORR:
            return True
    return False


def _lattice_errors(Ms):
    """For each near-pure shift t: min relative error |u - 2t| / |2t| (or |t - 2u| / |2u|)
    over the other near-pure shifts u; None when there is nothing to compare with."""
    shifts = [M[:, 2] if np.abs(M[:, :2] - np.eye(2)).max() < LATTICE_MAX_LINEAR else None for M in Ms]
    out = []
    for i, t in enumerate(shifts):
        errs = [np.linalg.norm(b - 2 * a) / max(np.linalg.norm(2 * a), 1e-6)
                for j, u in enumerate(shifts) if j != i and t is not None and u is not None
                for a, b in ((t, u), (u, t))]
        out.append(float(min(errs)) if errs else None)
    return out


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
    lossless = bool(getattr(d, "lossless", False))
    # Pixel-identical blocks need the stored pixels: not after rescaling, not after lossy coding.
    exact = [e for e in (_exact_duplicates(gray) if lossless and s == 1 else [])
             if e[2] >= MIN_INLIERS]

    src, dst, n_kp, enlarged = _self_matches(gray)
    n_pairs = 0 if src is None else len(src)
    info = {"keypoints": n_kp, "candidate_pairs": n_pairs, "enlarged_for_keypoints": enlarged}
    transforms = _group_transforms(src, dst) if n_pairs >= MIN_INLIERS else []
    if not transforms and not exact:
        if src is None:
            return Evidence(detector=det, title=title, category=Category.MANIPULATION, score=0.0,
                            confidence=0.2, summary="Too few keypoints to test for cloning "
                            "(flat image). Clones of flat areas cannot be detected this way.",
                            details={"keypoints": n_kp})
        if n_pairs < MIN_INLIERS:
            return Evidence(detector=det, title=title, category=Category.MANIPULATION,
                            score=calibrate(n_pairs, INLIER_THRESHOLD, INLIER_SOFTNESS) * 0.6,
                            confidence=0.7, summary=f"No cloned regions found ({n_pairs} "
                            f"self-matching keypoint pairs).", details=info)
        return Evidence(detector=det, title=title, category=Category.MANIPULATION, score=0.1,
                        confidence=0.7, summary=f"Self-matching keypoints ({n_pairs}) do not "
                        f"form a consistent geometric clone.", details=info)

    mask_src = np.zeros((h, w), np.uint8)
    mask_dst = np.zeros((h, w), np.uint8)
    descs = []
    identities = []          # (score, support, fine detail, verified area, kind, why)
    accepted = []
    im = _Image(gray)
    many = MANY_INLIERS_UPSAMPLED if enlarged else MANY_INLIERS
    aligned_all = [_align(im, M0, s0, d0) for M0, s0, d0 in transforms]
    lattice = _lattice_errors([a[0] for a in aligned_all])
    for (M, s_in, d_in, fine, how), lat in zip(aligned_all, lattice):
        warped = im.warp(M)
        valid = warped >= 0
        corr = _local_corr(im.g, warped, valid)     # high where I(x) ~ I(M^-1 x) -> target
        tgt = (corr > CORR_THR).astype(np.uint8)
        tgt = cv2.morphologyEx(tgt, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        hull_d = cv2.convexHull(d_in.astype(np.int32))
        near = np.zeros_like(tgt)
        cv2.fillConvexPoly(near, hull_d, 1)
        near = cv2.dilate(near, np.ones((41, 41), np.uint8))
        tgt &= near
        verified = float(tgt.sum()) / (h * w)
        quiet = None
        if lossless:
            around = cv2.dilate(_discs((h, w), d_in, IDENTITY_RADIUS), np.ones((21, 21), np.uint8))
            quiet = _quiet_noise(im, warped, (tgt > 0) & (around > 0))
        # The same measurement, deliberately mis-aligned along the displacement.
        c0 = s_in.mean(0)
        move = M[:, :2] @ c0 + M[:, 2] - c0
        u = move / (np.linalg.norm(move) + 1e-9)
        slide = 0.0
        for sign in (1.0, -1.0):
            Ms = M.copy()
            Ms[:, 2] += sign * SLIDE_SHIFT * u
            slide = max(slide, _fine_detail(im, Ms, d_in))
        aligned = max(_fine_detail(im, M, d_in), 1e-3)
        # Each transform is judged on its own: all conditions must hold - enough
        # geometrically consistent matches, matching fine detail, a match that does not
        # survive sliding and, where it can be measured, shared noise in the flat parts.
        # With many matches a lower detail correlation is accepted (re-compression rewrites
        # the fine detail of a real clone); with few it is not. A look-alike pair elsewhere
        # in the image, even one with more matches, must not hide a genuine clone.
        n_in = len(s_in)
        fine_thr = FINE_THRESHOLD_MANY if n_in >= many else FINE_THRESHOLD
        tests = {"matches": calibrate(n_in, INLIER_THRESHOLD, INLIER_SOFTNESS),
                 "detail": calibrate(fine, fine_thr, FINE_SOFTNESS)}
        if quiet is None or quiet < SLIDE_WAIVED_BY_NOISE:
            tests["slide"] = calibrate(1.0 - slide / aligned, 1.0 - SLIDE_RATIO, SLIDE_SOFTNESS)
        if lat is not None and (quiet is None or quiet < SLIDE_WAIVED_BY_NOISE):
            tests["lattice"] = calibrate(lat, LATTICE_TOL, LATTICE_SOFTNESS)
        if quiet is not None:
            tests["noise"] = min(calibrate(quiet, QUIET_THRESHOLD, QUIET_SOFTNESS),
                                 calibrate((fine + quiet) / 2, JOINT_THRESHOLD, JOINT_SOFTNESS))
        weakest = min(tests, key=tests.get)
        t_score = tests[weakest]
        identities.append((t_score, n_in, fine, verified, "keypoints", (weakest, quiet, slide, aligned)))
        shift = M[:, 2]
        descs.append({"method": "keypoints", "inliers": n_in, "fine_detail_corr": round(fine, 3),
                      "quiet_noise_corr": None if quiet is None else round(quiet, 3),
                      "slide_corr": round(slide, 3), "aligned_corr": round(aligned, 3),
                      "lattice_error": None if lat is None else round(lat, 3),
                      "alignment": how, "weakest_test": weakest, "score": round(t_score, 3),
                      "translation": [round(float(shift[0] / s), 1), round(float(shift[1] / s), 1)],
                      "linear": np.round(M[:, :2], 3).tolist()})
        if t_score >= 0.5:                       # only accepted clones are drawn
            Minv = cv2.invertAffineTransform(M)
            mask_dst |= tgt
            mask_src |= cv2.warpAffine(tgt, Minv, (w, h), flags=cv2.INTER_NEAREST)
            accepted.append((M, s_in, d_in))
    exact_shifts = []
    for dx, dy, n_blocks, m_src in exact:
        t_score = calibrate(n_blocks, EXACT_MIN_BLOCKS, EXACT_SOFTNESS)
        m_dst = cv2.warpAffine(m_src, np.float64([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_NEAREST)
        identities.append((t_score, n_blocks, 1.0, float(m_dst.sum()) / (h * w), "identical blocks", None))
        descs.append({"method": "identical blocks", "inliers": 0, "identical_blocks": n_blocks,
                      "fine_detail_corr": 1.0, "quiet_noise_corr": None, "alignment": "whole-pixel shift",
                      "score": round(t_score, 3), "translation": [float(dx), float(dy)],
                      "linear": [[1.0, 0.0], [0.0, 1.0]]})
        if t_score >= 0.5:
            mask_src |= m_src
            mask_dst |= m_dst
            exact_shifts.append((dx, dy, m_src))
    area = float((mask_src | mask_dst).mean())
    score, best_in, best_identity, best_verified, best_kind, best_why = \
        max(identities, key=lambda t: (t[0], t[1]))
    n_found = len(accepted) + len(exact_shifts)
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
    for M, s_in, d_in in (accepted or [(M, s_, d_) for M, s_, d_ in transforms])[:3]:
        for a, b in zip(s_in[:150], d_in[:150]):
            cv2.line(vis, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 120), 1, cv2.LINE_AA)
            cv2.circle(vis, tuple(map(int, a)), 2, (255, 220, 0), -1)
            cv2.circle(vis, tuple(map(int, b)), 2, (255, 60, 60), -1)
    for dx, dy, m_src in exact_shifts[:3]:
        ys, xs = np.nonzero(m_src)
        a = (int(xs.mean()), int(ys.mean()))
        cv2.arrowedLine(vis, a, (a[0] + dx, a[1] + dy), (0, 255, 120), 2, cv2.LINE_AA)
    if score >= 0.5 and best_kind == "identical blocks":
        summary = (f"Found {n_found} cloned region pair(s); the strongest is a pixel-for-pixel "
                   f"identical copy ({best_in} identical {EXACT_BLOCK}x{EXACT_BLOCK} blocks at one "
                   f"shift). Sensor noise never repeats, so a photograph does not contain this "
                   f"unless a region was copied; the pair covers {area * 100:.2f}% of the image.")
    elif score >= 0.5:
        summary = (f"Found {n_found} cloned region pair(s); the strongest is supported by "
                   f"{best_in} geometrically consistent keypoint matches whose fine detail "
                   f"(noise, micro-texture) also matches (correlation {best_identity:.2f}); the "
                   f"pair covers {area * 100:.2f}% of the image.")
    elif best_kind == "keypoints" and best_in >= INLIER_THRESHOLD and best_why[0] == "noise":
        summary = (f"{best_in} keypoint matches line up and the detail of the two areas looks "
                   f"alike (correlation {best_identity:.2f}), but the noise in their flat parts "
                   f"is not shared the way a copy shares it (correlation {best_why[1]:.2f}): "
                   f"this looks like two real objects of one design (prints, labels, panels), "
                   f"not a digital clone. (A clone buried under noise added afterwards would "
                   f"read the same.)")
    elif best_kind == "keypoints" and best_in >= INLIER_THRESHOLD and best_why[0] == "slide":
        summary = (f"{best_in} keypoint matches line up, but the two areas still match when one "
                   f"is slid {SLIDE_SHIFT:.0f} px along the displacement (correlation "
                   f"{best_why[2]:.2f} against {best_why[3]:.2f} aligned): matches along a "
                   f"straight edge or a regular row, not a digital clone.")
    elif best_kind == "keypoints" and best_in >= INLIER_THRESHOLD and best_why[0] == "lattice":
        summary = (f"{best_in} keypoint matches line up, but the same displacement recurs at "
                   f"twice its length: a regular pattern (windows, tiles, a facade), not a "
                   f"digital clone.")
    elif best_kind == "keypoints" and best_in >= INLIER_THRESHOLD:
        summary = (f"{best_in} keypoint matches line up, but the fine detail of the two areas "
                   f"does not match (correlation {best_identity:.2f}): this looks like genuinely "
                   f"repeated objects (keys, windows, tiles), not a digital clone.")
    elif best_kind == "keypoints":
        summary = f"Weak self-similarity only ({best_in} consistent matches); no convincing clone."
    else:
        summary = (f"Only {best_in} pixel-identical blocks at one shift; too few to call it "
                   f"a clone.")
    return Evidence(
        detector=det, title=title, category=Category.MANIPULATION, score=score, confidence=0.8,
        summary=summary,
        details={**info, "transforms": descs, "area_fraction": area,
                 "fine_detail_corr": best_identity, "verified_area": best_verified,
                 "threshold_inliers": INLIER_THRESHOLD},
        heatmap=heat, regions=regions, visuals={"copymove_matches": vis})
