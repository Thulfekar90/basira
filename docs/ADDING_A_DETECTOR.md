# Adding a detector

A worked example, start to finish, small enough to read in ten minutes. The Python blocks
on this page are executed by the test-suite (`tests/test_packaging.py` runs them in
order), so they are known to work with the code they ship with.

The example detector, **exact duplicate blocks**, is deliberately naive. It passes its own
synthetic tests perfectly and then fails on public data — which is the most important
thing this page has to show. It is not part of Basira and would not be merged.

The four things every detector needs ([CONTRIBUTING.md](../CONTRIBUTING.md)):

1. a synthetic forgery with ground truth,
2. a clean-file test,
3. a measured threshold,
4. a stated blind spot.

## 1. The idea, and its precondition

*If a region was copied inside an image and the file was then saved losslessly, the copy
is pixel-identical to its source. Camera sensor noise makes two textured blocks identical
by chance practically impossible.* So: cut the image into 16×16 blocks, and count the
textured blocks that have an exact twin.

Write the precondition down before the code, because it decides `confidence`:

- **lossless files only** — JPEG re-compression changes every block;
- only **textured** blocks — flat sky or clipped highlights repeat naturally;
- blocks are compared on a fixed grid, so the copy must be shifted by a **multiple of 16
  pixels**. That is a severe restriction. Keep it in mind for step 5.

## 2. The detector

A detector is a function that takes the engine's input object and returns one
`basira.core.types.Evidence`. For images the input is `basira.image.load.ImageData`
(`d.rgb` is the H×W×3 `uint8` array in stored orientation; `d.lossless`, `d.is_jpeg`,
`d.jpeg_quality` describe the file). In the package this file would be
`basira/image/dupblock.py` and would use relative imports.

```python
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from basira.core.types import Category, Evidence, Region
from basira.core.utils import calibrate
from basira.image.load import ImageData, load_image

DETECTOR, TITLE = "image.dupblock", "Exact duplicate blocks"
BLOCK = 16
MIN_STD = 4.0            # flatter blocks (sky, borders, clipped highlights) repeat by chance
STAT_THRESHOLD = 0.004   # share of textured blocks with an exact twin; measured in step 4
STAT_SOFTNESS = 0.002


def analyze_dupblock(d: ImageData) -> Evidence:
    if not d.lossless:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.MANIPULATION,
                                       "lossy file: re-compression changes every block")
    h, w = d.rgb.shape[:2]
    hb, wb = h // BLOCK, w // BLOCK
    blocks = d.rgb[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK, 3)
    blocks = blocks.transpose(0, 2, 1, 3, 4).reshape(hb * wb, -1)
    textured = blocks.std(axis=1) >= MIN_STD
    if textured.sum() < 64:
        return Evidence.not_applicable(DETECTOR, TITLE, Category.MANIPULATION,
                                       "too small or almost no texture")
    _, inverse, counts = np.unique(blocks, axis=0, return_inverse=True, return_counts=True)
    twin = ((counts[inverse.ravel()] > 1) & textured).reshape(hb, wb)

    stat = float(twin.sum() / textured.sum())                # the raw statistic
    score = calibrate(stat, STAT_THRESHOLD, STAT_SOFTNESS)   # 0.5 exactly at the threshold
    heat = np.kron(twin, np.ones((BLOCK, BLOCK))).astype(np.float32)
    heat = np.pad(heat, ((0, h - hb * BLOCK), (0, w - wb * BLOCK)))
    regions = []
    if score >= 0.5:
        ys, xs = np.nonzero(twin)
        regions = [Region(int(xs.min()) * BLOCK, int(ys.min()) * BLOCK,
                          int(xs.max() - xs.min() + 1) * BLOCK,
                          int(ys.max() - ys.min() + 1) * BLOCK, score, "identical blocks")]
    return Evidence(
        detector=DETECTOR, title=TITLE, category=Category.MANIPULATION,
        score=score,
        confidence=0.6,      # how far to trust it on this input (it only runs on lossless files)
        summary=(f"{int(twin.sum())} textured {BLOCK}x{BLOCK} blocks ({stat * 100:.1f}%) have a "
                 f"pixel-identical twin elsewhere in the image."
                 if score >= 0.5 else
                 f"No meaningful share of textured blocks repeats exactly "
                 f"(statistic {stat:.4f} < {STAT_THRESHOLD})."),
        details={"statistic": stat, "threshold": STAT_THRESHOLD, "block": BLOCK,
                 "duplicated_blocks": int(twin.sum()), "textured_blocks": int(textured.sum())},
        heatmap=heat, regions=regions)
```

What to notice:

- **`score`** is not the statistic. `calibrate(stat, threshold, softness)` maps it so that
  0.5 is the decision threshold, which is what lets the engine combine detectors.
- **`confidence`** says whether the method applies to this input. When it does not, return
  `Evidence.not_applicable(...)`: score 0, confidence 0, and a reason the user can read.
- **`details`** carries the raw statistic and the threshold. Nothing is a black box.
- **`summary`** is one or two plain sentences, in both the quiet and the firing case.
- **`heatmap`** is H×W, 0..1, in image coordinates; **`regions`** are boxes.
- Do not catch your own exceptions to hide them. The engine wraps every detector: a crash
  becomes an `Evidence` with `error` set, and the report is marked inconclusive.

## 3. A forgery with ground truth, and a clean file

These generators belong in `tests/synth_image.py`. `S.camera_photo(seed)` returns a
synthetic photograph with Bayer sampling, sensor noise and demosaicing; every generator is
driven by a seed and returns the path **and the mask of what was changed**.

```python
import synth_image as S          # tests/ is on sys.path under pytest


def make_clean_png(out: str, seed: int = 0):
    img = S.camera_photo(seed)
    path = os.path.join(out, f"clean_{seed}.png")
    Image.fromarray(img).save(path)
    return path, np.zeros(img.shape[:2], bool)


def make_block_clone(out: str, seed: int = 0):
    """Copy a 96x96 patch elsewhere, shifted by a multiple of 16 px; save as PNG."""
    rng = np.random.RandomState(7000 + seed)
    img = S.camera_photo(seed).copy()
    h, w = img.shape[:2]
    s = 96
    sy, sx = [int(v) * 16 for v in (rng.randint(1, 10), rng.randint(1, 20))]
    ty, tx = [int(v) * 16 for v in (rng.randint(24, (h - s) // 16),
                                    rng.randint(30, (w - s) // 16))]
    img[ty:ty + s, tx:tx + s] = img[sy:sy + s, sx:sx + s]
    mask = np.zeros((h, w), bool)
    mask[ty:ty + s, tx:tx + s] = True
    mask[sy:sy + s, sx:sx + s] = True      # source and copy cannot be told apart
    path = os.path.join(out, f"blockclone_{seed}.png")
    Image.fromarray(img).save(path)
    return path, mask
```

The tests (they belong in `tests/test_image.py`) check three things: quiet on clean files,
fires **in the right place** on the forgery, and declares itself not applicable outside
its precondition.

```python
def test_dupblock_stays_quiet_on_clean_files(tmp_path):
    for seed in range(3):
        path, _ = make_clean_png(str(tmp_path), seed)
        e = analyze_dupblock(load_image(path))
        assert e.error is None and e.score < 0.5, (seed, e.details)


def test_dupblock_finds_and_locates_the_clone(tmp_path):
    for seed in range(3):
        path, mask = make_block_clone(str(tmp_path), seed)
        e = analyze_dupblock(load_image(path))
        assert e.score > 0.5, (seed, e.details)
        assert S.iou(e.heatmap, mask) > 0.8        # the right pixels, not just "something"


def test_dupblock_knows_its_blind_spot(tmp_path):
    path, _ = S.make_copymove(str(tmp_path), 0)     # a JPEG: nothing survives exactly
    e = analyze_dupblock(load_image(path))
    assert e.confidence == 0.0 and e.summary.startswith("Not applicable")
```

To make the engine run the detector, add one line to `PIXEL_DETECTORS` in
`basira/image/__init__.py`. (The audio and video engines call their detectors one by one
in `analyze_audio` / `analyze_video`, each wrapped so that a crash becomes a failed
`Evidence`; add yours the same way.)

```diff
 PIXEL_DETECTORS = [
     ("image.jpeg_dq", "JPEG double-quantisation map", Category.MANIPULATION, analyze_double_quantization),
+    ("image.dupblock", "Exact duplicate blocks", Category.MANIPULATION, analyze_dupblock),
```

From then on `basira.analyze(path).get("image.dupblock")` returns its evidence, it takes
part in the verdict, and it appears in the terminal summary, the JSON and the HTML report.
The HTML report also shows a short description of each detector (what it measures, its
main blind spot, an Arabic name) taken from the `DETECTORS` table in
`basira/core/report_html.py`; add an entry there, using only facts that are also in
`docs/METHODS.md`.

## 4. Measure the threshold

Do not pick a threshold; measure the statistic on clean files and on forgeries, and put
the threshold in the gap. For this example, on 8 synthetic clean PNGs and 8 block clones
(seeds 0–7 of the generators above):

| | statistic (share of textured blocks with a twin) |
|---|---|
| clean, 8 files | 0.0000 on every file |
| block clone, 8 files | 0.0234 – 0.0235 |

Hence `STAT_THRESHOLD = 0.004`, with the measurement written next to the constant. The
image benchmark (`python benchmarks/synthetic_images.py`) prints this kind of table for
every registered detector; add your forgery family to it.

So far: 0 of 8 clean files flagged, 8 of 8 forgeries found, localisation IoU 1.00. It looks
finished.

## 5. Run it on data you did not make

Synthetic data validates the implementation. It says little about photographs.
`CoMoFoD small` is a public copy-move dataset of 200 originals and 200 forgeries
(lossless PNG) — exactly this detector's home ground. Measured with the code above:

| CoMoFoD small | flagged |
|---|---|
| originals | **11 / 200** |
| forgeries | 11 / 200 — the *same* 11 scenes |

The flagged forgeries are exactly the scenes whose originals are flagged too, so there is
no sign that it found a single real clone (their offsets are not multiples of 16 pixels,
so no block has an aligned twin) — and it flagged 11 untouched originals whose content
happens to contain exactly repeated blocks. Perfect on its own test data; worse than
useless on real data. The project's history has the same lesson at full scale: the first
run on public datasets flagged 23 of 183 Columbia originals, and four detectors had to be
reworked because of it (README, "Measured results").

Run `benchmarks/public_datasets.py`, `benchmarks/real_content.py` on your own photographs
and `benchmarks/evaluate_folder.py` on any dataset you have **before** proposing a
detector, and report the numbers whatever they are.

## 6. State the blind spots

In `docs/METHODS.md` and in the pull request. For this example they would read:

- lossless files only; one JPEG save erases the evidence;
- misses every clone whose offset is not a multiple of 16 pixels in both directions, which
  is nearly all real ones (on CoMoFoD it flags no forgery that it does not also flag as an
  original);
- fires on originals that contain exactly repeated textured blocks (11 of 200 on CoMoFoD);
- scaled, rotated or blended copies are invisible.

Basira's real copy-move detector (`basira/image/copymove.py`) exists because of the second
point: it matches keypoints, so the offset is arbitrary, and then checks that the fine
detail of both areas also matches.

## Checklist

- [ ] The detector returns `Evidence` with a calibrated `score`, an honest `confidence`,
      raw numbers in `details`, and `not_applicable` outside its precondition.
- [ ] A seeded forgery generator with ground truth in `tests/synth_*.py`.
- [ ] Tests: quiet on clean files, fires and localises on the forgery, not applicable
      where it should be.
- [ ] The threshold is a named constant with the measurement in a comment.
- [ ] Numbers from at least one benchmark on data you did not generate, before and after,
      including false positives.
- [ ] Blind spots in `docs/METHODS.md`; a line in `CHANGELOG.md`.
