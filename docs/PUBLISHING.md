# Publishing checklist

Published on 2026-11-16 at <https://github.com/Thulfekar90/basira> (public), with the
description and topics below. The first CI run failed on Linux and Windows; the fixes are
listed under "First CI run". Nothing has been released on PyPI.

- Description: "Explainable forensics for edited and AI-generated images, audio and video:
  shows where a file was changed and why."
- Topics: `forensics`, `image-forensics`, `audio-forensics`, `video-forensics`,
  `deepfake-detection`, `tamper-detection`, `c2pa`, `copy-move`, `python`.

## Rules for every later push (people and agents)

1. The whole test suite (`.venv/bin/python -m pytest -q`) passes locally first.
2. Push to `main` only with the owner's approval. Never force-push. Do not push `.venv/`,
   `examples/out/`, datasets or any file not tracked by git.
3. Check the GitHub Actions run for the push. Every required job must be green (the
   Windows jobs are experimental and do not turn the run red). If one fails, report the job
   and its log; do not disable the workflow.
4. Do not create a release, a PyPI upload or any social-media post: those are the owner's
   decisions.

## First CI run (2026-11-16)

Run #1 failed on every Linux and Windows job, although the suite had passed on macOS and in
a Linux Docker image. Fixes, made in the following commit:

- `benchmarks/video_robustness.py` read decoded video with a single `read(size * n)` and
  `n = 10**6`, which asks for hundreds of GB up front and raises `MemoryError` on Linux and
  Windows (37 test errors). It now reads frame by frame.
- `ubuntu-latest` is Ubuntu 26.04, which has no Python 3.9 build; the 3.9 job runs on
  `ubuntu-24.04`.
- Windows only: `tests/test_readme_tables.py` read the Arabic README in the system code page
  (now UTF-8), and `tests/test_report.py` used `<`, `>` and `"` in file names, which Windows
  refuses (it now uses the characters Windows allows).

## Before announcing it (owner's decisions)

- **Hero image.** Done: the owner's own photo with trees cloned into the fog
  (`docs/img/hero_snow.jpg`). Still to add: a screenshot of the HTML report.
- **More public-dataset numbers.** Columbia and CoMoFoD (in-sample) and CASIA v1, CASIA v2
  and COVERAGE (out-of-sample) are in the README. Still missing: IMD2020 (real-world
  edits) for images, and any real data at all for audio (ASVspoof) and video.
- **Name.** `basira` was free on PyPI on 2026-11-16; check again before a release.
- **PyPI release** once the above is done: `python -m build && twine upload dist/*`.
