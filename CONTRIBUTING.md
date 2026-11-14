# Contributing to Basira

Thank you for looking. This file is meant to answer everything you need without asking:
how to set up, how to run the tests and each benchmark, what a change must come with, and
how to report a wrong result.

The project's rule, in one sentence: **few reliable detectors over many flaky ones; every
claim is measured by a script in the repository; a clean file must stay unflagged.**
A measured negative result ("this does not work on X, here is the script") is a welcome
contribution. An improvement that is asserted but not measured is not.

## Set up

```bash
git clone https://github.com/OWNER/basira.git && cd basira
python -m venv .venv
.venv/bin/pip install -e ".[dev]"        # Windows: .venv\Scripts\pip
.venv/bin/basira doctor                  # versions, ffmpeg, and a self-test
```

Python 3.9 or newer. The code must keep running on 3.9: start modules with
`from __future__ import annotations`, no `match` statements, no `X | Y` unions evaluated at
runtime. Nothing needs a GPU, a model download or the network; please keep it that way —
new runtime dependencies need a very good reason.

## Run the tests

```bash
.venv/bin/python -m pytest -q                         # everything, a few minutes
.venv/bin/python -m pytest -q tests/test_image.py     # one engine
.venv/bin/python -m pytest -q -k copy_move            # one topic
```

The tests generate their own data (`tests/synth_image.py`, `tests/synth_audio.py`,
`tests/synth_video.py`): forgeries with exact ground truth, made from a simulated camera,
a simulated recording and encoded clips. No media file is stored in the repository.

CI runs the suite on Linux (Python 3.9 to 3.13) and macOS (3.9 and 3.13); see
`.github/workflows/ci.yml`.
Windows runs too but is marked experimental: it had never been run when it was added, so a
Windows failure is shown without failing the build. If you use Windows, a report of what
breaks is a useful contribution in itself.

## Run the benchmarks

The numbers in the README come from these scripts. If your change touches a detector, run
the ones that apply **before and after**, and put both sets of numbers in the pull request —
including anything that got worse.

| Script | Needs | What it reports |
|---|---|---|
| `python benchmarks/synthetic_images.py [--n 12]` | nothing | per detector: false positives on clean synthetic photos, detection and localisation per forgery family |
| `python benchmarks/synthetic_audio.py [--n 10]` | nothing | clean recordings flagged; cuts, insertions and inserted silence located |
| `python benchmarks/real_content.py --sources DIR [--videos V.mov ...] [--robust]` | a folder of your own real photographs | forgeries built on real content; with `--robust`, what survives re-compression and down-scaling |
| `python benchmarks/public_datasets.py comofod --root DIR [--variants JC]` | CoMoFoD small (download it yourself) | originals and forgeries flagged, pixel F1, ROC-AUC |
| `python benchmarks/public_datasets.py columbia --root DIR` | Columbia Uncompressed Splicing | the same |
| `python benchmarks/evaluate_folder.py --clean DIR --forged DIR [--masks DIR]` | any dataset: a folder of authentic and a folder of tampered files | ROC-AUC, true/false-positive rate, pixel F1 / IoU when masks are given |
| `python benchmarks/audio_robustness.py [--n 6 --only matrix]` | macOS (`say`) | simulated room recordings after re-encoding |
| `python benchmarks/video_robustness.py [--source A.mov ...] [--quick]` | optionally your own footage | video forgeries after the processing shared videos go through |

Each script's docstring (`--help`) says exactly what its data is and is not. No dataset is
bundled or downloaded; datasets have their own licences — never commit their files.

`python examples/make_demo.py` regenerates the demo forgeries, the figures in `docs/img/`
and HTML reports in `examples/out/`.

## What a change must come with

### Any change

- Tests pass. New behaviour has a test; a bug fix has a test that failed before it.
- Documentation stays true. If a sentence in `README.md`, `docs/` or a `--help` text
  describes what you changed, update it in the same pull request. Every command shown in
  the docs is expected to work as written.
- No new false positives. If a clean-file test or a benchmark's clean set gets worse, say
  so and explain why it is worth it — usually it is not.

### A new or changed detector

A detector is merged only with all four of these. A detector that cannot stay quiet on
clean files does not get merged; one was removed during development for exactly that
reason. [docs/ADDING_A_DETECTOR.md](docs/ADDING_A_DETECTOR.md) walks through a complete,
runnable example.

1. **A synthetic forgery with ground truth** in `tests/synth_*.py`: a generator that
   returns the file *and* where the edit is (a mask or time range), driven by a seed.
2. **A clean-file test**: the detector stays below its threshold on the existing clean
   generators (one save, re-saved, different noise levels) and on anything that might
   plausibly confuse it.
3. **Measured thresholds**: the threshold is a named constant with a comment saying what
   was measured ("clean files reach 0.40 on synthetic_images.py --n 16; threshold 0.45"),
   not a number that felt right. If you tuned it on a dataset, say which — numbers on the
   data you tuned on are optimistic, and the README says so wherever that applies.
4. **A stated blind spot**: what makes it fail or fire wrongly (re-compression, resizing,
   a file format, a kind of scene), written in `docs/METHODS.md` and, where it matters to
   the user, in the detector's own `summary` text.

Also expected: the score is calibrated so that 0.5 is the decision threshold;
`confidence` reflects whether the method applies to *this* file (return
`Evidence.not_applicable(...)` when it does not); the detector reports its raw statistic
and threshold in `details`; and it returns in reasonable time on a large file.

### Changes to the command line or the JSON output

The exit statuses, option names and the JSON fields listed as stable in
[docs/JSON.md](docs/JSON.md) are a contract with people's scripts. Adding is fine;
removing, renaming or changing a type needs a new `schema_version` and a changelog entry.

## Reporting a wrong result

A wrong result is the most useful report there is, in both directions. Use the
"False positive / false negative" issue form. What makes a report actionable:

- **The file itself**, exactly as analysed — not a screenshot of it, not a re-saved or
  re-uploaded copy (those destroy the very traces in question). If it cannot be public,
  say so; a description of how it was made is still useful. Only attach files you have the
  right to share, and nothing that identifies a private person.
- **What you know about its history**: camera or generator, editing software and what was
  done, every app or site it passed through (messengers and social networks re-compress).
- **For a false positive**: why you are confident the file is unedited (for example, you
  took the photograph and this is the camera's own file).
- **For a false negative**: what was changed and where. A mask image or a rough box is
  ideal. Check the README's Limitations first — an edit that was re-compressed or resized
  afterwards is usually an expected miss, not a bug.
- **The outputs**:
  ```bash
  basira doctor
  basira scan FILE --json --no-html > report.json
  ```

## Pull requests

- One topic per pull request. Describe what changed and what you measured.
- Follow the style of the code around you (4 spaces, type hints, lines up to 100 columns,
  comments that explain *why*).
- Add a line to the `Unreleased` section of `CHANGELOG.md` for anything a user would notice.
- The pull-request template has the checklist.

## Security

Basira parses untrusted files. If you find a file that makes it crash a process, hang,
read something it should not, or reach the network, please follow `SECURITY.md` rather
than opening a public issue.

## Releases (maintainers)

1. Set the same version in `basira/__init__.py`, `pyproject.toml` and `CITATION.cff`
   (`tests/test_packaging.py` fails if they differ) and date the section in `CHANGELOG.md`.
2. `python -m pip install build && python -m build && python .github/scripts/check_dist.py dist`
3. Tag `vX.Y.Z` and push the tag. `.github/workflows/release.yml` builds again and, only if
   the owner has enabled PyPI publishing as described at the top of that file, uploads.

## Licence

By contributing you agree that your contribution is released under the MIT licence of
this repository.
