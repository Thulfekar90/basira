# Working rules for AI agents and contributors

Basira is a forensic tool: its output may be used against a person. These rules exist so
that every number in the README stays true. They apply to people, Codex, Claude Code and
any other agent working in this repository.

## Evidence before claims

- A detector, threshold or fix is accepted only with a measurement produced by a script in
  `benchmarks/` or `tests/`. Report what the script printed, including bad numbers.
- **False alarms matter more than misses.** A change that raises detection but adds false
  alarms on clean material is rejected unless the trade-off is stated and approved.
- Every new detector must be silent on clean files (measured, with margin), expose its raw
  statistic and threshold in `details`, explain itself in English and Arabic in
  `basira/core/report_html.py`, and never say "authentic". See `docs/ADDING_A_DETECTOR.md`.

## Development and held-out data

- Thresholds are chosen on a **development** split only. **Held-out** data is run once, after
  the code is frozen, and its numbers go into the README whatever they are.
- Current splits: CASIA v2 is split by a SHA-1 hash of the target-image id (even = dev,
  odd = test). Columbia and CoMoFoD are in-sample (tuned on). CASIA v1 and COVERAGE were
  first run as out-of-sample on 2026-11-14 with frozen thresholds; tuning on them makes
  them in-sample and must be declared.
- Never tune on a held-out set and still call it out-of-sample. If you have looked, say so.

## Regression gate (run before every hand-off)

```bash
.venv/bin/python -m pytest -q
.venv/bin/python benchmarks/public_datasets.py columbia --root DATASETS/columbia
.venv/bin/python benchmarks/public_datasets.py comofod --root DATASETS/comofod --variants JC
.venv/bin/python benchmarks/synthetic_images.py
```

Plus the audio and video robustness benchmarks when those engines change. A hand-off that
did not run a step says so explicitly.

## Data, network and safety

- Public datasets are research-use only and are **never committed**. They live outside the
  repository.
- No downloads, network services or new dependencies without the maintainer's explicit
  approval; state name, source, size and licence when asking.
- Untrusted media is the input. Keep decoder hardening (`basira/core/io.py`, size and pixel
  caps, FFmpeg whitelists) intact; add a test for every parser change.
- No `git push`, no publishing, no repository settings changes without the maintainer.

## Style

- Python 3.9+, NumPy/OpenCV, no deep-learning runtime in the core package.
- Plain-language summaries; Arabic and English docs are kept in step
  (`README.md` / `README.ar.md`).
