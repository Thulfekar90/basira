# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is 0.x,
detector thresholds, scores and wording may change in any release; what a 0.x release may
*not* change silently is listed in [docs/JSON.md](docs/JSON.md).

## [0.1.0] - Unreleased

Nothing has been published yet. This section is derived from the commit history
(`git log`) and describes what the first release will contain.

### Fixed

- Short generator markers (`SDXL`, `Sora`, ...) found by chance inside compressed image
  data were reported as AI-generator metadata; they now count only when surrounded by text.

### Added

- **`image.jpeg_trace`**: maps pasted content in losslessly stored images through the
  quantisation of the JPEG they once were. Tuned on the CASIA v2 development half; on the
  held-out half TIFF splices found went from 9/334 to 63/334 with no added false alarm.
  `public_datasets.py casia2 --half dev|test` reproduces the split. `AGENTS.md` sets the
  rules for contributors and AI agents.

- **"How the verdict was reached"** in the HTML report: a diagram of the analysis (file →
  each detector → category scores → verdict) drawn only from the report's own numbers. A
  detector's line is as thick as its weight in the verdict; quiet, not-applicable and failed
  detectors are drawn as such. Static SVG with the same path written out as five steps, so
  it reads without JavaScript and in print; the script replays it (respecting reduced
  motion), and it is mirrored for the Arabic report.

- **Out-of-sample image benchmark** on CASIA v1.0, CASIA v2.0 (corrected masks) and
  COVERAGE, run once with thresholds frozen beforehand (`benchmarks/public_datasets.py
  casia1|casia2|coverage`, optional per-image `--csv`). Results, including the 12% false
  alarms on COVERAGE originals, are in the README.

- **Three engines over one evidence model.** Every detector returns its raw statistic, its
  threshold, a plain-language sentence and, where it applies, a heat-map or time ranges;
  an engine fuses them into category scores and a verdict that never says "authentic".
- **Image**: JPEG double-quantisation map, copy-move (keypoints verified on fine detail),
  colour-filter-array phase test, sensor-noise consistency, texture-normalised error level
  analysis, generator upsampling fingerprint for lossless files, generator metadata
  (validated structurally), C2PA / IPTC labels, EXIF provenance.
- **Audio**: mains-hum (ENF) phase jumps, splices from room-noise and bandwidth changes,
  inserted digital silence, hard-cut clicks, band-limiting and unnaturally clean pauses,
  generator tags.
- **Video**: removed frames inside a shot, repeated and frozen frames, container edit
  lists and generator tags (source-noise changes and a previous encoding's GOP are
  reported as information only); the sound track is analysed with the audio
  engine and its findings appear on the video's report.
- **Reports**: terminal summary, JSON, and one self-contained HTML file per input in English or Arabic (strict Content-Security-Policy, per-detector layers, before/after slider, time-axis lanes for audio and video).
- **Command line**: `basira scan` for files, folders (`--recursive`) and patterns, with
  `--json`, `--jsonl`, `--csv`, `--report-dir`, `--lang`, `--fail-on`, `--jobs`,
  `--timeout`, `--quiet`, `--no-color` and `--debug`; a worst-first summary table for
  batches; `basira doctor` to check an installation; `basira serve` for a drag-and-drop
  page on 127.0.0.1.
- **Stable JSON output**: `schema_version: 1`, strictly valid JSON (no NaN / Infinity),
  documented in [docs/JSON.md](docs/JSON.md).
- **Benchmarks that reproduce the published numbers**: public datasets (Columbia
  Uncompressed Splicing, CoMoFoD), forgeries built on your own real photographs and video
  frames with a re-compression / down-scaling robustness matrix, synthetic images with a
  simulated camera, synthetic audio, an audio robustness matrix (rooms, codecs, held-out
  voices) and a video robustness matrix (codecs, natural events);
  `benchmarks/evaluate_folder.py` for your own data.
- A seeded mutation fuzzer (`tests/fuzz_harness.py`) with a subprocess oracle that watches
  for crashes, hangs, memory use, process creation and network access.
- Synthetic forgeries with ground truth for every detector family, used by the tests.
- Documentation: methods, usage guide, JSON reference, contributor guide, publishing
  checklist; Arabic README.

### Changed (during development, before any release)

- Detectors were reworked after the first run on public data flagged 23 of 183 Columbia
  originals and more than half of the CoMoFoD originals at JPEG quality 90: copy-move now
  requires the fine detail of both areas to correlate, the CFA test became a Bayer
  phase-inversion test, the noise detector only counts abnormally smooth regions, the
  double-quantisation detector rejects smeared combs, and the spectral threshold was raised
  above what camera originals produce.
- The verdict follows the stronger single category; unrelated weak hints no longer add up.
- The double-quantisation detector was rewritten (folded between/within-phase variance for
  comb detection; block scores normalised under both hypotheses).
- Truncated images are decoded and noted instead of raising.
- Copy-move was rewritten: small images are enlarged before keypoint search, and every
  candidate clone must pass fine-detail, slide, shared-noise and lattice tests. On CoMoFoD
  (in-sample) originals flagged 12 → 0 and forgeries found 108 → 159; it is weaker at
  JPEG quality 20–30 and on clones covered by added noise. Identical-block matches in periodic texture
  (a generator's upsampling imprint) are rejected, so generated images are not called clones.
- Audio splice detection now compares settled pauses by spectral shape (level changes
  alone are automatic gain control); ENF uses a joint line + step phase fit with stricter
  applicability checks. Clean-file false positives over the robustness matrix 15/108 → 0.
- Video thresholds are relative to the clip's own motion; a removal needs the two frames
  to be the same scene, so hard cuts are no longer called removals. False positives on
  real-derived natural events 55/207 → 0; inserted footage with hard cuts is no longer
  claimed as detectable.
- Metadata signature scanning is about 8× faster (literal pre-filter).
- The spectral statistic is measured against an off-lattice reference, so plain noise no
  longer looks periodic.

### Removed

- One detector was removed during development because it could not stay quiet on clean
  files (see the README's "Develop" section).

### Security

- Decoders are restricted to local files; playlist and reference formats are refused.
- The local web UI checks the `Host` header and requires a per-session token to upload.
- Pixel and byte limits are enforced before an image is decoded; MP4 box sizes are bounded.
- A file that cannot be decoded, or a detector that crashes, is reported as inconclusive
  rather than as clean.
- Generator names found in free text are treated as hints, not as proof.
- Fuzzing fixes: only allow-listed image formats are opened (EPS no longer starts
  Ghostscript), audio shape and size are checked before allocation, FFmpeg runs with
  protocol and format allow-lists and an allocation cap, decoder messages no longer
  contain paths, a linear XMP scan replaces a backtracking regular expression, and an
  image or audio stream with nothing decodable is refused instead of reported clean.
- A video whose declared sound track cannot be decoded is reported as partly failed.
- `basira scan` refuses a `--csv` path that is one of its inputs, and a timed-out or
  crashed worker is killed together with any decoder it started (POSIX).
- The local server rejects malformed lengths, short bodies, non-ASCII tokens and look-alike
  paths with a proper status, drops silent clients after 60 s, and writes the report in
  the page's language.
- [SECURITY.md](SECURITY.md) documents the threat model, the limits and what is not covered.

[0.1.0]: https://github.com/OWNER/basira
