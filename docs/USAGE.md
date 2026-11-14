# Using Basira

Task by task. Every command on this page was run against the code it ships with.
What Basira can and cannot detect is in the README's
[Measured results](../README.md#measured-results) and
[Limitations](../README.md#limitations-read-this); this page does not promise more than those do.

- [Check that it works](#check-that-it-works)
- [Check one file](#check-one-file)
- [Check a folder](#check-a-folder)
- [Use it from a script or in CI](#use-it-from-a-script-or-in-ci)
- [Use it from Python](#use-it-from-python)
- [Read the report](#read-the-report)
- [When something goes wrong](#when-something-goes-wrong)
- [Common questions](#common-questions)
- [Command reference](#command-reference)

## Check that it works

```console
$ basira doctor
basira 0.1.0  (/path/to/site-packages/basira)
  ok      Python          3.9.6 (CPython, Darwin arm64)
  ok      executable      /path/to/python
  ok      NumPy           2.0.2
  ok      SciPy           1.13.1
  ok      OpenCV          5.0.0  (video decoding through its own FFmpeg: yes)
  ok      Pillow          11.3.0  (jpeg png tiff webp avif)
  ok      soundfile       0.13.1  (1.2.2: wav flac ogg mp3)
  ok      ffmpeg          7.1  (bundled with imageio-ffmpeg: /path/to/ffmpeg-macos-aarch64-v7.1)
  ok      ffmpeg decoders h264 hevc aac mp3 opus  (used for sound tracks and compressed audio)
  ok      self-test       640x480 JPEG analysed in 0.2 s, 9 detectors ran, verdict "No significant traces found"
Everything Basira needs is present and working.
```

`doctor` prints the versions it runs on, says whether the `ffmpeg` binary is the one on your
`PATH` or the one bundled with `imageio-ffmpeg`, and analyses a generated image from start
to finish (decode, every image detector, JSON, HTML). It exits with status 1 if a required
library or `ffmpeg` is missing or the self-test fails. A missing FFmpeg decoder is only a
warning (sound in that format cannot be read; everything else works) unless you pass
`--strict`. Paste this output into bug reports.

## Check one file

```bash
basira scan photo.jpg
```

prints a summary and writes `photo.jpg.basira.html` next to the file:

```console
$ basira scan examples/out/splice_2.jpg
splice_2.jpg  [image]
  Likely manipulated
  manipulation  ████████████████░░░░  80.0%
  AI generation ░░░░░░░░░░░░░░░░░░░░   0.0%
  provenance    ░░░░░░░░░░░░░░░░░░░░   0.0%
  · [JPEG double-quantisation map] The image was JPEG-compressed at least twice (8 DCT frequencies show double-quantisation periodicity), and a compact region (3.5% of blocks) does NOT share that history — consistent with content pasted in before the last save.
  report: examples/out/splice_2.jpg.basira.html
```

(`examples/out/splice_2.jpg` is a forgery with known ground truth made by
`python examples/make_demo.py`; use it to see what a positive looks like.)

Audio and video work the same way: `basira scan voice.wav`, `basira scan clip.mp4`.
For a video the sound track is analysed too and appears indented under the video.

| You want | Add |
|---|---|
| the report somewhere else | `--html report.html` (one file) or `--report-dir reports/` |
| no report file at all | `--no-html` |
| the report in Arabic (right-to-left) | `--lang ar` |
| machine-readable output | `--json` (see [docs/JSON.md](JSON.md)) |

## Check a folder

```bash
basira scan evidence/ --recursive --report-dir reports/ --csv summary.csv
```

- A folder contributes the media files directly inside it; `--recursive` (`-r`) also
  searches its sub-folders. Files that are not images, audio or video are skipped and
  counted in one line (`basira: skipped 3 unsupported files ...`), not reported one by one.
  Hidden files and earlier `*.basira.html` reports are ignored.
- Patterns work even where the shell does not expand them (Windows, or a quoted
  `'evidence/*.jpg'`).
- A file you name explicitly is different: if it is missing, unreadable or not media, that
  is an error and the exit status is 1.
- Without `--report-dir`, each report is written next to its file. With it, the reports
  keep the folder layout (`reports/2026/march/scan.jpg.basira.html`), so two files with the
  same name never overwrite each other.
- `--csv` writes one line per file for a spreadsheet: `path, media_type, verdict_level,
  verdict, manipulation, ai_generation, provenance, overall, detectors_failed, elapsed_sec,
  report, error`. It is UTF-8 with a byte-order mark so that Excel shows Arabic file names
  correctly. A cell that would start with `=`, `+`, `-` or `@` is prefixed with `'`, because
  spreadsheets would otherwise run it as a formula and a file name is not trusted text.

A run over several files ends with a table, worst first:

```console
lvl  verdict                      manip.      AI     time  type   file
  3  Likely manipulated            86.1%    0.0%     0.5s  image  evidence/IMG_0002.jpg
  3  Likely manipulated            80.0%    0.0%     0.6s  image  evidence/2026/march/scan.jpg
  0  No significant traces found    0.0%    0.0%     1.2s  image  evidence/IMG_0001.jpg
  0  No significant traces found    0.0%    0.0%     1.1s  audio  evidence/call.wav
  0  No significant traces found    0.0%    0.0%     0.5s  image  evidence/صورة 📷.jpg
5 files in 4.4 s: 2 flagged (level 2 or higher), 3 not flagged, 1 unsupported skipped
summary: summary.csv
```

The file name is the last column on purpose: everything before it is plain ASCII, so
Arabic, CJK or emoji names cannot push the columns out of line.

**Large batches.** `--jobs N` (`-j N`, `-j 0` = one per CPU core) analyses N files at the
same time in separate processes. `--timeout SECONDS` gives up on a single file after that
long. With either option each file is analysed in a worker process, so a file that crashes
a decoder or never finishes costs that one file — it is listed as *not analysed* — and the
rest of the batch carries on. Each worker needs a second or two to start, so `--jobs` only
pays off for more than a handful of files; measure on your own machine before relying on a
speed-up.

On a terminal a progress line (`[12/340] analysing IMG_0412.jpg`) is shown on stderr. It
is never written to a pipe or a log file.

## Use it from a script or in CI

**Exit status** of `basira scan`:

| Status | Meaning |
|---|---|
| 0 | every input was analysed and nothing reached `--fail-on` |
| 1 | at least one input could not be analysed (missing, unreadable, not media, too large, crashed, timed out), a report or CSV could not be written, or the command line was wrong |
| 2 | `--fail-on N` was given and at least one verdict level was `N` or higher (this wins over 1) |
| 130 | interrupted with Ctrl+C |

Verdict levels: 0 *No significant traces found*, 1 *Inconclusive*, 2 *Suspicious*,
3 *Likely manipulated / AI-generated*, 4 decisive provenance evidence (for example a
generator's own metadata).

```bash
# Fail the job if anything in uploads/ is "Suspicious" or worse; print errors only.
basira scan uploads/ --recursive --no-html --quiet --fail-on 2
```

**JSON.** `--json` prints one JSON document on stdout and nothing else: an object for a
single file, an array for several files or a folder. `--jsonl` prints one object per line
as each file finishes, which suits long batches. Messages and progress go to stderr. The
layout is documented field by field, with the stability promise, in [docs/JSON.md](JSON.md).

```bash
basira scan photo.jpg --json --no-html | jq '{verdict, verdict_level, scores}'

# which detectors fired on this file
basira scan photo.jpg --json --no-html \
  | jq -r '.evidences[] | select(.score >= 0.5) | "\(.detector)\t\(.score)\t\(.confidence)"'

# level and path of every file in a folder, as it is finished
basira scan evidence/ -r --jsonl --no-html \
  | jq -r 'select(.error == null) | [.verdict_level, .path] | @tsv'
```

An input that could not be analysed appears in the JSON as an error record
(`{"path": ..., "error": {"kind": ..., "message": ...}}`), so the output always has one
entry per input.

**Colour** is used only when stdout is a terminal. `--no-color` or a non-empty `NO_COLOR`
environment variable turns it off.

**GitHub Actions** (not run by this project's CI; adapt the paths):

```yaml
- run: pip install git+https://github.com/OWNER/basira.git
- run: basira scan assets/ --recursive --no-html --quiet --fail-on 2 --csv basira.csv
- uses: actions/upload-artifact@v4
  if: always()
  with: { name: basira-summary, path: basira.csv }
```

A caution for automation: Basira misses more than it catches (see the README's measured
results). `--fail-on` is a tripwire for the edits it can see, not a guarantee that what
passes is unedited.

## Use it from Python

```python
import basira

r = basira.analyze("photo.jpg")
print(r.verdict, r.verdict_level)      # "Likely manipulated" 3
print(r.scores)                        # manipulation / ai_generation / provenance / overall
for e in r.evidences:                  # one entry per detector, including the quiet ones
    print(e.detector, e.score, e.confidence, e.summary)

r.fused_heatmap      # HxW float32 localisation map, or None (images, video key frames)
r.fused_segments     # suspicious time ranges (audio, video)

data = r.to_dict()   # the same structure `basira scan --json` prints (docs/JSON.md)

from basira.core.report_html import save_html
save_html(r, "photo.report.html")              # lang="ar" for the Arabic report
```

`basira.analyze` raises `ValueError` for a file it does not recognise or refuses (a
playlist, an image over the size limit) and other exceptions for files it cannot decode;
a detector that crashes does not raise — it is recorded in that detector's `error` field
and counted in `r.info["detectors_failed"]`.

## Read the report

- **The verdict follows the strongest single category.** Weak hints of different kinds are
  not added up into an accusation.
- **Scores**: every detector's `score` is scaled so that 0.5 is its own decision
  threshold. `confidence` says how much that detector can be trusted on this particular
  file (error-level analysis on a PNG gets almost none). Only the part of a score above
  0.5 counts, weighted by confidence.
- **"No significant traces found" is not "authentic".** Basira never says a file is
  genuine: re-compression, resizing and careful editing erase the traces it relies on.
- **"Inconclusive (part of the analysis failed)"** means a detector crashed or the content
  could not be decoded, so the absence of findings means little. The failed detectors are
  listed.
- The HTML report shows the localisation map over the image, each detector's own picture,
  its raw measurements and thresholds, and a timeline for audio and video. It is one
  self-contained file and loads nothing from the network.
- How each detector works, with its thresholds and blind spots: [docs/METHODS.md](METHODS.md).

## When something goes wrong

Errors are one line each on stderr, prefixed `basira:`. Add `--debug` to see the Python
traceback behind one.

| Message | Meaning |
|---|---|
| `no such file: X` | the path does not exist |
| `cannot read X: Permission denied` | the file exists but cannot be opened |
| `empty file: X` | zero bytes |
| `unsupported file type: X (...)` | not an image, audio or video file Basira reads; playlists and other files that only *point* at media are refused on purpose |
| `file too large: X: ...` | over a built-in limit, e.g. an image above 60 megapixels or 300 MB; the message states the limit |
| `failed on X: ...` | the file was recognised but could not be decoded or analysed |
| `timed out after N s: X` | `--timeout` expired for this file |
| `analysis process crashed (...) on X` | a decoder took the worker process down (`--jobs` / `--timeout` runs) |
| `could not write report ...` | the analysis succeeded but the HTML file could not be written |

If a file you believe is ordinary media fails, run `basira doctor` and open an issue with
both outputs (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

## Common questions

**Why does it say nothing about my AI-generated image?**
Because that is the weakest question Basira answers, and the README says so. Without a
trained model it recognises AI generation reliably only through provenance: generator
metadata, C2PA Content Credentials and IPTC `trainedAlgorithmicMedia` labels — decisive
when present, and trivially stripped. The pixel-level generator fingerprint works on
lossless files that were not resized and is blind on JPEGs. A modern diffusion image
re-saved by a social network will usually come back as *"No significant traces found"*.
That result means "no traces", not "a camera took this".

**It says "No significant traces found". Is the file genuine?**
No conclusion follows. On public data Basira flags about half of copy-move forgeries and
under a third of uncompressed splices; most traces of splicing and in-painting are gone
after one more lossy save or a resize. A clean result proves little.

**It flagged a file I know is untouched.**
That is a false positive and worth reporting — see "Reporting a wrong result" in
[CONTRIBUTING.md](../CONTRIBUTING.md). One known cause: scenes with several identical
*printed* objects (coasters, book spines with the same label) look like copy-move clones.

**The edit was obvious before I sent the picture through a messenger. Now nothing.**
Platform re-compression and resizing destroy most traces; only geometric evidence
(copy-move) tends to survive. The robustness tables in the README show how quickly.

**Does it upload my files or download a model?**
No. There is no network code in the analysis path, no model download and no telemetry;
decoders are restricted to local files. `basira serve` listens on 127.0.0.1 only.

**Which formats?**
The test-suite exercises JPEG, PNG, TIFF, WebP and GIF images, WAV / FLAC / MP3 audio, and
H.264 video in MP4 and MOV containers. Other formats (HEIC, AAC, Opus, HEVC, ...) depend on
what Pillow, libsndfile and FFmpeg can decode on your machine and are not covered by tests;
`basira doctor` shows what your installation has. The type is decided from the file's first
bytes, not from its extension.

**Can I use the result as proof?**
No. It is a lead for a human examiner. Do not use its output as the sole basis for an
accusation.

**What does a detector such as `image.jpeg_dq` measure?**
[docs/METHODS.md](METHODS.md) describes each method, its precondition, its threshold and
its blind spots.

## Command reference

```text
basira scan PATH... [-r] [--json | --jsonl] [--csv PATH] [--html PATH] [--report-dir DIR]
                    [--no-html] [--lang {en,ar}] [-q] [--no-color] [--fail-on N] [-j N]
                    [--timeout SECONDS] [--debug]
basira doctor [--strict] [--no-color] [--debug]
basira serve [--port PORT] [--no-browser]
basira --version
```

`basira scan --help` describes each option. Options may be mixed with the paths
(`basira scan a.jpg --no-html b.jpg`). `python -m basira` is the same command.

**Docker.** The repository contains a `Dockerfile` (it has not been built or run by the
project yet — treat it as a starting point):

```bash
docker build -t basira .
docker run --rm --network none --user "$(id -u):$(id -g)" -v "$PWD:/data" basira scan photo.jpg
```

`--network none` makes "nothing leaves the machine" a property of the container;
`--user` lets the report be written into your folder with your own file ownership.
