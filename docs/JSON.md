# JSON output

`basira scan --json`, `basira scan --jsonl` and `Report.to_dict()` in Python produce the
same structure. This page documents every field and says which of them you can build on.

- [Getting it](#getting-it)
- [The report object](#the-report-object)
- [Evidence objects](#evidence-objects)
- [Segments and regions](#segments-and-regions)
- [Error records](#error-records)
- [What is stable and what is not](#what-is-stable-and-what-is-not)
- [A real example](#a-real-example)

## Getting it

| Command | stdout |
|---|---|
| `basira scan FILE --json` | one report object |
| `basira scan A B ... --json`, `basira scan FOLDER --json`, a pattern | an array with one entry per input, in input order |
| `basira scan ... --jsonl` | one object per line, written as each file finishes (with `--jobs` that is not input order; use `path` to match) |

Nothing else is written to stdout; messages and progress go to stderr. The output is
UTF-8 and strict JSON: it contains no `NaN` or `Infinity`
(`json.dumps(report.to_dict(), allow_nan=False)` always succeeds), and file names are
written as they are, not escaped to ASCII. Add `--no-html` if you do not want the HTML
report as well.

## The report object

| Field | Type | Meaning |
|---|---|---|
| `tool` | string | always `"basira"` |
| `schema_version` | integer | version of this layout, currently `1` (see [stability](#what-is-stable-and-what-is-not)) |
| `version` | string | version of Basira that produced the report, e.g. `"0.1.0"` |
| `path` | string | the path as it was given on the command line or to `basira.analyze` |
| `media_type` | string | `"image"`, `"audio"` or `"video"`, decided from the file's content |
| `info` | object | facts about the file (below) |
| `verdict` | string | the sentence shown to people, e.g. `"No significant traces found"` |
| `verdict_level` | integer 0–4 | the verdict as a number (below); use this in programs |
| `scores` | object | `manipulation`, `ai_generation`, `provenance`, `overall`: numbers from 0 to 1 |
| `explanation` | array of strings | the findings that drove the verdict, strongest first, each prefixed with `[detector title]`; or one sentence saying that nothing exceeded its threshold |
| `segments` | array of [segment](#segments-and-regions) | suspicious time ranges, merged across detectors (audio and video; empty for images) |
| `evidences` | array of [evidence](#evidence-objects) | one entry per detector that was run, including the quiet ones |
| `children` | array of report objects | sub-reports with this same layout; today, the sound track of a video |
| `elapsed_sec` | number | wall-clock seconds the analysis took |

**`verdict_level`**

| Level | Verdict text | Meaning |
|---|---|---|
| 0 | No significant traces found | no detector exceeded its threshold. **Not** a statement that the file is authentic |
| 1 | Inconclusive (...) / No content to analyse | weak anomalies, incomplete analysis (see `info.detectors_failed`), or an empty audio input (`info.content_empty`) |
| 2 | Suspicious | the strongest category score is at least 0.45 |
| 3 | Likely manipulated / Likely AI-generated / Metadata declares AI generation (unverified) | the strongest category score is at least 0.75; strong unsigned generator metadata is a lead, not proof |
| 4 | Reserved for verified provenance | requires cryptographic signature verification and a trust policy; not implemented, so current engines do not produce this level |

**`scores`**. `manipulation` (local edits: splicing, cloning, cuts), `ai_generation`
(synthetic content) and `provenance` (metadata) are each fused from the detectors of that
category: only the part of a detector's score above 0.5 counts, weighted by its
confidence. `overall` combines the three but does not drive the verdict; the verdict
follows the stronger of `manipulation` and `ai_generation`. These are evidence scores,
not calibrated probabilities. Unsigned generator claims use the same fusion; the
legacy `decisive` flag neither sets a category directly nor enables level 4.

When a strong unsigned generator claim accompanies an AI-driven level-3 result, the
English verdict is `Metadata declares AI generation (unverified)`. The Arabic HTML
report displays «البيانات الوصفية تصرّح بأنه مولّد بالذكاء الاصطناعي (غير متحقق منه)».
JSON verdict text remains English; do not parse its wording as a stable identifier.

**`info`**. Two keys are always present and are part of the stable layout:

| Key | Type | Meaning |
|---|---|---|
| `detectors_failed` | integer | how many detectors raised an error on this file. If it is not 0, a level-0 result would have been unreliable, so the verdict is at least level 1 |
| `children_failed` | integer | how many child reports (sound track) had a failure of their own |

The other keys describe the file and depend on the media type and on the Basira version.
They are informational: read them defensively (`info.get(...)`).

| Media type | Keys in 0.1.0 |
|---|---|
| image | `format`, `width`, `height`, `jpeg_quality` (estimate, or null), `jpeg_standard_table`, `subsampling`, `has_alpha`, `bytes`, `notes` (only when the decoder had something to say, e.g. a truncated file) |
| audio | `duration` (s), `sample_rate`, `channels`, `frames`, `source`, `format`, `format_info`, `subtype`, `subtype_info`, `lossy`, `peak_dbfs`, `rms_dbfs`, `clipped_fraction`, `file_size`; `label` on a video's sound track; `content_empty: true` when no samples are available to analyse |
| video | `bytes`, `encoder`, `container_codec`, `streams`, `width`, `height`, `fps`, `frames`, `duration` (s), `codec`, `analysed_every_nth_frame`, `analysed_seconds`, `truncated` |

## Evidence objects

One per detector. A detector that had nothing to say is still listed, with a low score.

| Field | Type | Meaning |
|---|---|---|
| `detector` | string | stable id, e.g. `"image.jpeg_dq"`, `"audio.enf"`, `"video.freeze"` |
| `title` | string | human-readable name |
| `category` | string | `"manipulation"`, `"ai_generation"` or `"provenance"` |
| `score` | number 0–1 | how anomalous the file is for this detector; **0.5 is the detector's own decision threshold** |
| `confidence` | number 0–1 | how far this detector can be trusted on *this* file. 0 means it did not apply (wrong format, too small, no signal) |
| `summary` | string | one or two plain-language sentences |
| `decisive` | boolean | retained for schema compatibility; unsigned metadata is false, and this field alone never proves provenance or enables level 4 |
| `details` | object | the raw measurements and thresholds behind the score. Keys differ per detector and per version |
| `segments` | array of [segment](#segments-and-regions) | time ranges this detector flagged |
| `regions` | array of [region](#segments-and-regions) | boxes this detector flagged |
| `has_heatmap` | boolean | whether a pixel map exists. The map itself is not in the JSON; it is in the HTML report and in `Evidence.heatmap` in Python |
| `error` | string or null | set when the detector crashed on this file; `score` and `confidence` are then 0 |

"Fired" means `score >= 0.5` with `confidence > 0` and `error == null`.

Strong unsigned generator declarations carry `details.unverified_generator_claim: true`.
That is an informational flag, not evidence that the author or signature was verified.
No signature-verification result is currently implemented.

A failed key-frame detector remains visible as a separate error evidence item, currently
`video.frame_copymove.failures`, with zero score and confidence. Its details include
`frames_attempted`, `frames_failed`, `frames_succeeded` and `failed_frames`. Successful
frames can still contribute their own aggregate evidence; a failure increments
`info.detectors_failed` and prevents a silently clean level-0 verdict. The counter counts
failed evidence items, not the number of failed frames.

An empty decoded audio input returns a not-applicable evidence item and the level-1
verdict `No content to analyse`, with `info.content_empty: true`. This also applies to
the direct audio API supplied with an empty sample array. It is distinct from a zero-byte
file rejected by input sniffing, and from a nonempty recording of silence.

Inside `details`, values are plain JSON: numbers are rounded to 6 decimals, a value that
was not finite is `null`, arrays longer than 256 elements are replaced by a string such as
`"<array (64, 64)>"`, and byte strings are cut to 64 bytes.

## Segments and regions

A **segment** is a time range in seconds:

| Field | Type | Meaning |
|---|---|---|
| `start`, `end` | number | seconds from the start of the file |
| `score` | number 0–1 | strength |
| `label` | string | what was found there |

A **region** is an axis-aligned box in pixels of the image as stored (EXIF rotation is not
applied):

| Field | Type | Meaning |
|---|---|---|
| `x`, `y` | integer | top-left corner |
| `w`, `h` | integer | width and height |
| `score` | number 0–1 | strength |
| `label` | string | what was found there |
| `frame` | integer | video frame index; present only for regions on video frames |

## Error records

When an input cannot be analysed, `--json` and `--jsonl` contain an error record in its
place, so there is always one entry per input. The exit status is then 1.

```json
{
  "tool": "basira",
  "schema_version": 1,
  "version": "0.1.0",
  "path": "missing.jpg",
  "error": {
    "kind": "not_found",
    "message": "no such file: missing.jpg"
  }
}
```

A record is an error record if and only if it has a top-level `error` key; report
objects never have one. `error.kind` is one of:

| `kind` | Meaning |
|---|---|
| `not_found` | the path does not exist, or a pattern matched nothing |
| `unreadable` | the file or folder could not be opened |
| `unsupported` | not an image, audio or video file Basira reads (also: empty file, playlist) |
| `too_large` | over a built-in size limit, or not enough memory |
| `failed` | recognised, but decoding or analysis failed |
| `timeout` | `--timeout` expired |
| `crashed` | the worker process died (`--jobs` / `--timeout` runs) |

Files that a folder scan *skips* because they are not media are not in the output at all.

## What is stable and what is not

`schema_version` is `1`. It will be raised only when something in the first list changes
incompatibly; additions do not raise it, so **ignore keys you do not know**.

**Covered — will not change without a new `schema_version`:**

- the names and types of the top-level fields, of the evidence fields, and of the segment
  and region fields listed above;
- the score keys `manipulation`, `ai_generation`, `provenance`, `overall`, each a number
  from 0 to 1;
- the meaning of `verdict_level` 0–4 as an ordering (higher is stronger), and that level 0
  is the only level meaning "nothing found";
- that 0.5 is every detector's decision threshold;
- the values of `media_type` and `category`;
- `info.detectors_failed` and `info.children_failed`;
- the error record and its `kind` values (new kinds may be added);
- one object for a single file with `--json`, an array otherwise, one object per line
  with `--jsonl`;
- strict JSON: no `NaN`, no `Infinity`.

**Not covered — expect these to change between versions:**

- **the numbers themselves.** Scores, and the thresholds on the category scores that
  separate the levels, move whenever a detector is recalibrated. The same file can get a
  different score, and occasionally a different level, from a different Basira version.
  Record `version` next to any result you keep;
- the wording of `verdict`, `summary`, `explanation`, `title` and `label` — never parse
  these; use `verdict_level`, `detector`, `score` and `category`;
- the keys and values inside `details`, and the keys of `info` other than the two named
  above;
- which detectors exist and the order of `evidences`. Detector ids are kept once
  released, but detectors can be added, and one that cannot stay quiet on clean files will
  be removed (the changelog will say so);
- `elapsed_sec`, and the order of lines in `--jsonl` output when `--jobs` is used;
- the HTML report and the terminal text. (For `--csv`, read columns by their header name:
  new columns may be added.)

Until version 1.0 this promise is an intention backed by a test
(`tests/test_cli_ux.py` checks the field names against this page), not by a track record.

## A real example

Unedited output of the command below on a forgery with known ground truth, which
`python examples/make_demo.py` generates. `path` and `elapsed_sec` will differ on your
machine; scores can differ slightly across platforms and Basira versions.

<!-- example:begin -->
```console
$ basira scan examples/out/splice_2.jpg --json --no-html
```

<details open>
<summary>Output (321 lines)</summary>

```json
{
  "tool": "basira",
  "schema_version": 1,
  "version": "0.1.0",
  "path": "examples/out/splice_2.jpg",
  "media_type": "image",
  "info": {
    "format": "JPEG",
    "width": 1024,
    "height": 768,
    "jpeg_quality": 92,
    "jpeg_standard_table": true,
    "subsampling": "4:2:0",
    "has_alpha": false,
    "bytes": 140733,
    "children_failed": 0,
    "detectors_failed": 0
  },
  "verdict": "Likely manipulated",
  "verdict_level": 3,
  "scores": {
    "manipulation": 0.8,
    "ai_generation": 0.0,
    "provenance": 0.0,
    "overall": 0.8
  },
  "explanation": [
    "[JPEG double-quantisation map] The image was JPEG-compressed at least twice (8 DCT frequencies show double-quantisation periodicity), and a compact region (3.5% of blocks) does NOT share that history — consistent with content pasted in before the last save."
  ],
  "segments": [],
  "evidences": [
    {
      "detector": "image.generator_metadata",
      "title": "Generator metadata",
      "category": "ai_generation",
      "score": 0.0,
      "confidence": 0.3,
      "summary": "No AI-generator metadata found. Metadata is easily stripped, so its absence does not prove anything.",
      "decisive": false,
      "details": {
        "iptc_source_types": []
      },
      "segments": [],
      "regions": [],
      "has_heatmap": false,
      "error": null
    },
    {
      "detector": "image.provenance",
      "title": "Metadata & provenance",
      "category": "provenance",
      "score": 0.0,
      "confidence": 0.3,
      "summary": "No editing-software traces in metadata (metadata may have been stripped).",
      "decisive": false,
      "details": {
        "format": "JPEG",
        "software": "",
        "creator_tool": "",
        "make": "",
        "model": "",
        "datetime": null,
        "datetime_original": null,
        "jpeg_quality_estimate": 92,
        "jpeg_standard_table": true,
        "subsampling": "4:2:0",
        "c2pa": false,
        "transcoders": [],
        "exif_tags": 0,
        "xmp_bytes": 0,
        "signatures": []
      },
      "segments": [],
      "regions": [],
      "has_heatmap": false,
      "error": null
    },
    {
      "detector": "image.file_prior",
      "title": "File-shape prior",
      "category": "ai_generation",
      "score": 0.4999,
      "confidence": 0.2,
      "summary": "Weak contextual hint: no camera EXIF, 1024x768 are multiples of 64.",
      "decisive": false,
      "details": {
        "width": 1024,
        "height": 768,
        "no_camera_exif": true,
        "generator_resolution": false,
        "multiple_of_64": true
      },
      "segments": [],
      "regions": [],
      "has_heatmap": false,
      "error": null
    },
    {
      "detector": "image.jpeg_dq",
      "title": "JPEG double-quantisation map",
      "category": "manipulation",
      "score": 1.0,
      "confidence": 0.8,
      "summary": "The image was JPEG-compressed at least twice (8 DCT frequencies show double-quantisation periodicity), and a compact region (3.5% of blocks) does NOT share that history — consistent with content pasted in before the last save.",
      "decisive": false,
      "details": {
        "periodic_frequencies": 8,
        "smeared_combs": 0,
        "periods": [
          {
            "freq": [
              0,
              1
            ],
            "q2": 2.0,
            "period": 3,
            "strength": 838.888
          },
          {
            "freq": [
              1,
              0
            ],
            "q2": 2.0,
            "period": 3,
            "strength": 775.911
          },
          {
            "freq": [
              2,
              0
            ],
            "q2": 2.0,
            "period": 7,
            "strength": 8.141
          },
          {
            "freq": [
              1,
              1
            ],
            "q2": 2.0,
            "period": 3,
            "strength": 1987.827
          },
          {
            "freq": [
              0,
              2
            ],
            "q2": 2.0,
            "period": 5,
            "strength": 6.661
          },
          {
            "freq": [
              1,
              2
            ],
            "q2": 2.0,
            "period": 7,
            "strength": 4.293
          },
          {
            "freq": [
              2,
              1
            ],
            "q2": 2.0,
            "period": 7,
            "strength": 4.587
          },
          {
            "freq": [
              3,
              1
            ],
            "q2": 3.0,
            "period": 3,
            "strength": 1059.883
          }
        ],
        "last_quality": 92,
        "statistic": 0.344254,
        "threshold": 0.06,
        "frac": 0.034993,
        "mean_z": 4.600714,
        "n_blocks": 430
      },
      "segments": [],
      "regions": [
        {
          "x": 696,
          "y": 552,
          "w": 168,
          "h": 168,
          "score": 1.0,
          "label": "single-compressed block cluster"
        }
      ],
      "has_heatmap": true,
      "error": null
    },
    {
      "detector": "image.ela",
      "title": "Error level analysis (texture-normalised)",
      "category": "manipulation",
      "score": 0.0033,
      "confidence": 0.25,
      "summary": "Compression error is consistent with texture everywhere (statistic 0.108 < 0.45).",
      "decisive": false,
      "details": {
        "resave_quality": 92,
        "statistic": 0.108253,
        "threshold": 0.45,
        "frac": 0.001302,
        "mean_z": 9.757545,
        "n_blocks": 4
      },
      "segments": [],
      "regions": [],
      "has_heatmap": true,
      "error": null
    },
    {
      "detector": "image.noise",
      "title": "Sensor-noise consistency",
      "category": "manipulation",
      "score": 0.0006,
      "confidence": 0.5,
      "summary": "Noise level is consistent across the image (statistic 0.066 < 0.29).",
      "decisive": false,
      "details": {
        "block": 32,
        "statistic": 0.066226,
        "threshold": 0.29,
        "direction": "smoother",
        "median_sigma": 0.45015,
        "region_sigma": 0.069867,
        "frac": 0.005208,
        "mean_z": 2.752951,
        "n_blocks": 4
      },
      "segments": [],
      "regions": [],
      "has_heatmap": true,
      "error": null
    },
    {
      "detector": "image.cfa",
      "title": "Camera demosaicing (CFA) consistency",
      "category": "manipulation",
      "score": 0.0,
      "confidence": 0.05,
      "summary": "No global demosaicing pattern (ratio 1.003); the image may have been resized, rendered or generated rather than captured. Local CFA consistency cannot be checked.",
      "decisive": false,
      "details": {
        "global_ratio": 1.003447,
        "threshold": 1.06
      },
      "segments": [],
      "regions": [],
      "has_heatmap": false,
      "error": null
    },
    {
      "detector": "image.copymove",
      "title": "Copy-move (cloning)",
      "category": "manipulation",
      "score": 0.0008,
      "confidence": 0.7,
      "summary": "No cloned regions found (0 self-matching keypoint pairs).",
      "decisive": false,
      "details": {
        "keypoints": 7388,
        "candidate_pairs": 0
      },
      "segments": [],
      "regions": [],
      "has_heatmap": false,
      "error": null
    },
    {
      "detector": "image.spectral",
      "title": "Generator fingerprint (residual spectrum)",
      "category": "ai_generation",
      "score": 0.0013,
      "confidence": 0.1,
      "summary": "JPEG blocking occupies the same frequency lattice, so generator periodicity cannot be measured reliably on this file.",
      "decisive": false,
      "details": {
        "lattice_excess_db": 0.0,
        "strong_peaks": 0,
        "threshold_db": 10.0,
        "offlattice_p99.97_db": 4.07917,
        "jpeg_lattice_db": 1.766392,
        "cfa_ratio": 1.003447,
        "peaks": {
          "-3,0": 3.94,
          "3,0": 3.94,
          "-4,0": 2.93,
          "1,-4": 2.62,
          "-1,-3": 2.36,
          "1,3": 2.36,
          "1,-1": 2.31,
          "-1,1": 2.31,
          "3,-3": 2.27,
          "-3,3": 2.27,
          "2,-2": 2.25,
          "-2,2": 2.25
        }
      },
      "segments": [],
      "regions": [],
      "has_heatmap": false,
      "error": null
    }
  ],
  "children": [],
  "elapsed_sec": 1.977
}
```

</details>
<!-- example:end -->
