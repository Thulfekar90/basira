<div align="center">

# Basira · بصيرة

**Explainable forensics for edited and AI-generated images, audio and video.**
It shows *where* a file was changed and *why* it thinks so — and says so plainly when it cannot tell.

[العربية](README.ar.md) · [How each detector works](docs/METHODS.md) · [Benchmarks](#measured-results) · [Publishing checklist](docs/PUBLISHING.md)

</div>

![A spliced image, the true edited area, and Basira's localisation map](docs/img/demo_splice.jpg)

```console
$ basira scan splice_2.jpg
splice_2.jpg  [image]
  Likely manipulated
  manipulation  ████████████████░░░░  80.0%
  AI generation ░░░░░░░░░░░░░░░░░░░░   0.0%
  provenance    ░░░░░░░░░░░░░░░░░░░░   0.0%
  · [JPEG double-quantisation map] The image was JPEG-compressed at least twice (8 DCT
    frequencies show double-quantisation periodicity), and a compact region (3.5% of blocks)
    does NOT share that history — consistent with content pasted in before the last save.
  report: splice_2.jpg.basira.html
```

(The image above and this output come from `python examples/make_demo.py`, which generates
forgeries with known ground truth.)

## What it is

One Python package, three engines, one report format:

| Engine | Finds | Localises |
|---|---|---|
| **Image** | splicing, copy-move cloning, smooth in-painting (generative fill), generator metadata, C2PA / IPTC AI labels, stale EXIF thumbnails, generator upsampling fingerprints in lossless files | pixel heat-map + boxes |
| **Audio** | cuts in recordings with mains hum (phase jumps), inserted segments from another room (background-noise changes in pauses), inserted digital silence, hard-cut clicks, TTS-like band-limiting and unnaturally clean pauses, generator tags | time ranges |
| **Video** | removed frames inside a continuous shot, replayed footage, frozen frames, edit lists, generator tags; plus the full audio engine on the sound track. Source changes and a previous encoding's GOP are reported as information only | time ranges |

Design choices that differ from most detectors:

- **Evidence, not a magic number.** Every detector is a documented signal-processing method that outputs its raw statistic, its threshold, a plain-language sentence and a map or time range. Nothing is a black box.
- **Calibrated to stay quiet.** A detector's score is 0.5 exactly at its decision threshold, and thresholds are set from measured clean-file distributions. Quiet detectors add nothing to the verdict, and weak hints of different kinds never add up to one. The price is recall: it misses more than it catches.
- **It never says "authentic".** The best possible result is *"No significant traces found"*, because re-compression, resizing and careful editing erase traces.
- **No model downloads, no network, no GPU.** `pip install` and run. Files never leave your machine; decoders are restricted to local files and playlist / reference formats are refused.

## Install

```bash
pip install git+https://github.com/OWNER/basira.git
```

Python 3.9+. Dependencies: NumPy, SciPy, Pillow, OpenCV (headless), soundfile, imageio-ffmpeg (bundles an FFmpeg binary, so nothing else needs installing).

## Use

```bash
basira scan photo.jpg                       # one file, summary + HTML report
basira scan evidence/ --recursive --report-dir reports/ --csv summary.csv
basira scan evidence/ -r --jsonl > results.jsonl   # one JSON object per file
basira scan evidence/ -r --jobs 4 --timeout 300    # parallel, per-file time limit
basira scan photo.jpg --lang ar             # Arabic report
basira scan evidence/ -r --fail-on 2        # exit status 2 if anything is "Suspicious" or worse
basira serve                                # local drag-and-drop UI on 127.0.0.1 (English / Arabic)
basira doctor                               # check ffmpeg, decoders and dependencies
```

See [docs/USAGE.md](docs/USAGE.md) for every option and exit status, [docs/JSON.md](docs/JSON.md) for the stable JSON format (`schema_version` 1), [CONTRIBUTING.md](CONTRIBUTING.md) to contribute, [docs/ADDING_A_DETECTOR.md](docs/ADDING_A_DETECTOR.md) to add a detector and [SECURITY.md](SECURITY.md) for the threat model.

```python
import basira

r = basira.analyze("photo.jpg")
r.verdict            # "Likely manipulated"
r.scores             # {"manipulation": 0.80, "ai_generation": 0.0, "provenance": 0.0, "overall": 0.80}
r.fused_heatmap      # HxW float32 localisation map (images, video key frames)
r.fused_segments     # suspicious time ranges (audio, video)
for e in r.evidences:
    print(e.detector, e.score, e.confidence, e.summary, e.details)
```

The HTML report is a single self-contained file (English or Arabic, no external requests, strict Content-Security-Policy): the verdict with what it does and does not mean, the localisation map with per-detector layers, opacity, a before/after slider and zoom, every detector grouped by fired / failed / quiet with what it measures and its blind spot, and one time axis with per-detector lanes for audio and video. A typical image report is 0.25–0.45 MB.

## Measured results

Every number here is produced by a script in this repository, so you can re-run it.
Nothing below is cherry-picked: the first run on public data was bad, and the fixes it
forced are listed under it.

### Public datasets

`benchmarks/public_datasets.py`, default thresholds, "flagged" = verdict "Suspicious" or worse.

| Dataset | Originals flagged | Forgeries flagged | Pixel F1 when flagged | ROC-AUC |
|---|---|---|---|---|
| **Columbia Uncompressed Splicing** (183 camera originals, 180 splices, TIFF) | **0 / 183** | 54 / 180 (30%) | 0.73 | 0.69 |
| **CoMoFoD small** (200 originals, 200 copy-move forgeries, PNG) | **0 / 200** | 159 / 200 (80%) | 0.83 | 0.90 |

CoMoFoD after JPEG re-compression of the whole image (the dataset's own `JC` variants):

| | Q100 | Q90 | Q80 | Q70 | Q60 | Q50 | Q40 | Q30 | Q20 |
|---|---|---|---|---|---|---|---|---|---|
| Forged flagged (of 200) | 78 | 129 | 91 | 69 | 58 | 51 | 37 | 22 | 6 |
| Originals flagged (of 200) | 0 | 5 | 3 | 3 | 0 | 3 | 0 | 0 | 0 |

How to read this:

- **False alarms are rare; misses are common.** That is the intended trade-off for a tool whose output may be used against someone, but it means a clean result proves little.
- **Columbia** images are uncompressed, so the JPEG-history detector cannot apply. What fires is the colour-filter-array test, which only sees splices whose two sources had opposite Bayer phase — about half of them by construction.
- **CoMoFoD**: the previous version flagged 12 originals, all from two scenes with identical *printed* objects (beer coasters, journal labels), because mass-printed copies share their fine detail just as a digital clone does. The rewritten copy-move detector makes every candidate clone pass several independent tests — keypoint geometry, fine-detail correlation, a 4-pixel "slide" test that separates a clone from a straight edge, a shared-sensor-noise test in flat areas (which printed duplicates fail), and a lattice test that rejects regular patterns such as windows and tiles — and small images are enlarged before keypoint search. Originals flagged went from 12 to 0 and forgeries found from 108 to 159; after JPEG re-compression from 55 to 14 and from 445 to 541. It is worse at the two lowest JPEG qualities (Q20: 13 → 6, Q30: 26 → 22) and on clones buried under noise added afterwards.
- **The first run on these datasets was worse**: 23 of 183 Columbia originals were flagged, and at JPEG quality 90 more than half of the CoMoFoD originals were. Fixing that meant rebuilding the copy-move verification, replacing the CFA test, cutting the noise detector in half and adding a comb-purity check to the JPEG detector (see [docs/METHODS.md](docs/METHODS.md)).
- **Caveat — these numbers are in-sample.** Several thresholds were tuned on these same two datasets; for the new copy-move tests some were tuned on odd-numbered files only, but the joint threshold, the slide test and the lattice test were set after looking at all of them. Expect worse numbers on data it has never seen, and run `benchmarks/evaluate_folder.py` on yours before trusting it.

### Real content with forgeries made by this project

`benchmarks/real_content.py` cuts real photographs and real video frames into 1024×768 photos, simulates a camera save, and builds forgeries with ground truth. Run on 34 textured crops from two photographs and one 4K video (not bundled: point the script at your own folder):

| Family | Flagged | Edit correctly located |
|---|---|---|
| Clean, one save | **0 / 34** | – |
| Clean, saved twice | **0 / 34** | – |
| Splice from another photo | 24 / 34 | 24 / 34 |
| Copy-move | 29 / 34 | 29 / 34 |
| Object removal (OpenCV in-painting) | 13 / 34 | 13 / 34 |

After the forgery is post-processed the way platforms do it:

| Flagged after… | re-save Q85 | re-save Q70 | re-save Q50 | downscale 75% | downscale 50% |
|---|---|---|---|---|---|
| Clean (false positives) | 0 / 34 | 0 / 34 | 0 / 34 | 0 / 34 | 0 / 34 |
| Splice | 11 / 34 | 0 / 34 | 0 / 34 | 2 / 34 | 0 / 34 |
| Copy-move | 28 / 34 | 24 / 34 | 16 / 34 | 23 / 34 | 22 / 34 |
| Object removal | 6 / 34 | 0 / 34 | 0 / 34 | 0 / 34 | 0 / 34 |

**Compression-history traces (splice, in-painting) are gone after one more lossy save.** Only geometric evidence (copy-move) survives. Crops without texture (blank sky, fog) are excluded: an edit there leaves nothing to measure.

### Synthetic, with a simulated camera

`benchmarks/synthetic_images.py` (Bayer sampling, sensor noise, demosaicing, JPEG), 12 images per family: clean 0 / 24 flagged; splice 12 / 12; copy-move 9 / 12; JPEG cut-out pasted off-grid 12 / 12; smooth in-painting 3 / 12; lossless images with a simulated generator upsampling imprint 12 / 12 (by the spectral fingerprint, not by copy-move).

### Audio

`benchmarks/synthetic_audio.py`, 10 recordings of 30 s (speech-like signal, room noise, weak mains hum), edits at random positions; located = flagged with a segment within 0.5 s of the true edit:

| Family | Result |
|---|---|
| Clean recording flagged | **0 / 10** |
| 0.4–1.2 s removed (mains-hum phase jump) | 8 / 10 |
| 3 s inserted from another room | 10 / 10 |
| 0.3 s of digital silence inserted | 10 / 10 |

That benchmark is optimistic. `benchmarks/audio_robustness.py` is harder: synthetic speech from the macOS voices placed in simulated rooms (impulse response, coloured noise, mains hum on every second recording, automatic gain control on every third), three cuts per recording, then delivered as WAV, MP3, AAC, Opus, Vorbis, 16 kHz, 8 kHz telephone and with added noise. Thresholds were tuned on 21 English voices; the table is the **held-out** set (20 German, French, Spanish, Italian, Portuguese and Dutch voices, other texts), run once:

| Family (per delivery format) | Result |
|---|---|
| Clean recording flagged | **0 / 12 in every format** |
| Cut, recording has mains hum | 7–10 / 18 (0 at 8 kHz) |
| Cut, no mains hum | **0 / 18** — not detectable |
| Inserted from another room | 4–5 / 12 (0 at 8 kHz) |
| Inserted digital silence | 12 / 12, but 0 / 12 after Opus or added noise |
| Switch to another voice | 0–1 / 12 |

On 60 macOS ring tones 3 are flagged (they contain produced digital silence); 0 of 30 alert and ambient sounds. A 10-minute 48 kHz file analyses in about 5 s. **No real microphone recording and no real voice-cloning output has been measured.**

### Video

`benchmarks/video_robustness.py` on 6 camera-like clips cut from two real high-resolution sources (1080p / 720p, 25 / 30 fps, with and without B-frames), each delivered in 9 ways (re-encodes, messenger-style profile, scaling), 585 files; all thresholds are relative to the clip's own motion:

| Family | Result |
|---|---|
| Clean clips flagged | **0 / 54** |
| Natural events flagged (hard cuts, similar-shot cuts, screen recordings, camera bumps, flashes, whip pans, exposure changes, dissolves, fades, pull-down) | **0 / 207** |
| 15 frames removed, located within 0.25 s | 54 / 54 |
| 5 frames removed | 45 / 54 |
| Replayed footage | 54 / 54 |
| Frozen frames | 54 / 54 |
| Footage from another camera cut in with hard cuts | **0 / 54** — two ordinary cuts look exactly like this |
| Speed change | 0 / 54 |

The thresholds were chosen on these same clips (no held-out set), and the sources are graded aerial and rendered footage, not phone video. A 60 s 1080p clip analyses in about 9 s.

## Limitations (read this)

- **"Is this AI-generated?" is the weakest question here.** Without a trained model, Basira answers it reliably only through provenance: generator metadata, C2PA Content Credentials and IPTC `trainedAlgorithmicMedia` labels (decisive when present — and trivially stripped). The pixel-level generator fingerprint works on lossless files that were not resized; it is blind on JPEGs, because JPEG's 8×8 blocking occupies the same frequencies. The audio "TTS-like" cues also fire on noise-gated and phone audio and are weighted low on purpose. A modern diffusion image re-saved by a social network will usually come back as *"No significant traces found"*.
- **Localising edits is the stronger part, and it is still modest.** On public data Basira flags most uncompressed copy-move forgeries (in-sample, see above), few after strong JPEG compression, and under a third of uncompressed splices. Each method has a precondition: the double-quantisation map needs a JPEG saved at least twice, directly, with the grid intact; the colour-filter test needs a near-lossless camera image; copy-move needs texture or an exactly copied smooth area, misses many clones after heavy JPEG compression or added noise, and can still be fooled by repeated printed objects in a JPEG.
- **Resizing, screenshots and platform re-compression destroy most traces** — see the robustness table above. An image that went through WhatsApp, X or Instagram has usually lost the splice and in-painting evidence. That is physics, not a bug.
- **Audio**: a cut is found only when the recording carries mains hum; an insertion only when its background noise differs and there are quiet pauses; a real change of room looks the same as a splice. Telephone-band audio hides almost everything. Re-recorded or resampled synthetic speech is invisible.
- **Video**: footage from another source cut in with hard cuts, and speed changes, cannot be seen — a legitimate edit looks the same. Removal is found only inside one continuous shot. The double-compression and source-noise tests are informational (they cannot change the verdict on their own); neither has been validated on real footage.
- **Two public image datasets, none for audio or video.** Audio and video were measured on simulated and derived material only (tables above). CASIA, IMD2020, NIST16, ASVspoof and FakeAVCeleb have not been run, and the thresholds were tuned on the two datasets that were. Use `benchmarks/evaluate_folder.py`; contributions with reproducible numbers are very welcome.
- A forensic tool gives leads for a human examiner. Do not use its output as the sole basis for an accusation.

## How it decides

Each detector returns `score` (0.5 = its threshold) and `confidence` (how applicable it is to *this* file — e.g. ELA on a PNG gets almost none). Only the part of a score above 0.5 counts, scaled by confidence, and detectors combine with a noisy-OR. So one strong trusted detector is enough, several moderate independent ones reinforce each other, and quiet ones contribute nothing. Generator metadata is "decisive" and short-circuits the verdict. Details: [docs/METHODS.md](docs/METHODS.md).

## Develop

```bash
git clone https://github.com/OWNER/basira.git && cd basira
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q                   # the test suite
.venv/bin/python benchmarks/synthetic_images.py # synthetic image numbers
.venv/bin/python benchmarks/real_content.py --sources ~/Pictures/some_folder --robust
.venv/bin/python examples/make_demo.py          # demo forgeries, figures and HTML reports
```

Adding a detector: write a function that returns a `basira.core.Evidence`, register it in the engine's `__init__.py`, add a synthetic forgery with ground truth and a clean-file test. A detector that cannot stay quiet on clean files does not get merged — one was removed during development for exactly that reason.

## License

MIT. See [LICENSE](LICENSE).
