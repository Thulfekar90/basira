# Security

Basira opens files that somebody else made: evidence, downloads, uploads. This page says
what it defends against, which limits it enforces, what has actually been tested, and
what has not.

## Reporting a vulnerability

Please **open a private security advisory on the repository** (GitHub: *Security* →
*Advisories* → *Report a vulnerability*). Do not open a public issue for something that
can be exploited. Include the Basira version, the platform, and a file or request that
reproduces the problem. A crash, a hang, runaway memory, a file that makes Basira touch
another file or the network, or a way to obtain a clean verdict for a file that could not
be analysed are all in scope. A manipulated file that detectors merely *miss* is a
limitation of forensics (see the README), not a vulnerability.

Only the latest release is supported.

## Threat model

| Adversary | What they control | What must not happen |
|---|---|---|
| The author of a file | every byte of an image, audio or video file | code execution, reading or writing other files, network access, an unbounded hang or memory use, a traceback instead of a clear error, or a "No significant traces found" verdict for content that was never decoded |
| The author of a file **name** | the name a file is saved under (`-i.mp4`, `pipe:0.mp4`, `a%03d.png`, `` `x`.wav ``) | the name being read as an option, a URL, a pattern or shell syntax |
| A web page open in the same browser while `basira serve` runs | requests to `127.0.0.1` | uploading or reading anything (DNS rebinding, cross-site POST) |

Out of scope: other users of the same machine, anyone who can already run code as you,
and memory-safety bugs inside the native decoders (see "Not covered").

## What the code guarantees

**Nothing but real media reaches a decoder.** `basira.core.io.sniff` decides the type from
magic bytes. It refuses non-regular files (a FIFO would block forever), empty files,
playlists and reference formats (HLS, DASH, ffconcat, SDP, RTSP text, ASX/SMIL), and any
file without media magic whose first bytes are plain text.

**Images are decoded by eight Pillow decoders only** (`ALLOWED_FORMATS` in
`basira/image/load.py`: JPEG, MPO, PNG, GIF, BMP, TIFF, WebP, AVIF, plus HEIF if a plugin
is installed). Pillow would otherwise choose among ~45 formats by content; one of them
(EPS) starts a Ghostscript process.

**FFmpeg can only open the file it was given.** Both the bundled `ffmpeg` binary and the
FFmpeg inside OpenCV run with a protocol whitelist (`file`; OpenCV's also has `pipe`)
*and* a demuxer whitelist of real containers (`mov, matroska, avi, mpegts, mpeg, flv, asf, wav, w64, aiff,
flac, mp3, ogg, aac, amr, caf`). Playlist, concat, SDP/RTSP, image-sequence and device
demuxers are not on the list, so a reference format is refused by the decoder itself even
if `sniff` were bypassed. The binary is the pinned build shipped by `imageio-ffmpeg`, not
whatever `ffmpeg` is on `PATH`; it is started with an argument list (never a shell),
`-nostdin`, and the input spelled `file:<absolute path>`.

**Every refusal is a `ValueError` that names the file, never its directory.** Decoder
messages are scrubbed of paths before they are shown or stored.

**It fails closed.** A file that cannot be decoded is refused. A file that decodes only
partly is analysed, marked `truncated or damaged`, and its blank remainder is not compared
with the EXIF thumbnail. A detector that crashes turns the verdict into "Inconclusive",
never into "No significant traces found".

**The local server** binds to `127.0.0.1`, checks the `Host` header, the `Origin` header
and a per-run token, stores an upload in a private temporary directory under a sanitised
name and deletes it after the analysis.

## Resource limits

All are checked on headers, before the expensive work. Constants live in
`basira/image/load.py` and `basira/core/io.py`.

| Limit | Value | Where |
|---|---|---|
| Image file size | 300 MB | `MAX_IMAGE_BYTES` |
| Image pixels | 60 MP (`BASIRA_MAX_IMAGE_PIXELS` overrides) | `MAX_IMAGE_PIXELS` |
| Image density, only above 12 MP | 48 pixels per file byte (`BASIRA_MAX_PIXELS_PER_BYTE`, `0` disables) | `MAX_PIXELS_PER_BYTE` |
| Frames decoded from GIF / WebP / TIFF | the first one | `load_image` |
| EXIF thumbnail | JPEG only, at most 2 MP | `MAX_THUMB_PIXELS` |
| XMP packet / one text chunk inspected | 200 kB / 1 MB | `MAX_XMP_BYTES`, `MAX_TEXT_FIELD` |
| Audio sample rate | 1 000 – 768 000 Hz | `MIN_SAMPLE_RATE`, `MAX_SAMPLE_RATE` |
| Audio channels | 64 | `MAX_AUDIO_CHANNELS` |
| Decoded audio | 600 s **and** 64 million samples × channels (256 MB), whichever is less | `MAX_AUDIO_VALUES` |
| FFmpeg probe / audio decode | 30 s / 120 s, output capped with `-fs` | `FFMPEG_*_TIMEOUT_S` |
| Largest single allocation inside the FFmpeg binary | 512 MB (`-max_alloc`) | `FFMPEG_MAX_ALLOC` |
| Video frames decoded | 5 400 at 1080p, fewer for larger frames | `basira/video/temporal.py` |
| MP4 `moov` box / sample table | 256 MB / 20 million entries, bounds-checked | `basira/video/container.py` |
| Upload size (server) | 2 GB | `basira/server.py` |

The density rule is what stops decompression bombs: a flat 59 MP PNG is 183 kB (317 pixels
per byte), a flat baseline JPEG 64, a camera JPEG 2–10. It also refuses legitimate images
that are huge *and* almost empty, such as a blank bilevel scan above 12 MP; set
`BASIRA_MAX_PIXELS_PER_BYTE=0` to analyse one.

What these limits cost in memory, measured (peak resident size of one `basira.analyze`,
Apple Silicon, Python 3.9):

| Input | Peak |
|---|---|
| 12 MP image | 1.3–1.4 GB |
| 24 MP image (≥ 500 kB) | 1.8–2.2 GB |
| 48 MP image (≥ 1 MB) | 3.1–3.5 GB |
| 10 minutes of 48 kHz stereo audio | 1.3 GB |
| audio "bombs" (hours of silence, 8 channels, 655 kHz) after the cap | 0.95–1.4 GB |
| 12 frames of 16384×8640 video in a 54 kB file | 1.35 GB |

The pixel detectors need roughly 60–110 MB per megapixel, so the 60 MP ceiling means about
4–5 GB for the largest accepted image. On a small machine, or when exposing the server to
files from strangers, lower `BASIRA_MAX_IMAGE_PIXELS`.

## What has been tested

* **Mutation fuzzing** (`tests/fuzz_harness.py`): a seeded, dependency-free fuzzer over a
  locally generated corpus of 39 valid files (JPEG with EXIF thumbnail / progressive / CMYK
  / grey, PNG with text chunks / palette / alpha / 16-bit, GIF, TIFF, WebP, BMP; WAV, FLAC,
  AIFF, MP3, Ogg, Opus, M4A; MP4, MOV, MKV, AVI, WebM, fragmented MP4, MPEG-TS, with and
  without sound). Mutations: bit flips, interesting bytes and integers, truncation at
  structural boundaries, chunk / segment / box length tampering, duplicated, swapped and
  dropped segments, huge declared dimensions, sample rates and counts, splices of two
  formats, wrong magic and wrong extension. Each case runs in a worker process with a
  20 s watchdog; the oracle rejects any exception other than `ValueError`, any path in a
  message, non-finite JSON, a report that does not render, a clean verdict for an
  undecodable file, more than 1.5 GB of memory for a file under 1 MB, any child process
  other than whitelisted FFmpeg, and any socket use. Run it with
  `python tests/fuzz_harness.py --out /some/dir --cases 5000 --seed 1`.
  `tests/test_fuzz_smoke.py` runs 120 cases plus one regression test per finding on every
  CI run.
* **Decoder containment by experiment**: HLS playlists, ffconcat scripts, SDP, RTSP
  reference text, MP4 files with external data references (`dref` `url ` and `alis`),
  QuickTime reference movies and Matroska attachments, each pointed at a loopback TCP
  listener, a FIFO and a second media file; valid media under 21 hostile names; and
  images whose ICC / XMP / TIFF tags name other files and URLs. None reached a canary
  through Basira. (Before the demuxer whitelist was added, an `.m3u8` playlist handed
  straight to the guarded FFmpeg or to OpenCV did open the second local file.)
* **Resource limits**: tiny PNG / GIF / BMP / WebP / TIFF files declaring up to 2³¹×2³¹
  pixels, real flat canvases up to 59 MP, GIF / WebP / TIFF with thousands of frames,
  self-referencing and 3 000-deep IFD chains, a 200 MB zTXt chunk, 60 000 text chunks,
  deeply nested JSON in generator chunks, 90 000 unterminated XMP openers, WAV / FLAC /
  Opus / AAC files declaring hours of multichannel audio, MP4 sample tables declaring
  20 million entries.

## Not covered

* **Memory-safety bugs in the decoders** (libjpeg, libpng, libtiff, libwebp, libsndfile,
  FFmpeg, OpenCV). Basira narrows which decoders run but does not sandbox them. To analyse
  files from an attacker who may hold a decoder exploit, run Basira in a container or VM
  without network access. Keep Pillow, OpenCV and imageio-ffmpeg up to date.
* **No hard memory or CPU cap.** The limits above are checked by Basira itself; nothing
  kills an analysis that exceeds them, except `basira scan --timeout`. Video frames
  above 40 megapixels are refused; below that, a large flat video can still take minutes.
  The FFmpeg inside OpenCV has no allocation cap either (a 19 kB AVI with a tampered
  length made it allocate about 0.8 GB before giving up).
* **The server is a single-user convenience on loopback**, not a service. A client that
  stays silent for 60 s is dropped, but uploads are analysed concurrently with no queue;
  do not expose it through a proxy.
* **On Windows, a timed-out `basira scan --jobs/--timeout` worker is killed alone.** On
  POSIX each worker leads its own process group and the whole group is killed, so an
  FFmpeg it started dies with it; on Windows that FFmpeg runs on until its own time limit
  (120 s for decoding). Windows support is experimental.
* **Importing Basira restricts OpenCV's FFmpeg backend for the whole process**
  (`OPENCV_FFMPEG_CAPTURE_OPTIONS`): other code in the same process cannot open network
  streams or playlists with `cv2.VideoCapture` afterwards.
* Fuzzing was mutation-based and black-box (no coverage feedback, no sanitizers), on one
  platform (macOS arm64). The detectors' *judgement* on damaged files was not an oracle:
  a corrupted file can still be called "Suspicious".
