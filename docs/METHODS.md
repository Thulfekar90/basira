# Methods

Every detector returns an `Evidence` with a raw statistic, the threshold it was compared
with, a `score` (0.5 = threshold, logistic in between) and a `confidence` describing how
applicable the method is to the file at hand. Thresholds are module-level constants next
to the code that uses them.

## Fusion

For each category (manipulation, AI generation, provenance):

    excess_i = clip((score_i - 0.5) / 0.5, 0, 1) * confidence_i
    category = 1 - prod(1 - excess_i)

Evidence marked *decisive* (a generator's own metadata) sets the category directly.
Verdict levels: 0 "No significant traces found" (< 0.20), 1 "Inconclusive" (< 0.45),
2 "Suspicious" (< 0.75), 3 "Likely manipulated / AI-generated", 4 provenance proof.
The fused localisation map averages the maps of detectors that fired, weighted by excess.

## Image

| id | Method | Precondition | Reference |
|---|---|---|---|
| `image.jpeg_dq` | Histograms of quantised DCT coefficients become periodic after a second JPEG save. Per 8×8 block, the likelihood ratio "follows the periodicity" vs "does not" is accumulated over the first 20 AC frequencies; coefficients below the pixel-rounding noise floor and the period window around zero are ignored. A compact cluster of non-conforming blocks is flagged. | JPEG saved ≥ 2 times, grid aligned | Lin, He, Tang & Tang, *Pattern Recognition* 2009 |
| `image.noise` | Immerkær noise estimate per block on non-edge pixels; expected value regressed robustly on brightness and local texture; compact cluster of residual outliers (either direction). | region ≥ a few blocks; weaker below JPEG Q80 | Immerkær, *CVIU* 1996; Mahdian & Saic 2009 |
| `image.copymove` | SIFT self-matching with the generalised 2NN test, iterative RANSAC affine grouping, then dense verification by local normalised correlation between the image and its warped self. | textured content | Amerini et al., *IEEE TIFS* 2011 |
| `image.cfa` | 2-periodic prediction error of the green channel left by demosaicing; local absence inside a demosaiced image. | uncompressed or very high quality, not resized | Popescu & Farid 2005; Ferrara et al. 2012 |
| `image.spectral` | Sharp peaks on the 1/8 frequency lattice of the noise-residual power spectrum (upsampling layers of generators). | lossless, not resized; disabled for JPEG | Zhang, Karaman & Chang 2019; Corvi et al. 2023 |
| `image.ela` | Re-save error regressed on texture; kept mainly as a familiar visual, low weight. | JPEG | Krawetz 2007 |
| `image.thumbnail` | Embedded EXIF thumbnail compared with the image (99th percentile of the smoothed difference). | thumbnail present | – |
| `image.generator_metadata` | PNG text chunks of SD WebUI / ComfyUI / InvokeAI / NovelAI, IPTC digital source type, C2PA AI assertions, generator names in XMP. | metadata not stripped | IPTC, C2PA specs |
| `image.provenance` | Editing software, XMP edit history, C2PA presence, modify ≠ capture date, camera EXIF with a libjpeg table. | – | – |

A block-artifact-grid detector was implemented and removed: it missed its designed case
and fired on clean photos; the double-quantisation map covers off-grid pastes.

## Audio

| id | Method |
|---|---|
| `audio.enf` | Mains hum (50/60 Hz or 2nd harmonic) isolated by narrow band-pass; instantaneous phase from the Hilbert transform; a cut shows as a phase step, a frequency step or a dropout. Not applicable below 10 dB hum SNR. After Rodríguez, Apolinário & Biscainho, *IEEE TIFS* 2010. |
| `audio.splice` | Noise floor and mean log-spectrum of the pauses, compared between adjacent windows; linear-prediction residual spikes in quiet context for hard-cut clicks. |
| `audio.silence` | Runs of identical samples ≥ 20 ms inside a recording that otherwise has a noise floor. |
| `audio.bandwidth`, `audio.bandwidth_shift` | Brick-wall cutoff far below Nyquist (upsampled / transcoded source); cutoff that changes within the file (sources spliced together). |
| `audio.synthetic` | Unnaturally clean pauses, flat noise floor, persistent high-frequency lines. Low weight by design. |
| `audio.metadata` | Encoder, editor and generator tags; IPTC / C2PA markers. |

## Video

| id | Method |
|---|---|
| `video.jump` | Single-frame spike in inter-frame difference (≥ 4× the local median and ≥ 2.5× both neighbours) while the histogram says the scene is unchanged. |
| `video.repeat` | Cosine similarity matrix of 16×9 frame signatures; diagonal runs ≥ 8 frames in which the content itself changes. |
| `video.freeze` | Runs of ≥ 3 identical frames inside moving footage; regular single duplicates are reported as frame-rate conversion. |
| `video.source` | Median noise level per shot against the duration-weighted median. Weak: codecs flatten the difference. |
| `video.double_compression` | Frame sizes from the MP4 `stsz` table, key frames from `stss`; residual against same-kind neighbours folded at every candidate period; one phase standing out reveals the previous encoding's GOP. Unreliable with B-frames. After Vázquez-Padín et al., *WIFS* 2012. |
| `video.frame_copymove` | The image copy-move detector on key frames; reported only if it persists across them. |
| `video.provenance`, `video.generator_metadata` | Container tags, edit lists (`elst`), transcoder and generator markers. |

The sound track is extracted and analysed by the audio engine as a child report.
