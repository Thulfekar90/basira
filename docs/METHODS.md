# Methods

Every detector returns an `Evidence` with a raw statistic, the threshold it was compared
with, a `score` (0.5 = threshold, logistic in between) and a `confidence` describing how
applicable the method is to the file at hand. Thresholds are module-level constants next
to the code that uses them.

## Fusion

For each category (manipulation, AI generation, provenance):

    excess_i = clip((score_i - 0.5) / 0.5, 0, 1) * confidence_i
    category = 1 - prod(1 - excess_i)

Evidence marked *decisive* (a generator's own, structurally validated metadata) sets the
category directly. The verdict follows the stronger of the manipulation and AI-generation
categories; weak hints from different categories never add up to a verdict. Levels:
0 "No significant traces found" (< 0.20), 1 "Inconclusive" (< 0.45), 2 "Suspicious"
(< 0.75), 3 "Likely manipulated / AI-generated", 4 provenance proof. If a detector
crashed or the content could not be decoded, the result is never level 0. The fused
localisation map averages the maps of detectors that fired, weighted by excess.

## Image

| id | Method | Precondition | Reference |
|---|---|---|---|
| `image.jpeg_dq` | Histograms of quantised DCT coefficients show a comb after a second JPEG save. The comb is detected per frequency by folding the histogram (divided by its one-period moving average) and requiring between-phase variance ≥ 3× within-phase variance with ≥ 30% modulation; single-compressed files give zero combs. Combs whose valleys are filled (valley/peak mass > 0.20) are rejected: they mean the image was processed between the saves, and the model below no longer holds. Each coefficient then contributes a log-likelihood ratio "single vs double", normalised by its expectation under both hypotheses over exactly the scored bins, so the block score (0 = double like the majority, 1 = single) does not depend on local texture. Coefficients below the pixel-rounding noise floor and the window around zero are ignored. A compact cluster of blocks scoring > 0.5 is flagged. | JPEG saved ≥ 2 times directly, grid aligned | Lin, He, Tang & Tang, *Pattern Recognition* 2009 |
| `image.copymove` | SIFT self-matching with the generalised 2NN test and iterative RANSAC affine grouping. A match is accepted as a clone only if the *fine detail* of the two areas also matches: correlation of the denoising residual over small discs around the matched keypoints ≥ 0.62 (≥ 0.52 with 50+ matches). Look-alike real objects (keys, windows, books of one series) match as keypoints but fail this test. The region mask comes from dense local correlation between the image and its warped self. | textured content | Amerini et al., *IEEE TIFS* 2011 |
| `image.cfa` | Demosaicing leaves a 2-periodic prediction error on the green channel; in a single-camera image the same lattice is the acquired one everywhere. A compact cluster of blocks with the *opposite* lattice (log error ratio < −0.15) was demosaiced on a different pixel grid, i.e. comes from another image. Regions pasted with the same Bayer phase are invisible, which bounds recall near one half. | uncompressed or very high quality, not resized | Popescu & Farid 2005; Ferrara et al. 2012 |
| `image.noise` | Immerkær noise estimate per block on non-edge pixels; expected value regressed robustly on brightness and local texture; compact cluster of *abnormally smooth* blocks (in-painting, heavy retouching). A weak vote: smooth regions also occur naturally. The "noisier than expected" direction is not used. | region ≥ a few blocks | Immerkær, *CVIU* 1996; Mahdian & Saic 2009 |
| `image.spectral` | Sharp peaks on the 1/8 frequency lattice of the noise-residual power spectrum, measured in dB above an off-lattice null reference. Camera originals reach 7.6 dB on their own (demosaicing, sharpening), so the threshold is 10 dB. Not validated on real generator output. | lossless, not resized; disabled for lossy files | Zhang, Karaman & Chang 2019; Corvi et al. 2023 |
| `image.ela` | Re-save error regressed on texture; kept mainly as a familiar visual, low weight. | JPEG | Krawetz 2007 |
| `image.thumbnail` | Embedded EXIF thumbnail compared with the image (99th percentile of the smoothed difference). | thumbnail present | – |
| `image.generator_metadata` | PNG text chunks of SD WebUI / ComfyUI / InvokeAI / NovelAI validated structurally (a chunk merely *named* `parameters` is not proof), IPTC digital source type, C2PA AI assertions. Generator names in free text are hints only. | metadata not stripped | IPTC, C2PA specs |
| `image.provenance` | Editing software, XMP edit history, C2PA presence, modify ≠ capture date, camera EXIF with a libjpeg table. | – | – |

Removed or cut back after measurement: a block-artifact-grid detector (missed its designed
case, fired on clean photos); the "noisier than expected" half of the noise detector (fired
on 11 of 183 Columbia originals, AUC 0.55); a "CFA trace missing" test (6 false alarms on
Columbia; replaced by the phase-inversion test above).

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
