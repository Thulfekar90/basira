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
| `image.jpeg_trace` | For losslessly stored images (TIFF, PNG, BMP). The 8×8 grid of an earlier JPEG save is located from the blocking at block boundaries; per low DCT frequency the earlier quantisation step q is the largest q whose phase cos(2πX/q) has mean ≥ 0.5 over coefficients with \|X\| ≥ q/2. Each coefficient then votes with a von Mises log-likelihood "on the comb vs uniform" weighted by the comb's concentration, giving a block score from 0 (on the comb) to 1 (off it); a compact cluster scoring > 0.5 is flagged. Blocks with more than 2% colour-clipped pixels are skipped (the decoder clips saturated colours, which moves the recomputed luma off the comb). JPEG files are left to `image.jpeg_dq`. Threshold set on the CASIA v2 development half: its 3,783 JPEG originals, decoded and stored losslessly, score at most 0.13; threshold 0.35, and the verdict needs about 0.41. | was a JPEG once, not resized or filtered since; pasted content scaled, shifted off-grid or from another quality | Fan & de Queiroz, *IEEE TIP* 2003; Bianchi & Piva, *IEEE TIFS* 2012 |
| `image.copymove` | SIFT self-matching with the generalised 2NN test (images under 800 px are enlarged 2× first) and iterative RANSAC affine grouping, each affine refined by a robust Gauss-Newton fit. A group is a clone only if it passes every test that applies: enough consistent matches; *fine-detail* correlation of the denoising residual around the matched keypoints; a *slide* test (a real clone stops matching when shifted 4 px along its displacement, a straight edge keeps matching); for lossless images a *shared-noise* test in flat pixels (printed duplicates fail it); and a *lattice* test (the same displacement also present at twice its length means a regular pattern such as windows or tiles). Lossless, unscaled images also get an exact-identical-block search that finds clones of smooth areas; a match is dropped when it is not unique (the blocks also repeat at a short step, at a nearby shift or at twice the shift) or when the region's fine detail (3×3-median residual) correlates with itself at a 2–16 px step ≥ 0.7, which is periodic texture such as a generator's upsampling imprint, not a copy. Constants and the data each was measured on are commented in the code. | textured content, or an exact copy of a smooth area | Amerini et al., *IEEE TIFS* 2011 |
| `image.cfa` | Demosaicing leaves a 2-periodic prediction error on the green channel; in a single-camera image the same lattice is the acquired one everywhere. A compact cluster of blocks with the *opposite* lattice (log error ratio < −0.15) was demosaiced on a different pixel grid, i.e. comes from another image. Regions pasted with the same Bayer phase are invisible, which bounds recall near one half. | uncompressed or very high quality, not resized | Popescu & Farid 2005; Ferrara et al. 2012 |
| `image.noise` | Immerkær noise estimate per block on non-edge pixels; expected value regressed robustly on brightness and local texture; compact cluster of *abnormally smooth* blocks (in-painting, heavy retouching). A weak vote: smooth regions also occur naturally. The "noisier than expected" direction is not used. | region ≥ a few blocks | Immerkær, *CVIU* 1996; Mahdian & Saic 2009 |
| `image.spectral` | Sharp peaks on the 1/8 frequency lattice of the noise-residual power spectrum, measured in dB above an off-lattice null reference. Camera originals reach 7.6 dB on their own (demosaicing, sharpening), so the threshold is 10 dB. Not validated on real generator output. | lossless, not resized; disabled for lossy files | Zhang, Karaman & Chang 2019; Corvi et al. 2023 |
| `image.ela` | Re-save error regressed on texture; kept mainly as a familiar visual, low weight. | JPEG | Krawetz 2007 |
| `image.thumbnail` | Embedded EXIF thumbnail compared with the image (99th percentile of the smoothed difference). | thumbnail present | – |
| `image.generator_metadata` | PNG text chunks of SD WebUI / ComfyUI / InvokeAI / NovelAI validated structurally (a chunk merely *named* `parameters` is not proof), IPTC digital source type, C2PA AI assertions. Generator names in free text are hints only. | metadata not stripped | IPTC, C2PA specs |
| `image.file_prior` | Contextual hint only: no camera EXIF, a common native generator resolution, sides that are multiples of 64, lossless PNG. Confidence 0.2, so it cannot change the verdict on its own. | – | – |
| `image.provenance` | Editing software, XMP edit history, C2PA presence, modify ≠ capture date, camera EXIF with a libjpeg table. | – | – |

Removed or cut back after measurement: a block-artifact-grid detector (missed its designed
case, fired on clean photos); the "noisier than expected" half of the noise detector (fired
on 11 of 183 Columbia originals, AUC 0.55); a "CFA trace missing" test (6 false alarms on
Columbia; replaced by the phase-inversion test above).

## Audio

| id | Method |
|---|---|
| `audio.enf` | Mains hum (50/60 Hz or 2nd harmonic) isolated by narrow band-pass; instantaneous phase from the Hilbert transform. A cut shows as a phase step: a joint line + step fit over ±2.5 s, with sigma from calm windows, z ≥ 6 (largest clean value measured: 4.9). Applies only when the hum line is within 0.12 Hz of 50/60 Hz, its envelope is steady (spread ≤ 5 dB) and SNR ≥ 10 dB; when the hum is too noisy for a jump to show, the phase score is 0 and the summary says so. After Rodríguez, Apolinário & Biscainho, *IEEE TIFS* 2010. |
| `audio.splice` | Settled pauses (level within 3 dB over ±30 ms, ≥ 15 dB below speech, ≥ 0.1 s) are grouped into background states whose spectral shape and level are compared. A level change alone does not count (automatic gain control does exactly that). Fewer than 3 pauses: "not measurable", never "consistent". Confidence ≤ 0.6, because a real change of room looks the same. Linear-prediction residual spikes (≥ 10 σ) in quiet context only, for hard-cut clicks (vocal fry gives 13–21 σ inside speech). |
| `audio.silence` | Runs of identical samples ≥ 20 ms inside a recording that otherwise has a noise floor. |
| `audio.bandwidth`, `audio.bandwidth_shift` | Brick-wall cutoff far below Nyquist (upsampled / transcoded source); cutoff that changes within the file (sources spliced together; confidence 0.4, because music and sound design also change bandwidth). |
| `audio.synthetic` | Unnaturally clean pauses, flat noise floor, persistent high-frequency lines. Low weight by design. A low score is not evidence of human speech: synthetic speech played in a room or resampled scores low. |
| `audio.metadata` | Encoder, editor and generator tags; IPTC / C2PA markers. |

## Video

| id | Method |
|---|---|
| `video.jump` | Removed frames inside one shot. All thresholds are relative to the clip's own motion. Needs a single-frame spike in inter-frame difference (≥ 4× the local level, ≥ 2.5× both neighbours), a spike in the 99.5th-percentile pixel change too (rejects key-frame "pops" of low-bitrate streams), no return to the previous picture (flash), survival of brightness/contrast normalisation (exposure), and a shift-tolerant correlation ≥ 0.8 between the two frames (same scene). A hard cut between different shots is not a removal. |
| `video.repeat` | RMS distance of 32×18 frame signatures, relative to the distance between neighbouring frames (≤ 1/1.5 of it); runs ≥ 8 frames and ≥ 0.25 s, ±1-frame lag tolerance (survives frame-rate conversion). A pattern repeating at three or more lags is a loop or periodic motion and is not reported. |
| `video.freeze` | Runs of ≥ 3 identical frames (≥ 0.1 s) with motion before them, not black, followed by a jump within the same scene; regular single duplicates are reported as frame-rate conversion. |
| `video.source` | Median noise level per shot against the duration-weighted median. Informational only (confidence 0): codecs flatten the difference, and on real multi-shot footage it could not separate a foreign insert from a normal cut. |
| `video.double_compression` | Frame sizes from the MP4 `stsz` table, key frames from `stss`; residual against same-kind neighbours folded at every candidate period; one phase standing out reveals the previous encoding's GOP. Unreliable with B-frames; confidence 0.3 (0.15 with B-frames), so it cannot change the verdict on its own. Not validated on real footage. After Vázquez-Padín et al., *WIFS* 2012. |
| `video.frame_copymove` | The image copy-move detector on key frames; reported only if it persists across them. |
| `video.provenance`, `video.generator_metadata` | Container tags, edit lists (`elst`), transcoder and generator markers. |

The sound track is extracted and analysed by the audio engine as a child report.
