"""D003 regression contracts for applicability, missing content and frame failures."""
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from basira.audio import analyze_audio
from basira.audio._dsp import to_mono
from basira.core.fusion import finalize
from basira.core.types import Category, Evidence, Report
from basira.image.jpeg_trace import analyze_jpeg_trace
from basira.image.load import ImageData, load_image
from basira.video import _keyframe_evidence


@pytest.mark.parametrize('fmt,kwargs,lossless', [
    ('PNG', {}, True), ('BMP', {}, True),
    ('TIFF', {'compression': 'raw'}, True),
    ('TIFF', {'compression': 'tiff_lzw'}, True),
    ('TIFF', {'compression': 'tiff_adobe_deflate'}, True),
    ('TIFF', {'compression': 'jpeg'}, False),
    ('WEBP', {'lossless': True}, True),
    ('WEBP', {'lossless': False, 'quality': 95}, False),
])
def test_loaded_format_losslessness_controls_trace(tmp_path, monkeypatch, fmt, kwargs, lossless):
    rgb = np.random.default_rng(51).integers(0, 256, (80, 80, 3), dtype=np.uint8)
    p = tmp_path / 'sample'
    Image.fromarray(rgb).save(p, format=fmt, **kwargs)
    d = load_image(str(p))
    assert d.lossless is lossless
    # Unsupported encodings must abstain before any DCT/noise statistic is calculated.
    if not lossless:
        def forbidden(*args):
            pytest.fail('lossy image entered JPEG trace estimator')
        monkeypatch.setattr('basira.image.jpeg_trace._y_plane', forbidden)
        e = analyze_jpeg_trace(d)
        assert e.score == e.confidence == 0
        assert e.summary.startswith('Not applicable:') and e.error is None


@pytest.mark.parametrize('fmt,info', [('TIFF', {'compression': 'unknown'}), ('WEBP', {}),
                                     ('AVIF', {}), ('HEIF', {}), ('GIF', {}), ('RAW', {})])
def test_trace_abstains_when_lossless_applicability_unknown(monkeypatch, fmt, info):
    d = ImageData('unknown', b'', fmt, np.zeros((80, 80, 3), np.uint8), info)
    def forbidden(*args):
        pytest.fail('unsupported format entered JPEG trace estimator')
    monkeypatch.setattr('basira.image.jpeg_trace._y_plane', forbidden)
    e = analyze_jpeg_trace(d)
    assert e.score == e.confidence == 0 and e.error is None


@pytest.mark.parametrize('shape', [(0,), (0, 1), (0, 2), (2, 0)])
def test_direct_empty_audio_is_not_applicable(shape):
    samples = np.empty(shape, np.float32)
    assert to_mono(samples).shape == (0,)
    report = analyze_audio('', samples=samples, sr=16000)
    assert report.info['frames'] == 0
    assert report.info['content_empty'] is True
    assert report.evidences and all(e.confidence == 0 and e.error is None for e in report.evidences)
    assert all(e.summary.startswith('Not applicable:') for e in report.evidences)


@pytest.mark.parametrize('failed_frames,raises', [([0, 1], False), ([1], False), ([1], True), ([0], True)])
def test_keyframe_failures_are_preserved_in_report(monkeypatch, failed_frames, raises):
    rgb = np.zeros((24, 24, 3), np.uint8)
    fs = SimpleNamespace(keyframes=[(i, rgb) for i in (0, 1)], time_of=lambda i: i / 30)
    def analyze_frame(rgb, name, detectors):
        fi = int(name.split()[-1])
        if fi in failed_frames:
            if raises:
                raise RuntimeError('keyframe probe failure')
            return [Evidence.failed('image.copymove', 'Copy move', Category.MANIPULATION,
                                    RuntimeError('keyframe probe failure'))]
        return [Evidence('image.copymove', 'Copy move', Category.MANIPULATION,
                         0.1, 0.8, 'No clone found')]
    monkeypatch.setattr('basira.image.analyze_frame', analyze_frame)
    evs = _keyframe_evidence(fs)
    failed = [e for e in evs if e.error]
    assert len(failed) == 1
    failure = failed[0]
    assert failure.detector == 'video.frame_copymove.failures'
    assert failure.confidence == failure.score == 0
    assert failure.details['frames_attempted'] == 2
    assert failure.details['frames_failed'] == len(failed_frames)
    assert set(failure.details['failed_frames']) == {str(i) for i in failed_frames}
    report = finalize(Report('fixture', 'video', {}, evidences=evs))
    assert report.info['detectors_failed'] == 1 and report.verdict_level >= 1
    assert 'No significant traces' not in report.verdict
    assert 'keyframe probe failure' in str(report.to_dict())
    if len(failed_frames) == 1:
        good = next(e for e in evs if not e.error)
        assert good.detector == 'video.frame_copymove'
        assert good.details['per_frame'] == {str(1 - failed_frames[0]): 0.1}
        assert good.confidence == pytest.approx(0.8 * 0.7)


def test_webp_losslessness_is_container_structure_not_metadata(tmp_path):
    rgb = np.random.default_rng(52).integers(0, 256, (80, 80, 3), dtype=np.uint8)
    for lossless in (False, True):
        p = tmp_path / ('with_metadata_%s.webp' % lossless)
        Image.fromarray(rgb).save(p, format='WEBP', lossless=lossless,
                                 exif=b'Exif\0\0VP8L\x05\0\0\0\x2f\0\0\0\0\0')
        assert load_image(str(p)).lossless is lossless


@pytest.mark.parametrize('damage', ['riff_size', 'chunk_size', 'truncated', 'duplicate', 'version'])
def test_webp_parser_abstains_on_malformed_structure(tmp_path, damage):
    from basira.image.load import _webp_lossless
    p = tmp_path / 'lossless.webp'
    Image.new('RGB', (80, 80), (1, 2, 3)).save(p, format='WEBP', lossless=True)
    raw = bytearray(p.read_bytes())
    assert _webp_lossless(bytes(raw))
    assert raw[12:16] == b'VP8L'
    if damage == 'riff_size':
        raw[4:8] = (len(raw) + 99).to_bytes(4, 'little')
    elif damage == 'chunk_size':
        raw[16:20] = (2**32 - 1).to_bytes(4, 'little')
    elif damage == 'truncated':
        raw = raw[:-1]
    elif damage == 'duplicate':
        raw += raw[12:]
        raw[4:8] = (len(raw) - 8).to_bytes(4, 'little')
    else:
        raw[24] |= 0xe0
    assert not _webp_lossless(bytes(raw))


def test_animated_webp_is_not_assumed_lossless(tmp_path):
    p = tmp_path / 'animated.webp'
    Image.new('RGB', (80, 80), (1, 2, 3)).save(
        p, format='WEBP', lossless=True, save_all=True,
        append_images=[Image.new('RGB', (80, 80), (4, 5, 6))], duration=100)
    assert not load_image(str(p)).lossless
