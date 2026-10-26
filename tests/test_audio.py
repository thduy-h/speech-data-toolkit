from __future__ import annotations

import numpy as np
import pytest

from speech_data_toolkit.audio import (
    convert_channels,
    energy_regions,
    normalize_audio,
    resample_audio,
    trim_edges,
    validate_audio,
)
from speech_data_toolkit.config import ValidationConfig


def test_resample_duration_and_frequency():
    rate = 8000
    data = np.sin(2 * np.pi * 440 * np.arange(rate) / rate).astype(np.float32)[:, None]
    result = resample_audio(data, rate, 16000)
    assert result.shape == (16000, 1)
    peak = np.argmax(np.abs(np.fft.rfft(result[:, 0])))
    assert peak == 440


def test_stereo_mean_and_silent_normalization():
    stereo = np.stack([np.ones(20), np.zeros(20)], axis=1)
    assert np.allclose(convert_channels(stereo, 1), 0.5)
    silent, clipped = normalize_audio(np.zeros((20, 1), dtype=np.float32), "peak", -1)
    assert np.count_nonzero(silent) == 0 and not clipped


def test_normalization_peak_and_rms():
    wave = np.array([[-0.2], [0.1]], dtype=np.float32)
    peak, clipped = normalize_audio(wave, "peak", -6)
    assert np.max(np.abs(peak)) == pytest.approx(10 ** (-6 / 20), abs=1e-5)
    assert not clipped
    rms, _ = normalize_audio(wave, "rms", -20)
    assert float(np.sqrt(np.mean(rms**2))) == pytest.approx(0.1, abs=1e-5)


def test_energy_regions_keep_padding_and_pause():
    rate = 8000
    wave = np.zeros((rate * 2, 1), dtype=np.float32)
    wave[rate // 2 : rate] = 0.4
    wave[rate + rate // 4 : rate + rate // 2] = 0.4
    regions = energy_regions(wave, rate, padding_ms=100, min_speech_ms=100)
    assert len(regions) == 2
    assert regions[0].start <= 0.41 and regions[0].end >= 1.09
    trimmed, start, end = trim_edges(wave, rate, regions)
    assert start == pytest.approx(regions[0].start)
    assert end == pytest.approx(regions[-1].end)
    assert len(trimmed) < len(wave)
    assert len(trimmed) > rate


def test_validation_errors_and_warnings():
    cfg = ValidationConfig()
    errors, warnings = validate_audio(np.zeros((800, 1), dtype=np.float32), 8000, cfg)
    assert errors == [] and any("silent" in w.lower() for w in warnings)
    errors, _ = validate_audio(np.full((10, 1), np.nan, dtype=np.float32), 8000, cfg)
    assert any("NaN" in e for e in errors)
    errors, warnings = validate_audio(np.ones((800, 1), dtype=np.float32), 8000, cfg)
    assert not errors and any("clipping" in w for w in warnings)


def test_decode_duration_limit_checked_before_loading(sine):
    from speech_data_toolkit.audio import read_audio

    with pytest.raises(ValueError, match="Audio too long"):
        read_audio(sine, max_duration_s=0.1)
