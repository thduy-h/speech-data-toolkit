from __future__ import annotations

import numpy as np

from speech_data_toolkit.config import FeatureConfig
from speech_data_toolkit.features import extract_feature, mel_filterbank


def test_feature_shapes_finite_and_silence():
    cfg = FeatureConfig()
    signal = np.zeros(1600, dtype=np.float32)
    mel = extract_feature(signal, 16000, "mel", cfg)
    log_mel = extract_feature(signal, 16000, "log_mel", cfg)
    mfcc = extract_feature(signal, 16000, "mfcc", cfg)
    assert mel.shape == (80, 11)
    assert log_mel.shape == (80, 11)
    assert mfcc.shape == (40, 11)
    assert np.isfinite(log_mel).all() and np.isfinite(mfcc).all()
    assert np.all(mel == 0)
    assert np.allclose(log_mel, np.log(cfg.eps))
    assert np.allclose(mfcc[0], np.sqrt(cfg.n_mels) * np.log(cfg.eps), atol=1e-4)
    assert np.allclose(mfcc[1:], 0, atol=1e-4)


def test_mel_filters_nonnegative_and_frequency_localized():
    cfg = FeatureConfig(n_mels=20, n_mfcc=10)
    bank = mel_filterbank(16000, cfg)
    assert bank.shape == (20, 201)
    assert np.all(bank >= 0)
    centers = np.argmax(bank, axis=1)
    assert np.all(np.diff(centers) >= 0)


def test_centering_and_reflect_padding():
    audio = np.arange(64, dtype=np.float32)
    cfg = FeatureConfig(n_fft=32, win_length=32, hop_length=16, n_mels=8, n_mfcc=4)
    centered = extract_feature(audio, 8000, "mel", cfg)
    cfg.center = False
    uncentered = extract_feature(audio, 8000, "mel", cfg)
    assert centered.shape[1] > uncentered.shape[1]
    cfg.center = True
    cfg.pad_mode = "reflect"
    reflected = extract_feature(audio, 8000, "mel", cfg)
    assert reflected.shape == centered.shape
    assert not np.allclose(reflected, centered)
