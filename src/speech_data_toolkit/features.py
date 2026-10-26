"""NumPy/SciPy Mel and MFCC extraction. Arrays are (frequency_or_coefficient, frames)."""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.fft import dct

from .config import FeatureConfig


def _hz_to_mel(hz: np.ndarray, scale: str) -> np.ndarray:
    if scale == "htk":
        return 2595 * np.log10(1 + hz / 700)
    linear = hz / (200 / 3)
    return np.where(hz < 1000, linear, 15 + np.log(np.maximum(hz, 1) / 1000) / (np.log(6.4) / 27))


def _mel_to_hz(mel: np.ndarray, scale: str) -> np.ndarray:
    if scale == "htk":
        return 700 * (10 ** (mel / 2595) - 1)
    return np.where(mel < 15, mel * (200 / 3), 1000 * np.exp((mel - 15) * np.log(6.4) / 27))


def mel_filterbank(sample_rate: int, cfg: FeatureConfig) -> np.ndarray:
    fmax = cfg.fmax or sample_rate / 2
    if fmax > sample_rate / 2:
        raise ValueError("fmax exceeds Nyquist frequency")
    mels = np.linspace(
        _hz_to_mel(np.array(cfg.fmin), cfg.mel_scale),
        _hz_to_mel(np.array(fmax), cfg.mel_scale),
        cfg.n_mels + 2,
    )
    hz = _mel_to_hz(mels, cfg.mel_scale)
    bins = np.linspace(0, sample_rate / 2, cfg.n_fft // 2 + 1)
    lower = (bins[None, :] - hz[:-2, None]) / np.maximum(hz[1:-1] - hz[:-2], 1e-12)[:, None]
    upper = (hz[2:, None] - bins[None, :]) / np.maximum(hz[2:] - hz[1:-1], 1e-12)[:, None]
    bank = np.maximum(0, np.minimum(lower, upper))
    if cfg.mel_scale == "slaney":
        bank *= (2 / np.maximum(hz[2:] - hz[:-2], 1e-12))[:, None]
    return cast(np.ndarray, bank.astype(np.float32))


def power_spectrogram(audio: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    if audio.ndim != 1:
        raise ValueError("Features require mono audio")
    if cfg.center:
        mode = cfg.pad_mode if len(audio) > 1 else "constant"
        audio = np.pad(audio, (cfg.n_fft // 2, cfg.n_fft // 2), mode=mode)
    if len(audio) < cfg.n_fft:
        audio = np.pad(audio, (0, cfg.n_fft - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(audio, cfg.n_fft)[:: cfg.hop_length]
    window = np.zeros(cfg.n_fft, dtype=np.float32)
    offset = (cfg.n_fft - cfg.win_length) // 2
    window[offset : offset + cfg.win_length] = np.hanning(cfg.win_length + 1)[:-1]
    spectrum = np.fft.rfft(frames * window, axis=1)
    return (np.abs(spectrum) ** 2).T.astype(np.float32)


def extract_feature(
    audio: np.ndarray, sample_rate: int, kind: str, cfg: FeatureConfig
) -> np.ndarray:
    power = power_spectrogram(audio, cfg)
    mel = mel_filterbank(sample_rate, cfg) @ power
    if kind == "mel":
        return cast(np.ndarray, mel.astype(np.float32))
    log_mel = np.log(np.maximum(mel, cfg.eps)).astype(np.float32)
    if kind == "log_mel":
        return cast(np.ndarray, log_mel)
    if kind == "mfcc":
        return cast(
            np.ndarray, dct(log_mel, axis=0, type=2, norm="ortho")[: cfg.n_mfcc].astype(np.float32)
        )
    raise ValueError(f"Unsupported feature type: {kind}")
