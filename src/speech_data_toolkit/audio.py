"""Audio I/O, signal operations, validation, and energy regions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import ValidationConfig


class AudioMetadata(Protocol):
    channels: int
    frames: int
    duration: float
    subtype: str


SUPPORTED_EXTENSIONS = {".wav", ".flac"}
DISCOVER_EXTENSIONS = SUPPORTED_EXTENSIONS | {".mp3", ".ogg", ".m4a"}


@dataclass(frozen=True)
class Region:
    start: float
    end: float
    source: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict[str, float | str]:
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "source": self.source,
        }


def read_audio(
    path: str | Path, max_file_mb: int = 256, max_duration_s: float = 3600
) -> tuple[np.ndarray, int, AudioMetadata]:
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format {p.suffix or '(none)'}: core supports WAV and FLAC")
    if p.stat().st_size > max_file_mb * 1024 * 1024:
        raise ValueError(f"Audio exceeds {max_file_mb} MB input limit")
    try:
        with sf.SoundFile(p) as stream:
            if stream.frames == 0:
                raise ValueError("Empty audio")
            if stream.frames / stream.samplerate > max_duration_s:
                raise ValueError(f"Audio too long: limit is {max_duration_s:g}s")
            data = stream.read(dtype="float32", always_2d=True)
            sample_rate = stream.samplerate
        return data, sample_rate, sf.info(p)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError(f"Failed decoding {p}: {exc}") from exc


def validate_audio(
    data: np.ndarray, sample_rate: int, cfg: ValidationConfig
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if data.ndim != 2 or data.shape[1] not in (1, 2):
        errors.append("Invalid channel count; only mono and stereo are supported")
    if data.shape[0] == 0:
        errors.append("Empty audio")
    if not cfg.min_sample_rate <= sample_rate <= cfg.max_sample_rate:
        errors.append(f"Unsupported sample rate: {sample_rate}")
    duration = data.shape[0] / sample_rate
    if duration < cfg.min_duration_s:
        errors.append(f"Audio too short: {duration:.3f}s")
    if duration > cfg.max_duration_s:
        errors.append(f"Audio too long: {duration:.3f}s")
    if not np.isfinite(data).all():
        errors.append("NaN or infinite samples")
        return errors, warnings
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak == 0 or 20 * math.log10(max(peak, 1e-30)) <= cfg.silence_peak_dbfs:
        warnings.append("Completely silent or below silence threshold")
    elif 20 * math.log10(peak) <= cfg.low_peak_dbfs:
        warnings.append("Extremely low signal amplitude")
    if peak >= cfg.clipping_threshold:
        warnings.append("Possible digital clipping")
    return errors, warnings


def convert_channels(data: np.ndarray, channels: int) -> np.ndarray:
    if data.ndim != 2 or data.shape[1] not in (1, 2) or channels not in (1, 2):
        raise ValueError("Only mono and stereo conversion are supported")
    if data.shape[1] == channels:
        return data
    if channels == 1:
        return cast(np.ndarray, data.mean(axis=1, keepdims=True))
    return np.repeat(data, 2, axis=1)


def resample_audio(data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    if original_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if original_rate == target_rate:
        return data
    divisor = math.gcd(original_rate, target_rate)
    out = resample_poly(data, target_rate // divisor, original_rate // divisor, axis=0)
    expected = round(data.shape[0] * target_rate / original_rate)
    if len(out) > expected:
        out = out[:expected]
    elif len(out) < expected:
        out = np.pad(out, ((0, expected - len(out)), (0, 0)))
    return cast(np.ndarray, out.astype(np.float32))


def normalize_audio(data: np.ndarray, method: str, target_dbfs: float) -> tuple[np.ndarray, bool]:
    magnitude = float(np.max(np.abs(data))) if data.size else 0.0
    if magnitude < 1e-8:
        return data.copy(), False
    reference = (
        magnitude if method == "peak" else float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    )
    if reference < 1e-8:
        return data.copy(), False
    result = data * (10 ** (target_dbfs / 20) / reference)
    clipped = bool(np.max(np.abs(result)) > 1.0)
    return np.clip(result, -1, 1).astype(np.float32), clipped


def energy_regions(
    data: np.ndarray,
    sample_rate: int,
    top_db: float = 40,
    window_ms: int = 25,
    padding_ms: int = 150,
    min_speech_ms: int = 0,
    min_silence_ms: int = 0,
    source: str = "",
) -> list[Region]:
    """Return padded active intervals; thresholds are relative to maximum window RMS."""
    if len(data) == 0:
        return []
    mono = np.mean(data, axis=1) if data.ndim == 2 else data
    width = max(1, round(sample_rate * window_ms / 1000))
    count = math.ceil(len(mono) / width)
    padded = np.pad(mono, (0, count * width - len(mono)))
    rms = np.sqrt(np.mean(padded.reshape(count, width).astype(np.float64) ** 2, axis=1))
    reference = float(np.max(rms))
    if reference < 1e-8:
        return []
    active = rms >= reference * 10 ** (-top_db / 20)
    # Bridge short internal gaps, then discard short bursts.
    max_gap = math.ceil(min_silence_ms * sample_rate / (1000 * width))
    if max_gap:
        index = 0
        while index < count:
            if active[index]:
                index += 1
                continue
            end = index
            while end < count and not active[end]:
                end += 1
            if index > 0 and end < count and end - index <= max_gap:
                active[index:end] = True
            index = end
    raw: list[tuple[int, int]] = []
    start: int | None = None
    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        if start is not None and (not is_active or i == count - 1):
            end = i if not is_active else i + 1
            if (end - start) * width * 1000 / sample_rate >= min_speech_ms:
                raw.append((start * width, min(end * width, len(mono))))
            start = None
    pad = round(padding_ms * sample_rate / 1000)
    merged: list[tuple[int, int]] = []
    for start_sample, end_sample in raw:
        start_sample = max(0, start_sample - pad)
        end_sample = min(len(mono), end_sample + pad)
        if merged and start_sample <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_sample))
        else:
            merged.append((start_sample, end_sample))
    return [Region(a / sample_rate, b / sample_rate, source) for a, b in merged]


def trim_edges(
    data: np.ndarray, sample_rate: int, regions: list[Region], min_retained_ms: int = 0
) -> tuple[np.ndarray, float, float]:
    if not regions:
        return data.copy(), 0.0, len(data) / sample_rate
    start = round(regions[0].start * sample_rate)
    end = round(regions[-1].end * sample_rate)
    if (end - start) * 1000 / sample_rate < min_retained_ms:
        return data.copy(), 0.0, len(data) / sample_rate
    return data[start:end], start / sample_rate, end / sample_rate
