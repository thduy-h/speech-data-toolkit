"""Strict, versioned configuration and deterministic fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AudioConfig(StrictModel):
    sample_rate: int = Field(default=16000, ge=8000, le=192000)
    channels: Literal[1, 2] = 1
    output_format: Literal["wav", "flac"] = "wav"
    output_subtype: Literal["PCM_16", "PCM_24", "FLOAT"] = "PCM_16"
    max_file_mb: int = Field(default=256, gt=0, le=4096)


class NormalizationConfig(StrictModel):
    enabled: bool = True
    method: Literal["peak", "rms"] = "peak"
    target_peak_dbfs: float = Field(default=-1.0, ge=-60, le=0)
    target_rms_dbfs: float = Field(default=-20.0, ge=-60, le=-1)


class SilenceConfig(StrictModel):
    enabled: bool = False
    top_db: float = Field(default=40, ge=1, le=120)
    window_ms: int = Field(default=25, ge=1, le=1000)
    padding_ms: int = Field(default=150, ge=0, le=5000)
    min_retained_ms: int = Field(default=100, ge=0)


class VADConfig(StrictModel):
    enabled: bool = False
    backend: Literal["energy"] = "energy"
    mode: Literal["trim_edges", "split", "concat"] = "trim_edges"
    top_db: float = Field(default=35, ge=1, le=120)
    window_ms: int = Field(default=25, ge=1, le=1000)
    min_speech_ms: int = Field(default=250, ge=0)
    min_silence_ms: int = Field(default=100, ge=0)
    speech_padding_ms: int = Field(default=150, ge=0, le=5000)


def _default_feature_types() -> list[Literal["mfcc", "mel", "log_mel"]]:
    return ["mfcc", "log_mel"]


class FeatureConfig(StrictModel):
    enabled: bool = True
    types: list[Literal["mfcc", "mel", "log_mel"]] = Field(default_factory=_default_feature_types)
    n_mfcc: int = Field(default=40, ge=1, le=256)
    n_mels: int = Field(default=80, ge=2, le=512)
    n_fft: int = Field(default=400, ge=16, le=65536)
    hop_length: int = Field(default=160, ge=1)
    win_length: int = Field(default=400, ge=1)
    fmin: float = Field(default=0, ge=0)
    fmax: float | None = None
    center: bool = True
    pad_mode: Literal["constant", "reflect"] = "constant"
    mel_scale: Literal["slaney", "htk"] = "slaney"
    eps: float = Field(default=1e-10, gt=0)

    @model_validator(mode="after")
    def check_spectrum(self) -> FeatureConfig:
        if self.win_length > self.n_fft or self.n_mfcc > self.n_mels:
            raise ValueError("win_length must be <= n_fft and n_mfcc must be <= n_mels")
        if self.fmax is not None and self.fmax <= self.fmin:
            raise ValueError("fmax must be greater than fmin")
        return self


class ValidationConfig(StrictModel):
    min_sample_rate: int = Field(default=8000, ge=1)
    max_sample_rate: int = Field(default=192000, ge=1)
    min_duration_s: float = Field(default=0.05, ge=0)
    max_duration_s: float = Field(default=3600, gt=0)
    low_peak_dbfs: float = Field(default=-50, le=0)
    silence_peak_dbfs: float = Field(default=-90, le=0)
    clipping_threshold: float = Field(default=0.999, gt=0, le=1)

    @model_validator(mode="after")
    def check_thresholds(self) -> ValidationConfig:
        if self.min_sample_rate > self.max_sample_rate:
            raise ValueError("validation.min_sample_rate must be <= max_sample_rate")
        if self.min_duration_s > self.max_duration_s:
            raise ValueError("validation.min_duration_s must be <= max_duration_s")
        if self.silence_peak_dbfs > self.low_peak_dbfs:
            raise ValueError("silence_peak_dbfs must be <= low_peak_dbfs")
        return self


class DatasetConfig(StrictModel):
    speaker_directory_level: int | None = Field(default=None, ge=0)
    speaker_manifest: str | None = None


class ExecutionConfig(StrictModel):
    workers: int = Field(default=1, ge=1, le=32)
    resume: bool = True
    continue_on_error: bool = True


class SplitConfig(StrictModel):
    train: float = Field(default=0.8, ge=0, le=1)
    validation: float = Field(default=0.1, ge=0, le=1)
    test: float = Field(default=0.1, ge=0, le=1)
    seed: int = 42
    speaker_disjoint: bool = False


class ToolkitConfig(StrictModel):
    schema_version: Literal[1] = 1
    audio: AudioConfig = Field(default_factory=AudioConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    silence: SilenceConfig = Field(default_factory=SilenceConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)

    @model_validator(mode="after")
    def check_compatibility(self) -> ToolkitConfig:
        if self.audio.output_format == "flac" and self.audio.output_subtype == "FLOAT":
            raise ValueError("FLAC does not support FLOAT subtype")
        if self.features.fmax is not None and self.features.fmax > self.audio.sample_rate / 2:
            raise ValueError("features.fmax exceeds the output Nyquist frequency")
        if abs(self.split.train + self.split.validation + self.split.test - 1) > 1e-8:
            raise ValueError("split ratios must sum to 1")
        return self

    def fingerprint(self) -> str:
        from . import __version__

        payload = {"config": self.model_dump(mode="json"), "pipeline_version": __version__}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path) -> ToolkitConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return ToolkitConfig.model_validate(data)


def write_default_config(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Speech Data Toolkit, schema version 1. Unknown keys are rejected.\n"
        "# Order: decode/validate, channels, resample, silence trim, energy VAD,\n"
        "# normalization, waveform output, then feature extraction.\n"
        "# Silence trim and VAD are disabled by default to preserve speech timing.\n"
        "# Energy VAD detects signal activity, not linguistic speech.\n"
        "# Feature arrays are float32 with frequency/coefficient rows and time columns.\n"
        "# Speaker labels: set dataset.speaker_directory_level or speaker_manifest.\n"
        "# See README.md for threshold meanings, manifest fields, and limitations.\n"
    )
    path.write_text(
        header + yaml.safe_dump(ToolkitConfig().model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
