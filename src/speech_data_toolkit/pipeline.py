"""Crash-safe processing pipeline and batch runner."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import __version__
from .audio import (
    convert_channels,
    energy_regions,
    normalize_audio,
    read_audio,
    resample_audio,
    trim_edges,
    validate_audio,
)
from .config import ToolkitConfig, load_config
from .dataset import _speaker_map, discover, safe_relative, sha256_file
from .features import extract_feature


@dataclass
class ProcessResult:
    source_path: str
    output_path: str | None = None
    output_paths: list[str] = field(default_factory=list)
    original_sample_rate: int | None = None
    output_sample_rate: int | None = None
    original_duration: float = 0.0
    output_duration: float = 0.0
    channel_count: int | None = None
    output_channel_count: int | None = None
    applied_steps: list[str] = field(default_factory=list)
    vad_segments: list[dict[str, Any]] = field(default_factory=list)
    source_intervals: list[dict[str, float]] = field(default_factory=list)
    feature_outputs: list[dict[str, Any]] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    processing_status: str = "failed"
    error_message: str | None = None
    source_sha256: str | None = None
    configuration_fingerprint: str | None = None
    cached: bool = False
    wall_time_s: float = 0.0
    output_bytes: int = 0
    speaker_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode())


def _safe_output(base: Path, relative: Path) -> Path:
    base = base.resolve()
    target = base / relative
    # Resolve existing components before creating anything. This rejects a
    # pre-existing symlinked directory that would escape the output root.
    parent = target.parent.resolve(strict=False)
    if not parent.is_relative_to(base) or target.is_symlink():
        raise ValueError(f"Output path escapes selected directory: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _atomic_audio(path: Path, data: np.ndarray, rate: int, subtype: str) -> None:
    fd, name = tempfile.mkstemp(prefix=".tmp-", suffix=".part", dir=path.parent)
    os.close(fd)
    try:
        sf.write(name, data, rate, format=path.suffix[1:].upper(), subtype=subtype)
        with open(name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _atomic_npy(path: Path, data: np.ndarray) -> None:
    fd, name = tempfile.mkstemp(prefix=".tmp-", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, data, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class AudioProcessor:
    def __init__(self, config: ToolkitConfig | None = None):
        self.config = config or ToolkitConfig()

    @classmethod
    def from_config(cls, path: str | Path) -> AudioProcessor:
        return cls(load_config(path))

    def process_file(
        self,
        source: str | Path,
        output_dir: str | Path,
        relative_path: str | Path | None = None,
        speaker_id: str | None = None,
    ) -> ProcessResult:
        start_time = time.monotonic()
        source = Path(source).resolve()
        output_dir = Path(output_dir).resolve()
        relative = safe_relative(Path(relative_path).as_posix() if relative_path else source.name)
        if output_dir == source.parent or source.is_relative_to(output_dir):
            raise ValueError("Output directory must not contain or overwrite source audio")
        result = ProcessResult(source_path=relative.as_posix(), speaker_id=speaker_id)
        cfg = self.config
        result.configuration_fingerprint = cfg.fingerprint()
        result.source_sha256 = sha256_file(source)
        output_suffix = "." + cfg.audio.output_format
        output_relative = (
            relative
            if relative.suffix.lower() == output_suffix
            else relative.with_name(relative.name + output_suffix)
        )
        cache_path = _safe_output(
            output_dir, Path(".cache") / relative.with_name(relative.name + ".json")
        )
        if cfg.execution.resume and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if (
                    cached["source_sha256"] == result.source_sha256
                    and cached["configuration_fingerprint"] == result.configuration_fingerprint
                    and cached["pipeline_version"] == __version__
                ):
                    files = cached["output_hashes"]
                    if files and all(
                        sha256_file(_safe_output(output_dir, safe_relative(name))) == digest
                        for name, digest in files.items()
                    ):
                        return ProcessResult(
                            **{
                                **cached["result"],
                                "cached": True,
                                "speaker_id": speaker_id,
                                "wall_time_s": time.monotonic() - start_time,
                            }
                        )
            except (OSError, ValueError, KeyError, TypeError):
                pass
        data, sample_rate, info = read_audio(
            source, cfg.audio.max_file_mb, cfg.validation.max_duration_s
        )
        errors, warnings = validate_audio(data, sample_rate, cfg.validation)
        if errors:
            raise ValueError("; ".join(errors))
        result.original_sample_rate = sample_rate
        result.original_duration = len(data) / sample_rate
        result.channel_count = info.channels
        result.validation_warnings = warnings
        data = convert_channels(data, cfg.audio.channels)
        if info.channels != cfg.audio.channels:
            result.applied_steps.append("channel_conversion")
        data = resample_audio(data, sample_rate, cfg.audio.sample_rate)
        rate = cfg.audio.sample_rate
        if sample_rate != rate:
            result.applied_steps.append("resample")
        offset = 0.0
        retained_end = len(data) / rate
        if cfg.silence.enabled:
            regions = energy_regions(
                data, rate, cfg.silence.top_db, cfg.silence.window_ms, cfg.silence.padding_ms
            )
            data, offset, retained_end = trim_edges(
                data, rate, regions, cfg.silence.min_retained_ms
            )
            result.applied_steps.append("silence_trim")
        segments: list[tuple[np.ndarray, str]] = []
        if cfg.vad.enabled:
            regions = energy_regions(
                data,
                rate,
                cfg.vad.top_db,
                cfg.vad.window_ms,
                cfg.vad.speech_padding_ms,
                cfg.vad.min_speech_ms,
                cfg.vad.min_silence_ms,
                relative.as_posix(),
            )
            result.vad_segments = [
                {
                    **region.as_dict(),
                    "start": region.start + offset,
                    "end": region.end + offset,
                    "backend": "energy",
                    "top_db": cfg.vad.top_db,
                    "window_ms": cfg.vad.window_ms,
                }
                for region in regions
            ]
            if not regions:
                raise ValueError("No active regions detected by energy VAD")
            if cfg.vad.mode == "trim_edges":
                result.source_intervals = [
                    {"start": regions[0].start + offset, "end": regions[-1].end + offset}
                ]
                trimmed, _, _ = trim_edges(data, rate, regions)
                segments = [(trimmed, "")]
            elif cfg.vad.mode == "split":
                result.source_intervals = [
                    {"start": r.start + offset, "end": r.end + offset} for r in regions
                ]
                segments = [
                    (data[round(r.start * rate) : round(r.end * rate)], f".segment-{i:04d}")
                    for i, r in enumerate(regions, 1)
                ]
            else:
                result.source_intervals = [
                    {"start": r.start + offset, "end": r.end + offset} for r in regions
                ]
                segments = [
                    (
                        np.concatenate(
                            [data[round(r.start * rate) : round(r.end * rate)] for r in regions]
                        ),
                        "",
                    )
                ]
            result.applied_steps.append(f"vad_{cfg.vad.mode}")
        else:
            segments = [(data, "")]
            result.source_intervals = [{"start": offset, "end": retained_end}]
        hashes: dict[str, str] = {}
        for segment, suffix in segments:
            if cfg.normalization.enabled:
                target = (
                    cfg.normalization.target_peak_dbfs
                    if cfg.normalization.method == "peak"
                    else cfg.normalization.target_rms_dbfs
                )
                segment, clipped = normalize_audio(segment, cfg.normalization.method, target)
                if clipped:
                    result.validation_warnings.append(
                        "Normalization would clip; samples were limited"
                    )
                if "normalization" not in result.applied_steps:
                    result.applied_steps.append("normalization")
            if cfg.audio.output_subtype != "FLOAT" and np.max(np.abs(segment)) > 1.0:
                segment = np.clip(segment, -1.0, 1.0)
                result.validation_warnings.append(
                    "Signal exceeded full scale after processing; PCM output was clipped"
                )
            wave_relative = output_relative.with_name(
                output_relative.stem + suffix + output_relative.suffix
            )
            wave_path = _safe_output(output_dir, wave_relative)
            _atomic_audio(wave_path, segment, rate, cfg.audio.output_subtype)
            wave_key = wave_relative.as_posix()
            hashes[wave_key] = sha256_file(wave_path)
            result.output_paths.append(wave_key)
            result.output_duration += len(segment) / rate
            if cfg.features.enabled:
                mono = segment.mean(axis=1)
                for kind in cfg.features.types:
                    feature = extract_feature(mono, rate, kind, cfg.features)
                    feature_relative = Path("features") / wave_relative.with_suffix(f".{kind}.npy")
                    feature_path = _safe_output(output_dir, feature_relative)
                    _atomic_npy(feature_path, feature)
                    meta_relative = feature_relative.with_suffix(".json")
                    meta_path = _safe_output(output_dir, meta_relative)
                    metadata = {
                        "feature_type": kind,
                        "configuration": cfg.features.model_dump(mode="json"),
                        "sample_rate": rate,
                        "shape": list(feature.shape),
                        "dtype": str(feature.dtype),
                        "source_file": wave_key,
                        "processing_version": __version__,
                    }
                    _atomic_json(meta_path, metadata)
                    hashes[feature_relative.as_posix()] = sha256_file(feature_path)
                    hashes[meta_relative.as_posix()] = sha256_file(meta_path)
                    result.feature_outputs.append(
                        {
                            "path": feature_relative.as_posix(),
                            "type": kind,
                            "shape": list(feature.shape),
                        }
                    )
        result.output_path = result.output_paths[0]
        result.output_sample_rate = rate
        result.output_channel_count = cfg.audio.channels
        result.output_bytes = sum(
            _safe_output(output_dir, safe_relative(name)).stat().st_size for name in hashes
        )
        result.processing_status = "success"
        result.wall_time_s = time.monotonic() - start_time
        _atomic_json(
            cache_path,
            {
                "source_sha256": result.source_sha256,
                "configuration_fingerprint": result.configuration_fingerprint,
                "pipeline_version": __version__,
                "output_hashes": hashes,
                "result": result.as_dict(),
            },
        )
        return result


def process_dataset(
    root: str | Path,
    output_dir: str | Path,
    cfg: ToolkitConfig,
    fail_fast: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    if output_dir == root or output_dir.is_relative_to(root) or root.is_relative_to(output_dir):
        raise ValueError("Output directory must be separate from dataset root")
    start = time.monotonic()
    summary: dict[str, Any] = {
        "total_discovered": 0,
        "successful": 0,
        "failed": 0,
        "cached": 0,
        "warning_count": 0,
        "total_input_duration_s": 0.0,
        "total_output_duration_s": 0.0,
        "output_bytes": 0,
        "feature_files": 0,
        "feature_types": {},
        "trimmed_duration_s": 0.0,
        "stopped_early": False,
    }
    if dry_run:
        for _ in discover(root):
            summary["total_discovered"] += 1
        return summary
    output_dir.mkdir(parents=True, exist_ok=True)
    speakers = _speaker_map(cfg.dataset.speaker_manifest)
    manifest_path = output_dir / "manifest.jsonl"
    fd, temp_name = tempfile.mkstemp(prefix=".manifest-", dir=output_dir)
    processor = AudioProcessor(cfg)

    def run(path: Path) -> ProcessResult:
        relative = path.relative_to(root).as_posix()
        speaker = speakers.get(relative)
        if speaker is None and cfg.dataset.speaker_directory_level is not None:
            parts = path.relative_to(root).parts[:-1]
            if cfg.dataset.speaker_directory_level < len(parts):
                speaker = parts[cfg.dataset.speaker_directory_level]
        try:
            return processor.process_file(path, output_dir, relative, speaker)
        except (OSError, ValueError, RuntimeError) as exc:
            try:
                source_hash = sha256_file(path)
            except OSError:
                source_hash = None
            return ProcessResult(
                source_path=relative,
                processing_status="failed",
                error_message=str(exc),
                configuration_fingerprint=cfg.fingerprint(),
                source_sha256=source_hash,
                speaker_id=speaker,
            )

    def record(result: ProcessResult, stream: Any) -> bool:
        summary["successful" if result.processing_status == "success" else "failed"] += 1
        summary["cached"] += int(result.cached)
        summary["warning_count"] += len(result.validation_warnings)
        summary["total_input_duration_s"] += result.original_duration
        summary["total_output_duration_s"] += result.output_duration
        summary["output_bytes"] += result.output_bytes
        summary["feature_files"] += len(result.feature_outputs)
        for feature in result.feature_outputs:
            kind = feature["type"]
            summary["feature_types"][kind] = summary["feature_types"].get(kind, 0) + 1
        if "silence_trim" in result.applied_steps or any(
            step.startswith("vad_") for step in result.applied_steps
        ):
            summary["trimmed_duration_s"] += max(
                0.0, result.original_duration - result.output_duration
            )
        stream.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")
        stream.flush()
        if result.processing_status != "success" and (
            fail_fast or not cfg.execution.continue_on_error
        ):
            summary["stopped_early"] = True
            return True
        return False

    try:
        with (
            os.fdopen(fd, "w", encoding="utf-8") as stream,
            ThreadPoolExecutor(max_workers=cfg.execution.workers) as pool,
        ):
            pending: deque[Future[ProcessResult]] = deque()
            limit = (
                1 if fail_fast or not cfg.execution.continue_on_error else cfg.execution.workers * 2
            )
            paths = discover(root)
            for path in paths:
                summary["total_discovered"] += 1
                pending.append(pool.submit(run, path))
                if len(pending) >= limit and record(pending.popleft().result(), stream):
                    break
            if not summary["stopped_early"]:
                while pending:
                    if record(pending.popleft().result(), stream):
                        break
            if summary["stopped_early"]:
                for _ in paths:
                    summary["total_discovered"] += 1
            stream.flush()
            os.fsync(stream.fileno())
        _write_manifest_csv(Path(temp_name), output_dir / "manifest.csv")
        os.replace(temp_name, manifest_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    summary["wall_time_s"] = time.monotonic() - start
    summary["throughput_audio_s_per_wall_s"] = summary["total_input_duration_s"] / max(
        summary["wall_time_s"], 1e-9
    )
    summary["duration_removed_s"] = (
        summary["total_input_duration_s"] - summary["total_output_duration_s"]
    )
    _atomic_json(output_dir / "report.json", summary)
    return summary


def _write_manifest_csv(jsonl_path: Path, csv_path: Path) -> None:
    """Write a portable CSV; list/dict fields use JSON strings."""
    fields = list(ProcessResult.__dataclass_fields__)
    fd, name = tempfile.mkstemp(prefix=".manifest-csv-", dir=csv_path.parent)
    try:
        with (
            os.fdopen(fd, "w", encoding="utf-8", newline="") as target,
            jsonl_path.open(encoding="utf-8") as source,
        ):
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            for line in source:
                record = json.loads(line)
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                        for key, value in record.items()
                    }
                )
            target.flush()
            os.fsync(target.fileno())
        os.replace(name, csv_path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
