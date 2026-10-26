"""Dataset discovery, portable manifests, and reproducible splits."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from .audio import DISCOVER_EXTENSIONS, read_audio, validate_audio
from .config import ToolkitConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("..", ".") for part in pure.parts)
        or "\\" in value
    ):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return Path(*pure.parts)


def discover(root: str | Path) -> Iterator[Path]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() or path.suffix.lower() not in DISCOVER_EXTENSIONS:
                continue
            yield path


def _speaker_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    manifest = Path(path)
    if manifest.suffix.lower() == ".csv":
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    else:
        with manifest.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    result: dict[str, str] = {}
    for row in rows:
        key = safe_relative(str(row["source_path"])).as_posix()
        result[key] = str(row["speaker_id"])
    return result


def scan_dataset(root: str | Path, cfg: ToolkitConfig | None = None) -> Iterator[dict[str, Any]]:
    root = Path(root).resolve()
    cfg = cfg or ToolkitConfig()
    speakers = _speaker_map(cfg.dataset.speaker_manifest)
    for path in discover(root):
        relative = path.relative_to(root).as_posix()
        record: dict[str, Any] = {
            "source_path": relative,
            "extension": path.suffix.lower(),
            "file_size": None,
            "source_sha256": None,
            "validation_status": "error",
            "validation_errors": [],
            "validation_warnings": [],
        }
        if relative in speakers:
            record["speaker_id"] = speakers[relative]
        elif cfg.dataset.speaker_directory_level is not None:
            parts = Path(relative).parts[:-1]
            level = cfg.dataset.speaker_directory_level
            if level < len(parts):
                record["speaker_id"] = parts[level]
        try:
            record["file_size"] = path.stat().st_size
            record["source_sha256"] = sha256_file(path)
            data, sample_rate, info = read_audio(
                path, cfg.audio.max_file_mb, cfg.validation.max_duration_s
            )
            errors, warnings = validate_audio(data, sample_rate, cfg.validation)
            record.update(
                sample_rate=sample_rate,
                channel_count=info.channels,
                frames=info.frames,
                duration=info.duration,
                subtype=info.subtype,
                validation_status="error" if errors else "warning" if warnings else "ok",
                validation_errors=errors,
                validation_warnings=warnings,
            )
        except (OSError, ValueError) as exc:
            record["validation_errors"] = [str(exc)]
        yield record


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("expected object")
                    if "source_path" in record:
                        safe_relative(str(record["source_path"]))
                    if "output_path" in record and record["output_path"]:
                        safe_relative(str(record["output_path"]))
                    yield record
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"Invalid manifest line {line_number}: {exc}") from exc


def split_manifest(records: list[dict[str, Any]], cfg: ToolkitConfig) -> list[dict[str, Any]]:
    rng = random.Random(cfg.split.seed)
    if cfg.split.speaker_disjoint:
        if any(not record.get("speaker_id") for record in records):
            raise ValueError("speaker_disjoint requires speaker_id for every record")
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record["speaker_id"])].append(record)
    else:
        groups = defaultdict(list)
        for index, record in enumerate(records):
            groups[str(index)].append(record)
    keys = sorted(groups)
    rng.shuffle(keys)
    total = len(records)
    target_train = total * cfg.split.train
    target_val = total * cfg.split.validation
    counts = {"train": 0, "validation": 0, "test": 0}
    output: list[dict[str, Any]] = []
    for key in keys:
        count = len(groups[key])
        if counts["train"] < target_train:
            split = "train"
        elif counts["validation"] < target_val:
            split = "validation"
        else:
            split = "test"
        counts[split] += count
        output.extend({**item, "split": split} for item in groups[key])
    return sorted(output, key=lambda item: str(item.get("source_path", "")))
