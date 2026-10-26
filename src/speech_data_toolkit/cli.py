"""speechprep command line interface."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from .audio import read_audio
from .config import ToolkitConfig, load_config, write_default_config
from .dataset import discover, read_jsonl, scan_dataset, split_manifest
from .features import extract_feature
from .pipeline import _atomic_bytes, _atomic_json, _atomic_npy, _safe_output, process_dataset

app = typer.Typer(help="Reproducible speech audio preparation toolkit", no_args_is_help=True)
console = Console()


def _config(path: Path | None) -> ToolkitConfig:
    try:
        return load_config(path) if path else ToolkitConfig()
    except (OSError, ValueError) as exc:
        console.print(f"[red]Invalid configuration:[/red] {exc}")
        raise typer.Exit(2) from exc


def _fail(exc: Exception) -> None:
    console.print(f"[red]Error:[/red] {exc}")
    raise typer.Exit(2) from exc


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    _atomic_bytes(path, payload.encode("utf-8"))


@app.command()
def init(
    output: Annotated[Path, typer.Option(help="YAML file to create")] = Path("config.yaml"),
) -> None:
    """Create a documented schema-1 configuration."""
    if output.exists():
        _fail(ValueError(f"File already exists: {output}"))
    write_default_config(output)
    console.print(f"Wrote {output}")


@app.command()
def scan(
    dataset: Path,
    output: Annotated[Path, typer.Option(help="JSONL manifest path")] = Path("manifest.jsonl"),
    config: Annotated[Path | None, typer.Option(help="Optional YAML configuration")] = None,
) -> None:
    """Scan audio metadata, hashes, and validation status."""
    cfg = _config(config)
    count = 0
    errors = 0
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        import os
        import tempfile

        fd, temp = tempfile.mkstemp(prefix=".scan-", dir=output.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                for row in scan_dataset(dataset, cfg):
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                    errors += row["validation_status"] == "error"
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, output)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    except (OSError, ValueError) as exc:
        _fail(exc)
    console.print(f"Scanned {count} files: {errors} validation errors. Wrote {output}")
    if errors:
        raise typer.Exit(1)


@app.command()
def validate(
    dataset: Path,
    report: Annotated[Path, typer.Option("--report", help="JSON validation report")] = Path(
        "validation.json"
    ),
    config: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Validate discovered audio and write detailed JSON results."""
    cfg = _config(config)
    try:
        rows = list(scan_dataset(dataset, cfg))
        summary = {
            "total": len(rows),
            "errors": sum(r["validation_status"] == "error" for r in rows),
            "warnings": sum(len(r["validation_warnings"]) for r in rows),
            "files": rows,
        }
        _atomic_json(report, summary)
    except (OSError, ValueError) as exc:
        _fail(exc)
    console.print(
        f"Validated {summary['total']} files: {summary['errors']} errors, "
        f"{summary['warnings']} warnings. Wrote {report}"
    )
    if summary["errors"]:
        raise typer.Exit(1)


@app.command()
def process(
    dataset: Path,
    config: Annotated[Path, typer.Option(help="YAML configuration")],
    output: Annotated[Path, typer.Option(help="Output directory")],
    workers: Annotated[int | None, typer.Option(min=1, max=32)] = None,
    dry_run: Annotated[bool, typer.Option()] = False,
    fail_fast: Annotated[bool, typer.Option()] = False,
    verbose: Annotated[bool, typer.Option()] = False,
    quiet: Annotated[bool, typer.Option()] = False,
) -> None:
    """Process a directory with resumable, atomic outputs."""
    cfg = _config(config)
    if workers is not None:
        cfg.execution.workers = workers
    if verbose and quiet:
        _fail(ValueError("--verbose and --quiet cannot be combined"))
    try:
        summary = process_dataset(dataset, output, cfg, fail_fast, dry_run)
    except (OSError, ValueError) as exc:
        _fail(exc)
    if not quiet:
        if verbose:
            console.print(json.dumps(summary, indent=2))
        elif dry_run:
            console.print(f"Dry run: {summary['total_discovered']} files discovered")
        else:
            console.print(
                f"{summary['successful']} processed, {summary['failed']} failed, "
                f"{summary['cached']} cached; {summary['total_discovered']} discovered. "
                f"{summary['total_input_duration_s']:.2f}s input -> "
                f"{summary['total_output_duration_s']:.2f}s output; "
                f"{summary['feature_files']} features, {summary['warning_count']} warnings; "
                f"{summary['throughput_audio_s_per_wall_s']:.2f} audio s/s"
            )
    if summary["failed"]:
        raise typer.Exit(1)


@app.command()
def features(
    dataset: Path,
    type: Annotated[str, typer.Option("--type", help="mfcc, mel, or log_mel")],
    output: Annotated[Path, typer.Option(help="Feature output directory")],
    config: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Export .npy features and JSON metadata from audio."""
    if type not in ("mfcc", "mel", "log_mel"):
        _fail(ValueError("--type must be mfcc, mel, or log_mel"))
    cfg = _config(config)
    root = dataset.resolve()
    out = output.resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        _fail(ValueError("Feature output must be separate from input directory"))
    count = 0
    failed = 0
    try:
        for path in discover(root):
            relative = path.relative_to(root)
            relative = relative.with_name(relative.name + f".{type}.npy")
            try:
                data, rate, _ = read_audio(
                    path, cfg.audio.max_file_mb, cfg.validation.max_duration_s
                )
                feature = extract_feature(data.mean(axis=1), rate, type, cfg.features)
                target = _safe_output(out, relative)
                _atomic_npy(target, feature)
                _atomic_json(
                    target.with_suffix(".json"),
                    {
                        "feature_type": type,
                        "configuration": cfg.features.model_dump(mode="json"),
                        "sample_rate": rate,
                        "shape": list(feature.shape),
                        "dtype": str(feature.dtype),
                        "source_file": path.relative_to(root).as_posix(),
                        "processing_version": "0.1.0",
                    },
                )
                count += 1
            except (OSError, ValueError) as exc:
                console.print(f"[red]{path}:[/red] {exc}")
                failed += 1
    except (OSError, ValueError) as exc:
        _fail(exc)
    console.print(f"Exported {count} feature files; {failed} failed")
    if failed:
        raise typer.Exit(1)


@app.command()
def split(
    manifest: Path,
    config: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option(help="Split output directory")],
) -> None:
    """Write reproducible train, validation, and test manifests."""
    cfg = _config(config)
    try:
        records = [
            row
            for row in read_jsonl(manifest)
            if row.get("processing_status", "success") == "success"
        ]
        assigned = split_manifest(records, cfg)
        for name in ("train", "validation", "test"):
            _write_jsonl(
                output / f"{name}.jsonl", [row for row in assigned if row["split"] == name]
            )
        _write_jsonl(output / "splits.jsonl", assigned)
    except (OSError, ValueError, KeyError) as exc:
        _fail(exc)
    console.print(f"Split {len(assigned)} records into {output}")


@app.command()
def report(
    dataset: Path,
    output: Annotated[Path, typer.Option(help="HTML report path")] = Path("report.html"),
) -> None:
    """Render the processing report and failures to HTML."""
    try:
        summary = json.loads((dataset / "report.json").read_text(encoding="utf-8"))
        failure_rows: list[str] = []
        comparison_rows: list[str] = []
        for record in read_jsonl(dataset / "manifest.jsonl"):
            source = html.escape(str(record.get("source_path", "")))
            if record.get("processing_status") == "success":
                original = float(record.get("original_duration", 0))
                processed = float(record.get("output_duration", 0))
                comparison_rows.append(
                    f"<tr><td>{source}</td><td>{original:.3f}</td>"
                    f"<td>{processed:.3f}</td><td>{original - processed:.3f}</td></tr>"
                )
            else:
                error = html.escape(str(record.get("error_message", "")))
                failure_rows.append(f"<tr><td>{source}</td><td>{error}</td></tr>")
        stats = "".join(
            f"<li><strong>{html.escape(str(k))}</strong>: {html.escape(str(v))}</li>"
            for k, v in summary.items()
        )
        page = (
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>Speech Data Toolkit report</title><h1>Processing report</h1>"
            f"<ul>{stats}</ul><h2>Before and after (seconds)</h2><table>"
            "<tr><th>Source</th><th>Original</th><th>Output</th><th>Removed</th></tr>"
            f"{''.join(comparison_rows)}</table><h2>Failures</h2><table>"
            "<tr><th>Source</th><th>Error</th></tr>"
            f"{''.join(failure_rows)}</table></html>"
        )
        _atomic_bytes(output, page.encode("utf-8"))
    except (OSError, ValueError, KeyError) as exc:
        _fail(exc)
    console.print(f"Wrote {output}")


if __name__ == "__main__":
    app()
