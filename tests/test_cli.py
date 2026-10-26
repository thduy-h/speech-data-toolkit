from __future__ import annotations

import json
import shutil

from typer.testing import CliRunner

from speech_data_toolkit.cli import app

runner = CliRunner()


def test_help_and_invalid_config(tmp_path):
    for command in ("init", "scan", "validate", "process", "features", "split", "report"):
        assert runner.invoke(app, [command, "--help"]).exit_code == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("unknown: true\n")
    result = runner.invoke(
        app, ["process", str(tmp_path), "--config", str(bad), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code == 2


def test_cli_workflow(sine, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy2(sine, data / sine.name)
    config = tmp_path / "config.yaml"
    assert runner.invoke(app, ["init", "--output", str(config)]).exit_code == 0
    scan = tmp_path / "scan.jsonl"
    assert runner.invoke(app, ["scan", str(data), "--output", str(scan)]).exit_code == 0
    assert json.loads(scan.read_text().splitlines()[0])["sample_rate"] == 8000
    valid = tmp_path / "valid.json"
    assert runner.invoke(app, ["validate", str(data), "--report", str(valid)]).exit_code == 0
    out = tmp_path / "processed"
    assert (
        runner.invoke(
            app, ["process", str(data), "--config", str(config), "--output", str(out)]
        ).exit_code
        == 0
    )
    assert (out / "tone.wav").exists()
    features = tmp_path / "external-features"
    assert (
        runner.invoke(
            app, ["features", str(out), "--type", "mfcc", "--output", str(features)]
        ).exit_code
        == 0
    )
    assert (features / "tone.wav.mfcc.npy").exists()
    splits = tmp_path / "splits"
    assert (
        runner.invoke(
            app,
            [
                "split",
                str(out / "manifest.jsonl"),
                "--config",
                str(config),
                "--output",
                str(splits),
            ],
        ).exit_code
        == 0
    )
    assert (splits / "splits.jsonl").exists()
    html = tmp_path / "report.html"
    assert runner.invoke(app, ["report", str(out), "--output", str(html)]).exit_code == 0
    assert "Processing report" in html.read_text()
    assert "Before and after" in html.read_text()


def test_cli_corrupt_exit_and_empty_dataset(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    assert (
        runner.invoke(app, ["scan", str(root), "--output", str(tmp_path / "empty.jsonl")]).exit_code
        == 0
    )
    (root / "broken.wav").write_bytes(b"broken")
    result = runner.invoke(app, ["scan", str(root), "--output", str(tmp_path / "bad.jsonl")])
    assert result.exit_code == 1
    assert "Failed decoding" in (tmp_path / "bad.jsonl").read_text()


def test_cli_fail_fast_has_data_error_exit_code(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "bad.wav").write_bytes(b"broken")
    config = tmp_path / "config.yaml"
    assert runner.invoke(app, ["init", "--output", str(config)]).exit_code == 0
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["process", str(data), "--config", str(config), "--output", str(out), "--fail-fast"],
    )
    assert result.exit_code == 1
    assert (out / "manifest.jsonl").exists()
    assert json.loads((out / "report.json").read_text())["stopped_early"]


def test_process_dry_run_and_worker_flags(sine, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy2(sine, data / "tone.wav")
    config = tmp_path / "config.yaml"
    assert runner.invoke(app, ["init", "--output", str(config)]).exit_code == 0
    out = tmp_path / "out"
    dry = runner.invoke(
        app,
        ["process", str(data), "--config", str(config), "--output", str(out), "--dry-run"],
    )
    assert dry.exit_code == 0 and "Dry run: 1 files discovered" in dry.output
    assert not out.exists()
    run = runner.invoke(
        app,
        [
            "process",
            str(data),
            "--config",
            str(config),
            "--output",
            str(out),
            "--workers",
            "2",
            "--quiet",
        ],
    )
    assert run.exit_code == 0 and run.output == ""
    assert (out / "tone.wav").exists()
