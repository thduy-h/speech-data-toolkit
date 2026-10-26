from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from speech_data_toolkit.config import ToolkitConfig
from speech_data_toolkit.dataset import read_jsonl, safe_relative, scan_dataset, split_manifest
from speech_data_toolkit.pipeline import AudioProcessor, process_dataset


def test_scan_nested_unicode_duplicate_and_corrupt(tmp_path):
    root = tmp_path / "input"
    for folder in ("α", "β"):
        (root / folder).mkdir(parents=True)
        sf.write(root / folder / "same.wav", np.zeros(800, dtype=np.float32), 8000)
    (root / "bad.wav").write_bytes(b"not a wav")
    rows = list(scan_dataset(root))
    assert len(rows) == 3
    assert {r["source_path"] for r in rows} == {"α/same.wav", "β/same.wav", "bad.wav"}
    assert next(r for r in rows if r["source_path"] == "bad.wav")["validation_status"] == "error"
    assert all(len(r["source_sha256"]) == 64 for r in rows)


def test_config_fingerprint_and_unknown_keys():
    a = ToolkitConfig()
    b = ToolkitConfig()
    assert a.fingerprint() == b.fingerprint()
    b.audio.sample_rate = 22050
    assert a.fingerprint() != b.fingerprint()
    with pytest.raises(ValueError):
        ToolkitConfig.model_validate({"surprise": True})


def test_resume_changed_input_config_and_partial_recovery(sine, tmp_path):
    out = tmp_path / "out"
    cfg = ToolkitConfig()
    first = AudioProcessor(cfg).process_file(sine, out)
    assert first.processing_status == "success" and not first.cached
    assert first.output_sample_rate == 16000
    assert first.output_duration == pytest.approx(1.0, abs=1e-4)
    assert AudioProcessor(cfg).process_file(sine, out).cached
    (out / first.output_paths[0]).write_bytes(b"partial")
    assert not AudioProcessor(cfg).process_file(sine, out).cached
    cfg.audio.sample_rate = 22050
    assert not AudioProcessor(cfg).process_file(sine, out).cached
    sf.write(sine, np.zeros(8000, dtype=np.float32), 8000)
    assert not AudioProcessor(cfg).process_file(sine, out).cached


def test_process_dataset_manifest_and_failure_isolation(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    sf.write(root / "good.wav", np.ones(800, dtype=np.float32) * 0.2, 8000)
    (root / "bad.wav").write_bytes(b"corrupt")
    out = tmp_path / "processed"
    cfg = ToolkitConfig()
    cfg.execution.workers = 2
    summary = process_dataset(root, out, cfg)
    assert summary["successful"] == 1 and summary["failed"] == 1
    assert (out / "manifest.csv").exists()
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    good = next(r for r in rows if r["processing_status"] == "success")
    assert good["output_path"] == "good.wav"
    assert good["source_intervals"] == [{"start": 0.0, "end": 0.1}]
    assert len(good["feature_outputs"]) == 2
    assert (out / good["output_path"]).exists()
    assert process_dataset(root, out, cfg)["cached"] == 1


def test_speaker_disjoint_and_path_traversal(tmp_path):
    cfg = ToolkitConfig()
    cfg.split.speaker_disjoint = True
    records = [
        {"source_path": f"{speaker}/{i}.wav", "speaker_id": speaker}
        for speaker in "abcd"
        for i in range(3)
    ]
    assigned = split_manifest(records, cfg)
    by_speaker = {}
    for row in assigned:
        by_speaker.setdefault(row["speaker_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_speaker.values())
    with pytest.raises(ValueError):
        split_manifest([{"source_path": "x.wav"}], cfg)
    with pytest.raises(ValueError):
        safe_relative("../escape.wav")
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"source_path":"../escape.wav"}\n')
    with pytest.raises(ValueError):
        list(read_jsonl(path))


def test_empty_dataset(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert list(scan_dataset(root)) == []
    summary = process_dataset(root, tmp_path / "out", ToolkitConfig())
    assert summary["total_discovered"] == 0


def test_silence_trim_and_vad_timestamp_modes(sine, tmp_path):
    rate = 8000
    wave = np.zeros(rate * 3, dtype=np.float32)
    wave[rate // 2 : rate] = 0.3
    wave[rate * 2 : rate * 2 + rate // 2] = 0.3
    sf.write(sine, wave, rate)
    cfg = ToolkitConfig()
    cfg.features.enabled = False
    cfg.silence.enabled = True
    cfg.silence.padding_ms = 100
    cfg.vad.enabled = True
    cfg.vad.speech_padding_ms = 50
    cfg.vad.min_speech_ms = 100
    cfg.vad.mode = "split"
    split = AudioProcessor(cfg).process_file(sine, tmp_path / "split")
    assert len(split.output_paths) == 2
    assert len(split.source_intervals) == 2
    assert split.source_intervals[0]["start"] < 0.5
    assert split.source_intervals[1]["start"] > 1.8
    assert split.output_duration < 2
    cfg.vad.mode = "trim_edges"
    trimmed = AudioProcessor(cfg).process_file(sine, tmp_path / "trim")
    assert len(trimmed.output_paths) == 1
    assert trimmed.output_duration > split.output_duration
    assert len(trimmed.source_intervals) == 1
    cfg.vad.mode = "concat"
    joined = AudioProcessor(cfg).process_file(sine, tmp_path / "concat")
    assert len(joined.source_intervals) == 2
    assert joined.output_duration == pytest.approx(split.output_duration, abs=1 / 8000)


def test_reject_output_inside_input_and_skip_symlink(tmp_path, sine):
    root = tmp_path / "data"
    root.mkdir()
    (root / "link.wav").symlink_to(sine)
    assert list(scan_dataset(root)) == []
    with pytest.raises(ValueError):
        process_dataset(root, root / "out", ToolkitConfig())


def test_same_stem_different_formats_do_not_collide(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    samples = np.ones(800, dtype=np.float32) * 0.2
    sf.write(root / "clip.wav", samples, 8000)
    sf.write(root / "clip.flac", samples, 8000)
    out = tmp_path / "out"
    summary = process_dataset(root, out, ToolkitConfig())
    assert summary["successful"] == 2
    assert (out / "clip.wav").exists()
    assert (out / "clip.flac.wav").exists()
    assert process_dataset(root, out, ToolkitConfig())["cached"] == 2


def test_validation_config_rejects_incompatible_thresholds():
    with pytest.raises(ValueError, match="min_sample_rate"):
        ToolkitConfig.model_validate(
            {"validation": {"min_sample_rate": 48000, "max_sample_rate": 8000}}
        )


def test_fail_fast_writes_partial_manifest_and_report(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "bad.wav").write_bytes(b"broken")
    sf.write(root / "good.wav", np.ones(800, dtype=np.float32) * 0.2, 8000)
    out = tmp_path / "out"
    summary = process_dataset(root, out, ToolkitConfig(), fail_fast=True)
    assert summary["failed"] == 1 and summary["stopped_early"]
    assert summary["total_discovered"] == 2
    rows = list(read_jsonl(out / "manifest.jsonl"))
    assert len(rows) == 1 and rows[0]["source_path"] == "bad.wav"
    assert (out / "report.json").exists()


def test_nested_duplicate_basenames_have_distinct_outputs(tmp_path):
    root = tmp_path / "data"
    for folder in ("speaker A", "spéaker B"):
        directory = root / folder
        directory.mkdir(parents=True)
        sf.write(directory / "same.wav", np.ones(800, dtype=np.float32) * 0.2, 8000)
    out = tmp_path / "out"
    summary = process_dataset(root, out, ToolkitConfig())
    assert summary["successful"] == 2
    assert (out / "speaker A" / "same.wav").exists()
    assert (out / "spéaker B" / "same.wav").exists()


def test_directory_speaker_metadata_flows_to_manifest(tmp_path):
    root = tmp_path / "data"
    for speaker in ("speaker 1", "speaker 2"):
        directory = root / speaker
        directory.mkdir(parents=True)
        sf.write(directory / "sample.wav", np.ones(800, dtype=np.float32) * 0.2, 8000)
    cfg = ToolkitConfig()
    cfg.dataset.speaker_directory_level = 0
    out = tmp_path / "out"
    assert process_dataset(root, out, cfg)["successful"] == 2
    records = list(read_jsonl(out / "manifest.jsonl"))
    assert {r["speaker_id"] for r in records} == {"speaker 1", "speaker 2"}
    cfg.split.speaker_disjoint = True
    assigned = split_manifest(records, cfg)
    assert len(assigned) == 2


def test_csv_speaker_manifest(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    sf.write(root / "recording.wav", np.ones(800, dtype=np.float32) * 0.2, 8000)
    labels = tmp_path / "labels.csv"
    labels.write_text("source_path,speaker_id\nrecording.wav,person α\n", encoding="utf-8")
    cfg = ToolkitConfig()
    cfg.dataset.speaker_manifest = str(labels)
    row = next(scan_dataset(root, cfg))
    assert row["speaker_id"] == "person α"


def test_output_symlink_cannot_escape_root(tmp_path, sine):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        AudioProcessor(ToolkitConfig()).process_file(
            sine, output, relative_path="escape/recording.wav"
        )
    assert not (outside / "recording.wav").exists()


def test_interrupted_batch_resumes_completed_file(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    samples = np.ones(800, dtype=np.float32) * 0.2
    sf.write(root / "a.wav", samples, 8000)
    sf.write(root / "b.wav", samples, 8000)
    out = tmp_path / "out"
    cfg = ToolkitConfig()
    original = AudioProcessor.process_file

    def interrupt_on_second(self, source, output_dir, relative_path=None, speaker_id=None):
        if source.name == "b.wav":
            raise KeyboardInterrupt
        return original(self, source, output_dir, relative_path, speaker_id)

    with monkeypatch.context() as patch:
        patch.setattr(AudioProcessor, "process_file", interrupt_on_second)
        with pytest.raises(KeyboardInterrupt):
            process_dataset(root, out, cfg)

    assert (out / "a.wav").exists()
    assert not (out / "manifest.jsonl").exists()
    resumed = process_dataset(root, out, cfg)
    assert resumed["successful"] == 2
    assert resumed["cached"] == 1


def test_resampling_overshoot_reports_pcm_clipping(tmp_path):
    source = tmp_path / "step.wav"
    samples = np.r_[np.full(100, 0.9), np.full(100, -0.9)].astype(np.float32)
    sf.write(source, samples, 8000)
    cfg = ToolkitConfig()
    cfg.normalization.enabled = False
    cfg.features.enabled = False
    cfg.validation.min_duration_s = 0
    result = AudioProcessor(cfg).process_file(source, tmp_path / "out")
    assert any("PCM output was clipped" in warning for warning in result.validation_warnings)
    output, rate = sf.read(tmp_path / "out" / "step.wav")
    assert rate == 16000
    assert np.max(np.abs(output)) <= 1.0
