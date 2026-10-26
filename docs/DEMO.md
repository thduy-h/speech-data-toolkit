# Reproducible synthetic comparison

`examples/make_demo.py` generated a 3.0-second 8 kHz mono WAV with two tone bursts and silence. With `silence.enabled: true`, the processing workflow on Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1, and SoundFile 0.14.0 produced the following measured manifest values:

| Original | Output | Removed | Original interval retained | Output |
|---:|---:|---:|---|---|
| 3.000 s | 2.250 s | 0.750 s | 0.325–2.575 s | 16 kHz mono WAV |

The retained interval includes the pause between bursts. Results were read from `/tmp/sdt-demo-trimmed/manifest.jsonl` after the actual CLI run; rounding may differ slightly with other dependency versions. This is a correctness demonstration, not a performance benchmark.

To reproduce:

```bash
uv run python examples/make_demo.py demo-audio
uv run speechprep init --output config.yaml
# Edit config.yaml: set silence.enabled: true
uv run speechprep process demo-audio --config config.yaml --output processed
uv run speechprep report processed --output report.html
```

The HTML report shows per-file original, output, and removed durations. Its JSON counterpart is `processed/report.json`.
