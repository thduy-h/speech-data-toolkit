# Architecture and contracts

The public API is `AudioProcessor.from_config(path)` and `process_file(source, output_dir)`; results are structured `ProcessResult` records. The CLI uses the same library functions. The schema is strict YAML version 1. JSONL manifests use relative `source_path` and `output_path` fields; optional metadata is represented as JSON values. A processing record is successful only after all output hashes and its atomic cache record are committed.

Pipeline order: decode and validate input; downmix; anti-alias resample; optional silence edge trim; optional energy VAD; optional amplitude normalization; write waveform; extract and atomically write features and metadata. Segment timestamps always refer to the original audio timeline in seconds. Stereo downmix is the arithmetic mean of channels.

Core decoding uses libsndfile through SoundFile. WAV and FLAC are supported. Compressed decoding and neural VAD are intentionally deferred pending compatible, tested dependency combinations; there is no Torch requirement. The energy detector estimates signal activity and is not a neural speech detector.

Official documentation pages for SciPy `signal.resample_poly`, python-soundfile, Pydantic models, and Typer were reachable and checked on 2026-10-25. PyPI wheel metadata and an actual Python 3.11 installation supplied the version compatibility evidence. Core dependency ranges remain intentionally broad within tested major versions. TorchCodec, TorchAudio, and PyTorch are not dependencies because no optional neural or compressed-format integration is shipped.
