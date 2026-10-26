# Contributing

Open an issue describing the use case and expected behavior before proposing major dependency changes. Keep core CPU dependencies small and optional ML integrations isolated. Run `uv sync --extra dev`, `uv run ruff format .`, `uv run ruff check .`, `uv run mypy src`, and `uv run pytest` before submitting a pull request. Add deterministic synthetic tests for signal changes and document timestamp or manifest changes. Never add third-party audio without redistribution rights.
