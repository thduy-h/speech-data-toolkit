from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sine(tmp_path):
    rate = 8000
    signal = (0.2 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, signal, rate)
    return path
