"""Generate a small, uncopyrighted speech-like activity example."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo-audio") / "speaker_01"
    out.mkdir(parents=True, exist_ok=True)
    rate = 8000
    time = np.arange(rate * 3, dtype=np.float32) / rate
    audio = np.zeros_like(time)
    for start, end, frequency in ((0.45, 1.15, 180), (1.55, 2.45, 240)):
        mask = (time >= start) & (time < end)
        envelope = np.sin(np.linspace(0, np.pi, int(np.count_nonzero(mask)), dtype=np.float32)) ** 2
        audio[mask] = 0.35 * envelope * np.sin(2 * np.pi * frequency * time[mask])
    sf.write(out / "utterance.wav", audio, rate)
    print(out / "utterance.wav")


if __name__ == "__main__":
    main()
