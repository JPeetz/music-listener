#!/usr/bin/env python3
"""Synthetic smoke test for the music-listener pipeline.

Synthesizes a ~6s multi-part track (chord pad + bassline + arpeggiated melody +
percussive clicks) with numpy, renders it to WAV via soundfile, runs the real
worker.py end-to-end, and asserts the three artifacts exist and the JSON is valid.

Usage:
    ~/.hermes/plugins/music-listener/.venv/bin/python \
        ~/.hermes/plugins/music-listener/tests/test_pipeline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PLUGIN_DIR = Path(__file__).resolve().parents[1]
VENV_PY = PLUGIN_DIR / ".venv" / "bin" / "python"
WORKER = PLUGIN_DIR / "worker.py"

SR = 44100
DUR = 6.0


def make_test_audio(path: Path) -> None:
    import soundfile as sf
    t = np.arange(int(SR * DUR)) / SR
    mix = np.zeros_like(t)
    # chord pad (C major): C4 E4 G4
    for freq in (261.63, 329.63, 392.00):
        mix += 0.12 * np.sin(2 * np.pi * freq * t)
    # bassline: low C2 pumping eighth notes
    for k in range(0, 12):
        s = k * 0.5
        env = np.exp(-6 * np.clip(t - s, 0, None))
        mix += 0.18 * np.sin(2 * np.pi * 65.41 * t) * (t >= s) * env
    # arpeggiated melody: C5 E5 G5 C6 quarter notes
    mel = [523.25, 659.25, 783.99, 1046.5][:6]
    for i, f in enumerate(mel * 2):
        s = i * 0.9
        env = np.exp(-4 * np.clip(t - s, 0, None))
        mix += 0.22 * np.sin(2 * np.pi * f * t) * (t >= s) * env
    # percussive clicks every 0.25s (transient noise)
    rng = np.random.default_rng(0)
    clicks = np.zeros_like(t)
    for k in range(0, int(DUR / 0.25)):
        s = k * 0.25
        n_i = int(s * SR)
        n_len = int(0.03 * SR)
        if n_i + n_len < len(t):
            clicks[n_i:n_i + n_len] = rng.standard_normal(n_len) * 0.9
    mix += clicks
    # Normalize to ~0.9 peak so per-stem files (after separation) stay loud
    # enough for basic-pitch to detect real notes.
    peak = float(np.max(np.abs(mix))) or 1.0
    mix = mix * (0.9 / peak)
    sf.write(str(path), mix.astype(np.float32), SR)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        audio = tmp / "synth_test.wav"
        out_dir = tmp / "out"
        make_test_audio(audio)
        assert audio.exists() and audio.stat().st_size > 1000, "test audio empty"

        cmd = [str(VENV_PY), str(WORKER), str(audio), str(out_dir),
               "--grid", "16th", "--lyrics", "false", "--device", "cpu"]
        print(">> running worker:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        print(">> stderr tail:\n", (proc.stderr or "")[-1500:])
        if proc.returncode != 0:
            print("WORKER EXIT", proc.returncode)
            return 1

        stdout_lines = (proc.stdout or "").strip().splitlines()
        if not stdout_lines:
            print("NO JSON ON STDOUT"); return 1
        try:
            result = json.loads(stdout_lines[-1])
        except json.JSONDecodeError as e:
            print("BAD JSON:", e); print(stdout_lines[-1]); return 1

        # assertions
        assert result.get("ok") is True, f"worker not ok: {result}"
        artifacts = result.get("artifacts", {})
        for key in ("midi", "musicxml", "summary"):
            assert key in artifacts, f"missing artifact key {key}"
            p = Path(artifacts[key])
            assert p.exists(), f"artifact missing: {p}"
        assert result.get("tracks"), "no tracks in summary"

        # score_summary.json matches printed summary
        summary_path = Path(artifacts["summary"])
        on_disk = json.loads(summary_path.read_text())
        assert on_disk.get("ok") is True, "on-disk summary not ok"

        print("=== SMOKE TEST PASSED ===")
        print("  midi:       ", artifacts["midi"])
        print("  musicxml:   ", artifacts["musicxml"])
        print("  summary:    ", artifacts["summary"])
        print("  backend:    ", result.get("backend"))
        print("  tempo bpm:  ", result.get("analysis", {}).get("tempo_bpm"))
        print("  tracks:     ", result.get("tracks"))
        print("  stems dir:  ", artifacts.get("stems_dir"))
        return 0


if __name__ == "__main__":
    sys.exit(main())