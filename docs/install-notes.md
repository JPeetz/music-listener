# Install notes & platform gotchas

`install.sh` builds an isolated `.venv` and installs the pinned dependencies. The
pins in `requirements.txt` are what actually resolve (and build) on this
project's reference host — an **Intel i7 MacBook**: no CUDA, no MPS, CPU-only.
These notes will save you the debugging time we spent the first time.

## Python version: use 3.11, not the system python

The macOS system `python3` is often far newer (e.g. 3.15) than what ML wheels
support. Pin the venv to **Python 3.11** (available as `python3.11` /
`~/.local/bin/python3.11`). torch/torchaudio/demucs/basic-pitch/faster-whisper
all have solid cp311 x86_64 macOS wheels.

```bash
python3.11 -m venv ~/.hermes/plugins/music-listener/.venv
```

`install.sh` auto-picks python3.11 on PATH; override via the `PY` detection or
just create the venv yourself first.

## The wheel pins that actually work (x86_64 macOS)

- **torch / torchaudio: 2.2.2.** PyTorch stopped shipping prebuilt macosx x86_64
  wheels after 2.2.2 — newer builds are Apple-Silicon only. Without a pin, pip
  can't find a matching Intel-mac distribution.
- **llvmlite 0.42.0, numba 0.59.1.** The last versions with macosx x86_64
  wheels. Anything newer has no Intel-mac wheel, so pip falls back to building
  llvmlite from source, which needs LLVM + a C toolchain and fails on a bare Mac.
  librosa pulls numba; pin both or the install dies on llvmlite.
- **numpy 1.26.4** (`numpy<2`) — matches the torch 2.2.2 ABI.
- **setuptools<81.** pretty_midi imports `pkg_resources`, which setuptools
  removed in 81+. Without the pin:
  `ModuleNotFoundError: No module named 'pkg_resources'`.

On Apple Silicon or Linux you can bump torch freely; keep the numba/llvmlite /
setuptools pins unless you've verified newer ones ship wheels for your arch.

## Demucs: don't pass `--two-stems`

`htdemucs_6s` emits **all 6 stems by default**. Passing `--two-stems vocals`
collapses to a 2-stem `vocals` + `no_vocals` split and you lose
drums/bass/guitar/piano/other. The worker calls
`demucs.separate -n htdemucs_6s -o <dir> <audio>` and reads the 6 `<stem>.wav`
files back from the nested `<dir>/htdemucs_6s/<slug>/` layout.

## torchcrepe hangs on CPU — librosa pyin is the reliable path

torchcrepe's `full.pth` CPU inference can hang **indefinitely** on Intel-mac
CPU hosts. The worker makes **librosa `pyin` the primary** f0 extractor and runs
torchcrepe behind a 30s watchdog (daemon thread; if it's still alive, fall through
to pyin). Never call `torchcrepe.predict` unconditionally on CPU. `pyin` returns
N/2 frames at a 20ms hop; the worker reindexes it onto a 10ms grid by linear
interpolation to match the contour length downstream expects.

## faster-whisper: run it FIRST, not last

ctranslate2 (faster-whisper's engine) segfaults or OOM-kills when loaded *after*
torch + demucs + basic-pitch in the same process on 16 GB hosts. The worker runs
whisper in its **own subprocess** and schedules it **immediately after
separation** (before the heavy models load), storing the words for later stages.
This is why the pipeline can finish real 2-3 minute songs that would otherwise
die mid-lyrics.

## Model downloads & caches

First run downloads Demucs weights (to `~/.cache/torch/hub/`) and, if lyrics are
enabled, the faster-whisper model (to `HF_HOME`). Set `HF_HOME` if you want the
cache somewhere specific:

```bash
export HF_HOME=~/.hermes/plugins/music-listener/.hf-cache
```

## Keep the smoke test fast

`tests/test_pipeline.py` synthesizes a ~6s multi-part clip and runs the **real**
worker (separation → transcription → export). Keep `--lyrics false` in the smoke
test; whisper adds a long model download not needed to prove the pipeline wiring.
Use `--lyrics true` on a real vocal track to exercise the full path.