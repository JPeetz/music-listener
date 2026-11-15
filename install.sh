#!/usr/bin/env bash
# Build the music-listener worker's isolated .venv and install its heavy
# audio/ML stack. Safe to re-run (idempotent). Never mutates Hermes' core
# Python environment.
#
# Requires a Python 3.11 (torch/torchaudio/demucs wheels are solid on 3.11
# for both macOS and Linux; 3.12/3.13/3.14 are not reliably wheel-covered).
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PLUGIN_DIR/.venv"
REQ="$PLUGIN_DIR/requirements.txt"

# --- Pick a python3.11 ---------------------------------------------------
PY=""
for c in python3.11 /usr/local/bin/python3.11 "$HOME/.local/bin/python3.11" python3.10; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [[ -z "$PY" ]]; then
  echo "FATAL: no python3.11 found. Install one (e.g. via uv, pyenv, or brew)."
  exit 1
fi
echo ">> Using Python: $($PY --version 2>&1)"

# --- Create the venv ------------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  echo ">> Creating isolated venv at $VENV ..."
  "$PY" -m venv "$VENV"
else
  echo ">> Venv already exists: $VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
echo ">> venv python: $($VENV/bin/python --version 2>&1)"

# --- Install deps (CPU torch) ----------------------------------------------
echo ">> Installing audio/ML dependencies (this downloads ~1-2 GB, takes a while)..."
"$VENV/bin/pip" install --quiet -r "$REQ" || {
    echo "WARN: bulk install failed; retrying a minimal core (torch/audio) first..."
    "$VENV/bin/pip" install --quiet "torch==2.2.2" "torchaudio==2.2.2" "numpy==1.26.4" || exit 1
    "$VENV/bin/pip" install --quiet -r "$REQ" || exit 1
  }

# --- Sanity check -----------------------------------------------------------
echo ">> Sanity check:"
"$VENV/bin/python" - <<'PY'
import importlib.util
mods = ["torch","torchaudio","demucs","basic_pitch","librosa","soundfile",
        "scipy","numpy","torchcrepe","faster_whisper","pretty_midi","music21"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("  MISSING:", ", ".join(missing)); raise SystemExit(1)
import torch
print(f"  torch {torch.__version__} backend: "
      f"cuda={torch.cuda.is_available()} mps={getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()} cpu=True")
print("  OK - all", len(mods), "modules importable")
PY

echo
echo "=== install complete ==="
echo "  venv:        $VENV"
echo "  next:        hermes plugins doctor music-listener && hermes plugins enable music-listener"