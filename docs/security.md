# Security

`music-listener` is designed so that even a hostile or corrupt audio file, or a
crafted filename, cannot execute code, read secrets, or escape its output
directory. This document is the honest blast-radius assessment, not a marketing
checklist.

## Verified posture

### No shell injection
Every subprocess call uses an **argv list** (`subprocess.run([...])`) with
`shell=False` — there is **no `shell=True` anywhere**. A crafted
`audio_path`/filename or grid value cannot become a shell command. The demucs and
whisper invocations also pass the audio path as a positional argv element, not a
shell string.

### No path traversal
`slugify()` keeps only `[a-zA-Z0-9_-]` from the input filename for the output
directory name, so a path with `../` or `/` cannot redirect writes outside the
intended output root. The output root itself comes from Hermes' resolved agent
cwd + the configured `output_dir`.

### No credential access
`__init__.py` and `worker.py` read **no** secrets, tokens, `.env` files, or
credential stores. The only optional env the manifest exposes is `HF_TOKEN`
(for gated HuggingFace whisper models), and it is not touched unless the user
chooses such a model.

### Bounded resource use
The worker subprocess is capped with a **60-minute hard timeout** so a huge or
maliciously-shaped audio file cannot pin the CPU forever. Model weights are
native-inference only; a corrupt/fuzz audio file feeds standard decoders
(soundfile/librosa) and cannot reach shell or arbitrary file writes.

### No network surface
The plugin **binds no ports, opens no sockets, and runs no server**. Its only
outbound network activity is model-weight downloads from the official upstream
endpoints (Demucs weights, faster-whisper). There is no telemetry, no callback,
no "phone home."

## Blast radius

| Vector | Exposure | Mitigation |
|---|---|---|
| Malicious audio content | No code execution — only decode + model inference | Standard decoders; no evaluation of audio bytes as code |
| Filename injection | No shell execution | argv lists, `shell=False`, slugify |
| Output path escape | No writes outside output root | slugify + resolved cwd |
| Secret exfiltration | None | No credential reads |
| CPU exhaustion | Worker capped at 60m | Hard subprocess timeout (also protects the gateway event loop) |
| Model supply chain | Demucs/whisper weights from official hosts | Standard ML-tool exposure — verify pinned model hashes if threat model requires it |

## Residual / known (accepted) risks

1. **Supply chain (third-party).** The plugin trusts PyTorch/Demucs/Basic Pitch/
   faster-whisper weights from their official hosts. This is the same exposure as
   any ML tool and is accepted for a local, user-triggered transcription tool.
2. **Processing can be slow/heavy** on very long or very dense files (CPU
   transcription). This is a user-triggered workload, not an ambient service, so
   the blast radius is limited to the calling session.
3. **Model download destinations** are the official PyPI/pytorch/HF endpoints;
   if your org pins supply chains, pin the wheel hashes and the model files
   too.

## Recommendations

- Run inside the user's existing Hermes profile (the normal case) — do not grant
  the worker elevated privileges.
- If the threat model is high (e.g. transcribing untrusted third-party media on
  a shared host), consider adding per-model hash pinning on the downloaded
  weights before they are loaded into Inference.
- The venv is disposable: `rm -rf ~/.hermes/plugins/music-listener/.venv` and
  re-run `./install.sh` to rebuild it from the pinned requirements.