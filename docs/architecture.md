# Architecture

The music-listener pipeline is a single standalone CLI (`worker.py`) driven by
a thin Hermes plugin entry point (`__init__.py`). The heavy audio/ML stack lives
in an isolated `.venv`; the plugin handler only shells out to it and parses the
JSON it returns.

## Data flow

```
audio file
   │  stage1: ffmpeg/soundfile decode + Demucs htdemucs_6s
   ▼
6 stems (vocals, drums, bass, guitar, piano, other)
   │  stage2: librosa → tempo, beats, bars, key      (uses mix + vocals)
   │  stage3: Basic Pitch → piano/guitar/other/bass   (drums: onset+centroid → GM ch10)
   │  stage4: pyin → lead vocal notes; Basic Pitch → harmonies; whisper → lyrics
   ▼
score_summary.json (stdout + disk)  ·  full_score.mid  ·  full_score.musicxml
```

## Why two separate runtimes (Hermes core vs worker .venv)

The pipeline needs torch, torchaudio, demucs, basic_pitch, librosa, torchcrepe,
faster-whisper, pretty_midi, music21 — a heavy, mutually-pinned stack with
native binaries. Loading those into Hermes' core Python risks breaking the agent
itself (version conflicts, native segfaults) and bloats its memory. So the worker
runs as a **subprocess with the venv interpreter**; the plugin handler reads only
the compact JSON the worker prints to stdout. Heavy libs are never imported into
`__init__.py`.

## Stage order matters for memory (Intel Mac / 16GB)

faster-whisper (ctranslate2) segfaults or OOM-kills when loaded *after* torch +
demucs + basic-pitch have filled the parent process. So the worker runs whisper
**immediately after separation, before any heavy model loads**, stores the lyric
words, and passes them downstream. This keeps peak memory low enough to finish
2-3 minute commercial tracks on 16 GB hosts. (torchcrepe's `full.pth` also hangs
on Intel-Mac CPU inference, so it's behind a 30s watchdog with librosa `pyin` as
the reliable primary.)

## In-context compactness

The worker writes progress markers to `stderr` and only the structural JSON to
`stdout`. The plugin handler parses the **last non-empty stdout line** and returns
a trimmed envelope (artifact paths + analysis + first N lyric words), so the agent
can decide whether to load the full score without being flooded by raw MIDI events.

## Transport & lifecycle

- `plugin.yaml` declares the tool, config defaults and optional env.
- `__init__.py` `register(ctx)` wires `listen_to_music` (check_fn gates on the
  venv existing), the `/listen` slash command, and the `music-listener:music-listener`
  skill.
- Settings overrides are read from `config.yaml` (`plugins.entries.*.settings`)
  via the authoritative `load_config_readonly()` — never private ctx internals.