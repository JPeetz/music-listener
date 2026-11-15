"""music-listener: reconstruct MP3/WAV into multi-track MIDI, MusicXML and JSON.

Launches the standalone `worker.py` inside the plugin's own .venv so the heavy
audio/ML stack (torch/demucs/basic-pitch/whisper) never touches Hermes' core
Python. Lives at ~/.hermes/plugins/music-listener/.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PLUGIN_DIR = Path(__file__).parent
VENV_PY = PLUGIN_DIR / ".venv" / "bin" / "python"
WORKER = PLUGIN_DIR / "worker.py"

TOOL_SCHEMA = {
    "name": "listen_to_music",
    "description": (
        "Listen to an MP3/WAV file by reconstructing every instrumental track, "
        "the vocal melody, pitch bends and timed lyrics into a multi-track MIDI, "
        "a MusicXML score and a compact structural JSON summary. Returns the paths "
        "to full_score.mid, full_score.musicxml and score_summary.json plus a "
        "concise musical breakdown (sections, bars, chords, vocal range, lyrics). "
        "Long-running: stems separation + transcription usually take 1-5 min for "
        "a full song. Use for transcribing, analyzing structure, extracting "
        "vocals/melody, arranging or re-harmonizing existing audio."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": "Path to an MP3, WAV, FLAC or M4A audio file.",
            },
            "quantize_grid": {
                "type": "string",
                "enum": ["16th", "32nd", "none"],
                "default": "16th",
                "description": "Rhythmic grid to snap note onsets to. 16th or 32nd for score/arrangement; none keeps raw timing.",
            },
            "extract_lyrics": {
                "type": "boolean",
                "default": True,
                "description": "Run faster-whisper word-level lyric alignment (adds time but attaches lyrics to notes).",
            },
        },
        "required": ["audio_path"],
    },
}


def _settings() -> dict:
    # config_schema defaults apply; read overrides from plugins.entries.<id>.settings
    # via the authoritative config, never via private ctx helpers.
    try:
        from hermes_cli.plugins_manifest import load_manifest
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        entries = (cfg.get("plugins", {}).get("entries", {}) or {})
        for key in ("music-listener", "music_listener"):
            if key in entries:
                return dict(entries[key].get("settings", {}) or {})
    except Exception:
        pass
    return {}


def _resolved_output_dir() -> str:
    try:
        from agent.runtime_cwd import resolve_agent_cwd
        base = str(resolve_agent_cwd())
    except Exception:
        base = os.getcwd()
    out = _settings().get("output_dir", "transcriptions")
    if out.startswith("/"):
        return out
    return str(Path(base) / out)


def _check_available() -> bool:
    return VENV_PY.exists() and WORKER.exists()


def _run_worker(audio_path: str, grid: str, lyrics: bool, out_dir: str) -> dict:
    s = _settings()
    cmd = [str(VENV_PY), str(WORKER), audio_path, out_dir,
           "--grid", grid, "--lyrics", "true" if lyrics else "false"]
    if s.get("device"):
        cmd += ["--device", s["device"]]
    if s.get("demucs_model"):
        cmd += ["--demucs-model", s["demucs_model"]]
    if s.get("whisper_model"):
        cmd += ["--whisper-model", s["whisper_model"]]
    # Hard cap so a huge/crafted audio file can't pin the CPU forever.
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return json.dumps({"ok": False, "error": "worker exceeded 60m; aborted",
                           "audio_path": audio_path})
    try:
        last_json = proc.stdout.strip().splitlines()[-1]
        result = json.loads(last_json)
    except Exception:
        result = {"ok": False, "error": "worker produced no JSON",
                  "stderr_tail": (proc.stderr or "")[-1500:]}
    return result


def _handle(args: dict) -> str:
    audio = args.get("audio_path")
    if not audio:
        return json.dumps({"ok": False, "error": "audio_path required"})
    grid = args.get("quantize_grid", "16th")
    lyrics = bool(args.get("extract_lyrics", True))
    out_dir = _resolved_output_dir()
    result = _run_worker(audio, grid, lyrics, out_dir)
    if result.get("ok"):
        return json.dumps({
            "ok": True,
            "message": "Transcribed. Use the skill `music-listener:music-listener` on the JSON paths to reason over the song.",
            "artifacts": result.get("artifacts", {}),
            "analysis": result.get("analysis", {}),
            "tracks": result.get("tracks", {}),
            "synced_lyrics": result.get("synced_lyrics", [])[:30],
        }, indent=2)
    return json.dumps(result)


def _listen_command(raw: str) -> str:
    parts = raw.strip().split()
    audio = parts[0] if parts else ""
    if not audio:
        return "Usage: /listen <audio_path> [--grid 16th|32nd|none] [--no-lyrics]"
    grid = "16th"
    lyrics = True
    rest = parts[1:]
    for i, tok in enumerate(rest):
        if tok == "--grid" and i + 1 < len(rest):
            grid = rest[i + 1] if rest[i + 1] in {"16th", "32nd", "none"} else "16th"
        elif tok == "--no-lyrics":
            lyrics = False
    result = _run_worker(audio, grid, lyrics, "transcriptions")
    return json.dumps({"ok": result.get("ok", False),
                       **({k: result[k] for k in ("artifacts", "analysis", "error")
                           if k in result})}, indent=2)


def register(ctx):
    ctx.register_tool(
        name="listen_to_music",
        toolset="audio_music",
        schema=TOOL_SCHEMA,
        handler=_handle,
        check_fn=_check_available,
        is_async=False,
        emoji="🎵",
    )
    ctx.register_command(
        name="listen",
        handler=_listen_command,
        description="Listen to an audio file and transcribe it to MIDI/MusicXML.",
        args_hint="<audio_path> [--grid 16th|32nd|none] [--no-lyrics]",
    )
    ctx.register_skill(
        "music-listener",
        PLUGIN_DIR / "skills" / "music-listener" / "SKILL.md",
        description="Use the listen_to_music transcription output (MIDI/MusicXML/JSON) for analysis and arrangement.",
    )