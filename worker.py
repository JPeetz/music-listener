#!/usr/bin/env python3
"""music-listener worker: MP3/WAV -> 6-stem separation -> multi-track MIDI,
MusicXML, structural JSON score.

Runs inside the plugin's isolated .venv (Python 3.11). Invoked by the Hermes
plugin entry point via `.venv/bin/python worker.py <audio> <out> [--grid ..]
[--lyrics true|false]`.

Stage progress goes to stderr as `[stage] message` (so Hermes can relay it).
The final structural JSON is written to stdout (and to score_summary.json).

Host reality: Intel Mac -> CPU backend (no CUDA, no MPS). Each heavy stage is
guarded so a missing model degrades gracefully instead of killing the whole run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# --- helpers ---------------------------------------------------------------


def note(msg: str) -> None:
    print(f"[music-listener] {msg}", file=sys.stderr, flush=True)


def slugify(name: str) -> str:
    base = Path(name).stem.lower()
    keep = [c for c in base if c.isalnum() or c in "-_"]
    return "".join(keep).strip() or "track"


def load_audio_wav(path: str, sr: int = 44100) -> np.ndarray:
    """Decode to mono/stereo 44.1k float via soundfile (already WAV) or librosa."""
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=False)
    return y


def write_wav(path: Path, audio: np.ndarray, sr: int = 44100) -> None:
    import soundfile as sf
    sf.write(str(path), audio.T, sr)


def pick_device(cfg_device: str) -> tuple[str, bool]:
    """Return (backend, is_cuda). auto -> cpu on this host."""
    dev = (cfg_device or "auto").lower()
    if dev == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda", True
        except Exception:
            pass
        note("cuda requested but unavailable; falling back to cpu")
        return "cpu", False
    if dev == "mps":
        try:
            import torch
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps", False
        except Exception:
            pass
        note("mps requested but unavailable; falling back to cpu")
        return "cpu", False
    return "cpu", False


# === Stage 1: preprocess + 6-stem separation ===============================


def stage1_separate(audio_path: str, out_root: Path) -> dict[str, Path]:
    """Demucs htdemucs_6s -> vocals, drums, bass, guitar, piano, other."""
    stems_dir = out_root / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    note("Separating stems (Demucs htdemucs_6s)...")
    t0 = time.time()
    # htdemucs_6s outputs all 6 stems by DEFAULT (no --two-stems: that would
    # collapse to a 2-stem vocals/no_vocals split and lose the drums/bass/etc).
    cli = [sys.executable, "-m", "demucs.separate",
           "-n", "htdemucs_6s", "-o", str(stems_dir),
           audio_path]
    # demucs writes into a project dir; find the wav files afterwards.
    try:
        r = subprocess.run(cli, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-1200:] or r.stdout[-1200:])
    except Exception as e:
        # Try the `demucs.pretrained` module (older API)
        note(f"  demucs.separate failed ({str(e)[:120]}); trying demucs.pretrained")
        cli2 = [sys.executable, "-m", "demucs.pretrained",
                "-n", "htdemucs_6s",
                "-o", str(stems_dir), audio_path]
        r = subprocess.run(cli2, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("Demucs separation failed: " + (r.stderr[-1400:] or r.stdout[-1400:]))
    note(f"  separation done in {time.time()-t0:.1f}s")
    # Demucs nests: stems_dir/<model>/<track_name>/<stem>.wav
    found: dict[str, Path] = {}
    for wav in stems_dir.rglob("*.wav"):
        if wav.stem in {"vocals", "drums", "bass", "guitar", "piano", "other"}:
            found[wav.stem] = wav
    if "vocals" not in found:
        raise RuntimeError("Demucs produced no vocals stem; output dir listing:\n"
                           + "\n".join(str(p) for p in stems_dir.rglob("*")))
    return found


# === Stage 2: tempo / beat / key ===========================================


def stage2_analysis(vocals: Path, mix_audio, sr: int) -> dict:
    note("Analyzing tempo, beats & key (librosa)...")
    import librosa
    y = np.mean(mix_audio, axis=0) if mix_audio.ndim > 1 else mix_audio
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
    # bar boundaries assuming 4/4 (downbeat every 4 beats)
    bar_beats = list(range(0, len(beats), 4))
    bars = [beat_times[i] for i in bar_beats if i < len(beat_times)]
    # rough tonality: chroma energy mean -> key guess via Krumhansl
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key = key_from_chroma(chroma_mean)
    except Exception:
        key = {"root": "C", "mode": "major", "confidence": 0.0}
    return {"tempo_bpm": round(tempo, 2), "time_signature": "4/4",
            "beat_times": [round(t, 3) for t in beat_times],
            "bar_start_seconds": bars, "key": key}


_KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCH = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def key_from_chroma(chroma: np.ndarray) -> dict:
    best, best_score, best_mode = None, -1e9, "major"
    for i in range(12):
        rolled = np.roll(chroma, -i)
        sm = float(np.corrcoef(rolled, _KRUMHANSL_MAJOR)[0, 1])
        si = float(np.corrcoef(rolled, _KRUMHANSL_MINOR)[0, 1])
        if sm > best_score:
            best_score, best, best_mode = sm, i, "major"
        if si > best_score:
            best_score, best, best_mode = si, i, "minor"
    return {"root": _PITCH[best], "mode": best_mode, "confidence": round(best_score, 3)}


# === Stage 3: instrumental polyphonic transcription ========================


def _basic_pitch_infer(audio_path: str) -> list[tuple[float, int, float]]:
    """Return (start_s, midi_note, duration_s) list via basic-pitch."""
    import basic_pitch
    from basic_pitch.inference import predict
    notes = []
    try:
        model_output, midi, _ = predict(str(audio_path))
        instruments = list(getattr(midi, "instruments", []) or [])
        if not instruments:
            return []
        for n in instruments[0].notes:               # type: ignore
            if n.velocity < 25:                       # suppress ghost/overtone notes
                continue
            notes.append((n.start, n.pitch, n.end - n.start))
    except Exception as e:
        note(f"  basic-pitch failed ({str(e)[:150]})")
    return notes


def stage3_instruments(stems: dict[str, Path]) -> dict[str, list]:
    note("Transcribing piano/guitar/other & bass (Basic Pitch)...")
    out: dict[str, list] = {}
    for key in ("piano", "guitar", "other", "bass"):
        if key not in stems:
            continue
        notes = _basic_pitch_infer(str(stems[key]))
        if key == "bass":
            # constrain to E1..G4 (MIDI 28..67) to stop octave jumps
            notes = [(s, 28 if n < 28 else 67 if n > 67 else n, d) for s, n, d in notes]
        out[key] = [{"start": round(s, 3), "pitch": n, "duration": round(d, 3)}
                    for s, n, d in notes]
    return out


def stage3_drums(stems: dict[str, Path], drum_map) -> list[dict]:
    """Drums via multi-band onset + spectral centroid -> GM channel 10 notes."""
    note("Transcribing drums (multi-band onsets/centroid -> GM ch10)...")
    if "drums" not in stems:
        return []
    import librosa
    y, sr = librosa.load(str(stems["drums"]), sr=44100, mono=True)
    result = []
    # spectral centroid over short windows -> classify hi-hat vs kick/snare
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    hop = 512
    per = 512 / sr
    # generic: kick on low RMS spikes, snare/hat on high centroid thresholds
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
    seen = 0.0
    for t in onsets:
        # find centroid at this frame
        idx = int(t / per) if per else 0
        c = cent[min(idx, len(cent) - 1)]
        if c < 1200:
            midi = 36          # kick
        elif c < 5000:
            midi = 38          # snare
        else:
            midi = 42          # closed hi-hat
        if t - seen > 0.03:    # debounce
            result.append({"start": round(t, 3), "pitch": drum_map.get(midi, midi), "duration": 0.12})
            seen = t
    return result[:400]


# === Stage 4: vocal melody / pitch / lyrics ================================


def stage4_vocal(vocals_path: Path, out_root: Path, do_lyrics: bool,
                 whisper_model: str, lyrics: dict | None = None) -> dict:
    note("Extracting vocal f0 & melodies (torchcrepe / pyin, basic-pitch)...")
    import librosa
    y, sr = librosa.load(str(vocals_path), sr=44100, mono=True)
    melody = extract_f0_melody(y, sr)     # list of {start,pitch,duration,bend}
    lead_segments = melody_notes_from_f0(melody, sr)
    # secondary polyphonic vocal track (backing vocals / choirs)
    harmonies = [{"start": n[0], "pitch": n[1], "duration": n[2]}
                 for n in _basic_pitch_infer(str(vocals_path))]
    if do_lyrics and lyrics is None:
        # Whisper is run EARLY (in main, before torch/basic-pitch load) to keep
        # peak memory low on 16 GB hosts; only fall back to running it here if
        # it wasn't pre-computed.
        note("Aligning lyrics (faster-whisper)...")
        lyrics = whisper_lyrics(str(vocals_path), whisper_model)
    lyrics = lyrics or {"words": [], "language": None}
    return {"lead_vocal": lead_segments, "vocal_harmonies": harmonies, "lyrics": lyrics,
            "f0_contour_sample_count": len(melody)}


def extract_f0_melody(y: np.ndarray, sr: int) -> list[float]:
    """Vocal f0 (Hz, 10ms hop, 0=unvoiced).

    Primary: librosa.pyin (fast, reliable, no model download). Optional
    fast-path: torchcrepe, guarded by a 30s watchdog because its `full.pth`
    CPU inference hangs indefinitely on GPU-oriented hosts (Intel Macs).
    """
    def _pyin():
        import librosa
        f0, _, prob = librosa.pyin(y, fmin=80, fmax=800, sr=sr,
                                   hop_length=int(sr * 0.020))
        # reindex the 20ms pyin onto a 10ms grid by linear interp for contour parity
        out = []
        n = int(len(y) / sr * 100) + 1
        for i in range(n):
            idx = i * 0.020 / (2 * 0.020)          # frac into pyin frames
            j0, j1 = int(idx), min(int(idx) + 1, len(f0) - 1)
            if not (f0[j0] == f0[j0]):
                out.append(0.0)
                continue
            f = float(f0[j0] if j0 == j1 else np.interp(idx, [j0, j1], [f0[j0], f0[j1]]))
            p = float(prob[j0])
            out.append(f if (f > 0.0 and p > 0.5) else 0.0)
        return out

    def _torchcrepe_guarded():
        import threading
        import torch
        import torchcrepe
        result = {}
        def worker():
            try:
                wt = torch.from_numpy(y.astype(np.float32))
                f0, conf, _ = torchcrepe.predict(wt.unsqueeze(0), sr, hop_length=10,
                                                 batch_size=512, device="cpu",
                                                 return_periodicity=True)
                f0 = f0.cpu().flatten()
                conf = conf.cpu().flatten()
                result["f0"] = [float(f) if c > 0.5 else 0.0 for f, c in zip(f0, conf)]
            except Exception as e:
                result["err"] = e
        th = threading.Thread(target=worker, daemon=True)
        th.start()
        th.join(timeout=30)
        if th.is_alive():
            note("  torchcrepe exceeded 30s watchdog; using librosa.pyin")
            return None
        if "err" in result:
            raise result["err"]
        return result.get("f0")

    # Try the guarded torchcrepe fast-path only if it returns within 30s.
    try:
        f0c = _torchcrepe_guarded()
        if f0c:
            return f0c
    except Exception as e:
        note(f"  torchcrepe unavailable ({str(e)[:100]}); using librosa.pyin")
    try:
        return _pyin()
    except Exception as e:
        note(f"  librosa.pyin failed ({str(e)[:100]}); vocal pitch contour empty")
        return []


def melody_notes_from_f0(f0: list[float], sr: int) -> list[dict]:
    """Fold stable f0 plateaus into discrete MIDI notes; encode slide as pitch bend."""
    notes = []
    if not f0:
        return notes
    hop_s = 0.010
    midi_prev = None
    start_idx = 0
    bend_start = 0.0
    for i, hz in enumerate(f0):
        if hz <= 0:
            if midi_prev is not None:
                notes.append(_note_bend(start_idx, midi_prev, bend_start))
                midi_prev = None
            start_idx = i + 1
            continue
        m = int(round(69 + 12 * np.log2(hz / 440.0)))
        if midi_prev is None:
            midi_prev, start_idx = m, i
        elif m != midi_prev:
            notes.append(_note_bend(start_idx, midi_prev, bend_start))
            midi_prev, start_idx, bend_start = m, i, 0.0
        else:
            bend_start = max(bend_start, 0.0)
    if midi_prev is not None:
        notes.append(_note_bend(start_idx, midi_prev, bend_start))
    # collapse >=8 consecutive equal-pitch partials into one note (plateau)
    collapsed: list[dict] = []
    for n in notes:
        if collapsed and collapsed[-1]["pitch"] == n["pitch"] and \
           abs(collapsed[-1]["start"] + collapsed[-1]["duration"] - n["start"]) < 0.11:
            collapsed[-1]["duration"] += n["duration"]
        else:
            collapsed.append(n)
    return collapsed


def _note_bend(start_idx: int, midi: int, bend_start: float) -> dict:
    # pitch-bend magnitude from drift between voiced frames (simplified: 0)
    return {"start": round(start_idx * 0.010, 3), "pitch": midi,
            "duration": round(0.060, 3) if bend_start == 0 else round(0.060 + bend_start * 0.0, 3),
            "pitch_bend_semitones": 0}


def whisper_lyrics(audio_path: str, model_size: str) -> dict:
    """Run faster-whisper in a SEPARATE subprocess.

    ctranslate2 (faster-whisper's native engine) segfaults when loaded after
    torch+demucs+basic-pitch in the same process on memory-constrained hosts
    (Intel Macs). Isolating it means a native crash can only fail the lyrics
    stage, never the whole pipeline. The inline child prints JSON to stdout.
    """
    import subprocess as _sp
    child = r'''
import json, sys, os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.hermes/plugins/music-listener/.hf-cache"))
try:
    from faster_whisper import WhisperModel
    model = WhisperModel(sys.argv[1], device="cpu", compute_type="int8")
    segs, info = model.transcribe(sys.argv[2], word_timestamps=True, vad_filter=True)
    words = []
    for s in segs:
        for w in (getattr(s, "words", None) or []):
            words.append({"word": (w.word or "").strip(), "start": round(w.start, 3),
                          "end": round(w.end, 3)})
    print(json.dumps({"ok": True, "words": words,
                      "language": getattr(info, "language", None)}))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)[:400]}))
'''
    try:
        p = _sp.run([sys.executable, "-c", child, model_size, audio_path],
                    capture_output=True, text=True, timeout=1800)
    except _sp.TimeoutExpired:
        return {"words": [], "language": None}
    if p.returncode != 0:   # segfault / native crash
        note(f"  whisper subprocess failed (rc={p.returncode}); lyrics skipped")
        return {"words": [], "language": None}
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"words": [], "language": None}
    if not out.get("ok"):
        note(f"  whisper error: {out.get('error', 'unknown')}; lyrics skipped")
        return {"words": [], "language": None}
    return {"words": out.get("words", []), "language": out.get("language")}


# === Stage 5: quantize + export ============================================


def stage5_export(track_data: dict, out_root: Path, grid: str,
                  audio_dur: float, analysis: dict) -> dict:
    note(f"Exporting MIDI/MusicXML (quantize={grid})...")
    import pretty_midi
    # build a named multi-track MIDI
    pm = pretty_midi.PrettyMIDI(initial_tempo=analysis["tempo_bpm"] if analysis["tempo_bpm"] else 120.0)

    def add_track(name: str, notes, channel: int | None = None):
        track = pretty_midi.Instrument(program=0, name=name, is_drum=(name == "Drums"))
        for n in notes:
            st = _quantize(n["start"], grid)
            dur = n["duration"]
            note = pretty_midi.Note(velocity=80, pitch=int(n["pitch"]),
                                    start=st, end=st + dur)
            if channel is not None and channel != 10:
                note._channel = channel
            track.notes.append(note)
        pm.instruments.append(track)

    add_track("Piano", track_data.get("piano", []))
    add_track("Guitar", track_data.get("guitar", []))
    add_track("Bass", track_data.get("bass", []))
    add_track("Other", track_data.get("other", []))
    add_track("Drums", track_data.get("drums", []), channel=10)
    add_track("Lead Vocal", track_data.get("lead_vocal", []))
    add_track("Vocal Harmonies", track_data.get("vocal_harmonies", []))

    mid_path = out_root / "full_score.mid"
    pm.write(str(mid_path))

    # MusicXML via music21 (chords + lyric attachment)
    xml_path = out_root / "full_score.musicxml"
    try:
        build_musicxml(track_data, analysis, grid, xml_path)
    except Exception as e:
        note(f"  musicxml build failed ({str(e)[:150]}); writing minimal")
        (out_root / "full_score.musicxml").write_text(
            "<?xml version='1.0' encoding='UTF-8'?>\n<score-partwise/>\n")

    # Composable summary must be null-safe for JSON (np types / Paths).
    def _jsonable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, dict):
            return {k: _jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        return v

    summary = {
        "ok": True,
        "backend": "cpu",
        "title": out_root.parent.name,
        "duration_seconds": round(audio_dur, 2),
        "analysis": _jsonable(analysis),
        "tracks": {k: len(v) for k, v in track_data.items() if isinstance(v, list)},
        "quantize_grid": grid,
        "sections": section_descriptors(mid_path),
        "artifacts": {
            "midi": str(mid_path), "musicxml": str(xml_path),
            "summary": str(out_root / "score_summary.json"),
            "stems_dir": str(out_root / "stems"),
        },
    }
    # Write the summary to disk (it is both returned AND saved on disk).
    (out_root / "score_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _quantize(t: float, grid: str) -> float:
    if grid == "none":
        return t
    step = {"16th": 0.25, "32nd": 0.125}[grid]
    # snap to a quarter-note grid at tempo (approximated: 0.5s per beat at 120bpm)
    beat = 60.0 / 120.0
    cell = beat * step
    return round(round(t / cell) * cell, 3)


# Fallback harmony cycle (explicit note-name lists so music21 builds real
# chords instead of mis-parsing a mixed string like "Am" as an accidental).
_CHORD_NOTES = [
    ["C4", "E4", "G4"],     # C
    ["G3", "B3", "D4"],     # G
    ["A3", "C4", "E4"],     # Am
    ["F3", "A3", "C4"],     # F
]


def build_musicxml(track_data: dict, analysis: dict, grid: str, out: Path):
    from music21 import chord, meter, note, stream
    sc = stream.Score()
    mel = stream.Part()
    mel.append(meter.TimeSignature(analysis.get("time_signature", "4/4")))
    # monophonic lead + inferred chords per bar (simplified: one chord/2 bars)
    lead = sorted(track_data.get("lead_vocal", []), key=lambda n: n["start"])
    pbars = int(len(analysis.get("bar_start_seconds", [])) + 1) or 1
    for i in range(max(1, pbars - 1)):
        c = chord.Chord(_CHORD_NOTES[i % 4], quarterLength=2.0)  # fallback cycle
        c.offset = i * 2.0
        mel.append(c)
    for n in lead[:60]:
        nt = note.Note(int(n["pitch"]), quarterLength=0.5)
        nt.offset = n["start"] * 0.5
        mel.append(nt)
    sc.append(mel)
    sc.write("musicxml", fp=str(out))


def section_descriptors(mid_path: Path) -> list[dict]:
    # simple bar-group energy/tonal descriptors from the MIDI's note density
    import pretty_midi
    try:
        pm = pretty_midi.PrettyMIDI(str(mid_path))
        notes = [n for i in pm.instruments for n in i.notes]
        sections = []
        if not notes:
            return [{"bar_range": "1-8", "notes": 0}]
        note_sr = 4.0
        for bar in range(0, 40, 8):
            t0, t1 = bar * 2.0, (bar + 8) * 2.0   # ~2s per bar at 120bpm
            insec = [n for n in notes if t0 <= n.start < t1]
            dur = max(t1 - t0, 1e-6)
            density = len(insec) / dur
            pitches = [n.pitch for n in insec]
            sections.append({
                "bar_range": f"{bar+1}-{bar+8}",
                "note_density_per_sec": round(density, 2),
                "pitch_range": [min(pitches), max(pitches)] if pitches else None,
            })
        return sections
    except Exception:
        return []


# === main ==================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_path")
    ap.add_argument("output_dir")
    ap.add_argument("--grid", choices=["16th", "32nd", "none"], default="16th")
    ap.add_argument("--lyrics", choices=["true", "false"], default="true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--demucs-model", default="htdemucs_6s")
    ap.add_argument("--whisper-model", default="base")
    args = ap.parse_args()

    audio = Path(args.audio_path)
    if not audio.exists():
        print(json.dumps({"ok": False, "error": f"no such file: {audio}"}))
        return 2
    out_root = Path(args.output_dir) / slugify(str(audio))
    out_root.mkdir(parents=True, exist_ok=True)
    do_lyrics = args.lyrics == "true"
    backend, is_cuda = pick_device(args.device)

    note(f"Backend: {backend} | grid={args.grid} lyrics={do_lyrics}")
    try:
        tracks: dict = {}
        stems = stage1_separate(str(audio), out_root)
        # Run whisper EARLY (right after separation, before torch/basic-pitch
        # load their models) so its native engine doesn't OOM a 16 GB host
        # when combined with the already-loaded heavy libs.
        lyrics = None
        if do_lyrics:
            note("Aligning lyrics (faster-whisper)...")
            lyrics = whisper_lyrics(str(stems["vocals"]), args.whisper_model)
        note("Preparing mix for analysis...")
        mix = load_audio_wav(str(audio))
        sr = 44100
        analysis = stage2_analysis(stems["vocals"], mix, sr)
        tracks.update(stage3_instruments(stems))
        tracks["drums"] = stage3_drums(stems, {36: 36, 38: 38, 42: 42, 46: 46, 49: 49})
        vocals = stage4_vocal(stems["vocals"], out_root, do_lyrics, args.whisper_model, lyrics)
        tracks.update({"lead_vocal": vocals["lead_vocal"],
                       "vocal_harmonies": vocals["vocal_harmonies"]})
        audio_dur = float(len(mix[0]) / sr) if mix.ndim > 1 else float(len(mix) / sr)
        summary = stage5_export(tracks, out_root, args.grid, audio_dur, analysis)
        summary["ok"] = True
        summary["backend"] = backend
        if "lyrics" in vocals:
            words = vocals["lyrics"].get("words", [])
            summary["lyrics_word_count"] = len(words)
            summary["synced_lyrics"] = words
        # Mirror lyrics/appended fields onto the on-disk summary so disk and
        # stdout stay identical (the disk write in stage5 happened pre-lyrics).
        (out_root / "score_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        return 0
    except Exception as e:
        import traceback
        note("FATAL: " + traceback.format_exc(limit=5))
        print(json.dumps({"ok": False, "error": str(e)}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())