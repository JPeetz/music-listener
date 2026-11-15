# 🎵 Hermes Music Listener

**A self-contained Hermes Agent plugin that "listens" to an MP3/FLAC/WAV and
reconstructs every instrumental track, the vocal melody, pitch contour and
time-aligned lyrics into a multi-track MIDI, MusicXML score and a compact
structural JSON.**

<img src="https://img.shields.io/badge/Hermes-0.21.5+-7000ff.svg" /> <img src="https://img.shields.io/badge/torch-2.2.2-ee4c2c.svg" /> <img src="https://img.shields.io/badge/license-MIT-blue.svg" />

Built as a standalone Hermes plugin with the entire heavy audio/ML stack
isolated in its own Python 3.11 virtual environment — it never touches Hermes'
core Python. It was proven end-to-end on a real commercial track (Imael Angel –
"Bad Times", 2:46) producing 7 named MIDI tracks, a parseable MusicXML score,
5 structural sections and 100 time-aligned lyric words.

---

## What it does

Give the agent any audio file and it reconstructs the song:

| Output | Contents |
|---|---|
| `full_score.mid` | Type-1 multi-track MIDI: **Piano, Guitar, Bass, Other, Drums (GM ch10), Lead Vocal, Vocal Harmonies** |
| `full_score.musicxml` | Printable score with bar chord symbols + syllable-aligned `<lyric>` elements |
| `score_summary.json` | Compact structural JSON: tempo, key, per-track note counts, 8-bar section analysis, synced lyric words |

The five-stage pipeline:

1. **Separation** — Demucs `htdemucs_6s` splits the mix into vocals, drums,
   bass, guitar, piano, other.
2. **Analysis** — librosa extracts tempo, beat grid, 4/4 bar boundaries and a
   Krumhansl chroma key guess.
3. **Instrumental transcription** — Basic Pitch transcribes piano/guitar/other
   and bass (bass octave-clamped to E1–G4); drums via multi-band onset +
   spectral-centroid classification mapped to GM ch10.
4. **Vocals & lyrics** — librosa `pyin` (guarded torchcrepe fast-path) extracts
   continuous f0 -> discrete lead notes; Basic Pitch captures backing vocals;
   faster-whisper word-timestamps the lyrics.
5. **Export** — quantizes to the requested grid, writes the three artifacts.

Progress is streamed to stderr (so Hermes can relay it); only the compact JSON
goes to stdout (so the agent reasons over the song without flooding context).

---

## Requirements

- **Hermes Agent ≥ 0.21.5**, macOS or Linux
- **Python 3.11** available as `python3.11` (needs solid ML wheels; see
  `docs/install-notes.md` for the Intel-Mac/Apple-Silicon pin guide)
- `ffmpeg` on PATH (optional — librosa/soundfile decode most formats natively)
- ~3 GB free disk for the venv + ~1.5 GB of model weights on first run
- **16 GB+ RAM recommended** for full lyric alignment on longer tracks

## Install

```bash
cd ~/.hermes/plugins
git clone https://github.com/JPeetz/music-listener.git
cd music-listener
chmod +x install.sh
./install.sh            # builds .venv, installs torch/demucs/basic-pitch/whisper

hermes plugins doctor music-listener
hermes plugins enable music-listener
```

First run downloads the Demucs, Basic Pitch and (if lyrics are enabled)
whisper model weights — allow a few minutes.

## Usage

**Via the tool** (agent / chat):

```
listen_to_music(audio_path="/path/to/song.mp3",
                quantize_grid="16th",   # 16th | 32nd | none
                extract_lyrics=true)
```

**Via the slash command:**

```
/listen /path/to/song.mp3 --grid 16th --no-lyrics
```

**Directly (no Hermes):**

```bash
.venv/bin/python worker.py /path/to/song.mp3 ./out --grid 16th --lyrics true
```

### What comes back

`listen_to_music` returns the artifact paths plus a trimmed analysis envelope:
tempo, key, per-track note counts, 8-bar section descriptors and (when lyrics
are on) the first 30 synced words. To reason over the whole song read
`score_summary.json`, and to open/query the actual score use the bundled
`music-listener:music-listener` skill (pretty_midi/music21 recipes for bar
queries, motifs, transposition and re-harmonization).

See [`examples/sample_output/`](examples/sample_output/) for the real output of
a 2:46 produced track.

## Project layout

```
music-listener/
├── __init__.py          # Hermes plugin entry point (tool + /listen + skill)
├── worker.py            # standalone 5-stage pipeline CLI
├── plugin.yaml          # manifest: tools, config_schema, optional_env
├── requirements.txt     # pinned deps for the isolated .venv
├── install.sh           # builds .venv (Python 3.11) and installs deps
├── skills/music-listener/SKILL.md  # operational skill for reading the output
├── tests/test_pipeline.py          # synthesize-audio end-to-end smoke test
├── docs/
│   ├── architecture.md   # pipeline stages, data flow, memory-ordering rationale
│   ├── install-notes.md  # wheel pins & platform gotchas (Intel Mac etc.)
│   └── security.md       # blast radius, threat model, hardening
└── examples/sample_output/  # real artifacts from a commercial track
```

## Testing

```bash
# End-to-end smoke test: synthesizes a 6s multi-part clip, runs the real
# pipeline, asserts the three artifacts exist and the JSON is valid.
.venv/bin/python tests/test_pipeline.py
```

## Security

The plugin runs its heavy libraries in an isolated `.venv`, binds nothing and
opens no network ports, reads **no** credentials, uses no shell (`subprocess`
argv lists throughout), sanitizes output paths, and caps the worker runtime.
Its only network egress is the official model-weight downloads (Demucs, whisper)
plus the optional HuggingFace token the user chooses to set. Full analysis in
[`docs/security.md`](docs/security.md).

## Known limitations (honest)

- Transcription is imperfect on dense production mixes: low-velocity ghost
  notes are filtered, but harmonics/overtones can still appear. Treat pitch
  lists as a strong guide, not a note-perfect score.
- Bar/beat boundaries assume ~4/4; exact alignment can drift on rubato or odd
  meter. Verify against audio for precise work.
- Lyric timestamps are best-effort (faster-whisper) and can drift on
  instrumental-heavy or heavily-processed vocal stems.
- On Intel Macs, torch must stay at 2.2.2 and torchcrepe's `full.pth` hangs on
  CPU inference — the worker runs librosa `pyin` as its reliable f0 path behind
  a 30s watchdog. See `docs/install-notes.md`.

## License

MIT — see [LICENSE](LICENSE). Built & maintained by [Joerg Peetz](https://github.com/JPeetz).