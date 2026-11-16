---
name: music-listener
description: "Read the listen_to_music transcription output (score_summary.json, full_score.mid, full_score.musicxml) for analysis, arrangement and re-harmonization."
version: 1.0.0
author: music-listener
license: MIT
metadata:
  hermes:
    tags: [music, transcription, midi, musicxml, score, analysis]
---

# Music Listener — reading & using a transcription

After `listen_to_music` runs on an MP3/WAV you get three artifacts per track at
`<output_dir>/<track_slug>/`:

- `full_score.mid` — Type 1 multi-track MIDI: Piano, Guitar, Bass, Other, Drums
  (GM ch10), Lead Vocal, Vocal Harmonies. Each track is named.
- `full_score.musicxml` — score with bar-by-bar chord symbols and (when lyrics
  were enabled) syllable-aligned `<lyric>` elements on the vocal part.
- `score_summary.json` — the compact structural summary the tool returns. Use
  this first so you rarely need to load the raw MIDI.

## 1. Read `score_summary.json` first

The JSON is designed to be small enough to reason over in-context. Key fields:

| Field | Meaning |
|---|---|
| `analysis.tempo_bpm` | global tempo |
| `analysis.time_signature` | meter detected (4/4 unless otherwise) |
| `analysis.key` | `{root, mode, confidence}` from chroma/Krumhansl |
| `analysis.bar_start_seconds` | second boundaries of the bars |
| `tracks` | note counts per transcribed part |
| `sections` | bar-group descriptors (note density, pitch range) every 8 bars |
| `synced_lyrics` | `[{word, start, end}]` – word-level lyric timeline |
| `artifacts` | absolute paths to the three outputs + stems dir |

To understand the song, read: tempo → key → per-track note counts (which parts
are present?) → sections (what changes every 8 bars) → the lyric words and their
time alignment → the lead vocal note range.

## 1b. Reading the lead vocal correctly

Do NOT judge a transcription by note length, repetition or small intervals - short and
repetitive notes are the NORMAL signature of a hook, and both failure modes below are
silent (the output looks like plausible data either way).

**Non-lexical vocals are melody, not noise.** `na-na-na`, `da-da-da`, `hey-hey-hey`,
`di-da-di-da` and syllable-per-note chant are genuine melodic material. Their signature
is short notes in repeated A-B-A-B alternation. Filtering short or repetitious notes
deletes the hook - the most memorable material in the track.

**Diagnose a bad transcription by measurement, not impression.** On the Lead Vocal track
count: distinct note durations, onsets carrying more than one note, overlapping pairs.
A healthy monophonic lead has many distinct durations, zero multi-note onsets, zero
overlaps. Exactly ONE distinct duration means every real note length was discarded.
Multi-note onsets a semitone apart are NOT tracker jitter by default - check the
quantiser and the note-boundary code before blaming the pitch tracker.

**Verify register with pyin/MIDI pitches, never an FFT peak-pick.** An FFT argmax
returns the loudest HARMONIC, so a male voice at ~110 Hz with a strong 3rd harmonic
reads ~330 Hz and is mislabelled female. Read `Lead Vocal` pitches from `full_score.mid`.
The Demucs vocal stem holds ALL voices, so where a choir answers the reading is the
choir, not the soloist.

## 2. Query specific bars / motifs with Python

Use the plugin's venv python (it has pretty_midi + music21) to query without
flooding context:

```bash
~/.hermes/plugins/music-listener/.venv/bin/python - <<'PY'
import pretty_midi
pm = pretty_midi.PrettyMIDI("<output_dir>/<track>/full_score.mid")
lead = [i for i in pm.instruments if i.name == "Lead Vocal"][0]
# notes in bar 3 (bar ~2s at 120bpm; recompute from analysis.tempo_bpm if needed)
b0, b1 = 4.0, 6.0
motif = [n for n in lead.notes if b0 <= n.start < b1]
print("bar3 motif pitches:", [n.pitch for n in motif])
PY
```

`pretty_midi.PrettyMIDI(str(score))` lets you transpose (shift all note pitches),
extract motifs per bar/time window, or regenerate arrangements by editing
`Instrument.notes` and calling `pm.write("new.mid")`.

## 3. Re-harmonize / analyze chords with music21

```python
from music21 import converter
sc = converter.parse("<output_dir>/<track>/full_score.musicxml")
for part in sc.parts:
    ch = part.recurse().getElementsByClass('Chord')
    print([c.pitchedCommonName for c in ch][:12])
```

To re-harmonize: load as music21, replace the chord stream with a new progression
(`Am - F - C - G`, etc.), adjust the melody's scale tones accordingly, and
re-export via `sc.write('midi', fp='reharm.mid')`.

## 4. Practical extraction uses

- **Vocal melody** → Lead Vocal track (monophonic notes with pitch-bend events on
  slides). Good for cover arrangements, karaoke backing, or melody-only MIDI.
- **Instruments** → Piano/Guitar/Bass/Other tracks; use them for a backing-track
  MIDI or to learn a song's chord voicings.
- **Vocal harmonies** → Vocal Harmonies track (basic-pitch polyphonic capture of
  stacked backing vox/choirs).
- **Timings** → `synced_lyrics` for karaoke-style lyric sync; `sections` to find
  the verse/chorus shape.

## Pitfalls / honesty

- Transcription is imperfect. On dense production audio, low-velocity ghost notes
  are filtered (velocity < 25) but harmonics can still appear — treat pitch
  lists as a strong guide, not a note-perfect score.
- Tempo/bar boundaries are assumed ~4/4 (120bpm) unless meter detection differs;
  offset scales are approximate. For exact work verify against the audio.
- Lyric word timestamps come from faster-whisper and can drift on instrumental
  heavy or highly-mixed tracks. They are best-effort alignment.
- MIDI pitch values are 0-127 (middle C = 60). Drum notes follow GM channel 10
  (36 kk, 38 snr, 42 CHH, 46 OHH, 49 crash).