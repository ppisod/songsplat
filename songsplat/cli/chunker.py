"""songsplat chunk - import audio and split into chunks -> .splatchunk"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from songsplat.cli.formats import ChunkInfo, SplatChunk


def run_chunker(
    audio_path: str,
    output: str = "",
    mode: str = "fixed",
    duration: float = 2.0,
    beats_per_chunk: int = 4,
) -> str:
    """Chunk an audio file and save as .splatchunk. Returns the output path."""
    from songsplat.audio.loader import load_audio, is_supported

    audio_path = os.path.abspath(audio_path)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")
    if not is_supported(audio_path):
        raise ValueError(f"Unsupported audio format: {Path(audio_path).suffix}")

    print(f"Loading {os.path.basename(audio_path)}...", flush=True)
    audio, sr = load_audio(audio_path)
    total_duration = len(audio) / sr
    print(f"Duration: {total_duration:.1f}s  SR: {sr}Hz", flush=True)

    sc = SplatChunk(
        source_path=audio_path,
        song_name=Path(audio_path).stem,
        sample_rate=sr,
        duration=total_duration,
        chunk_mode=mode,
        chunk_duration=duration,
        beats_per_chunk=beats_per_chunk,
    )

    if mode == "beat":
        sc.chunks = _beat_chunks(audio, sr, beats_per_chunk, total_duration)
    else:
        sc.chunks = _fixed_chunks(total_duration, duration)

    print(f"Created {len(sc.chunks)} chunks ({mode} mode)", flush=True)

    if not output:
        output = str(Path(os.getcwd()) / (Path(audio_path).stem + ".splatchunk"))
    sc.save(output)
    print(f"Saved to {output}", flush=True)
    return output


def _fixed_chunks(duration: float, chunk_dur: float) -> list[ChunkInfo]:
    chunks = []
    start = 0.0
    i = 0
    while start < duration:
        end = min(start + chunk_dur, duration)
        if end - start < 0.05:
            break
        chunks.append(ChunkInfo(index=i, start=round(start, 6), end=round(end, 6)))
        start = end
        i += 1
    return chunks


def _beat_chunks(audio, sr: int, beats_per_chunk: int, duration: float) -> list[ChunkInfo]:
    try:
        import librosa
    except ImportError:
        raise RuntimeError("librosa is required for beat-aligned chunking")

    _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, units="time")
    all_beats = [float(t) for t in beat_frames]

    if not all_beats or all_beats[0] > 0.01:
        all_beats.insert(0, 0.0)
    if all_beats[-1] < duration - 0.01:
        all_beats.append(duration)

    boundaries = all_beats[::beats_per_chunk]
    if boundaries[-1] < duration - 0.01:
        boundaries.append(duration)

    chunks = []
    for i in range(len(boundaries) - 1):
        chunks.append(ChunkInfo(
            index=i,
            start=round(boundaries[i], 6),
            end=round(boundaries[i + 1], 6),
        ))
    return chunks
