"""Audio loading, waveform caching, and chunking."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from songsplat.core.models import Chunk, Song

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}

# Target sample rate for ML models and beat tracking
TARGET_SR = 22050
WAVEFORM_DISPLAY_SAMPLES = 8000


def is_supported(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def load_audio(path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load audio file, convert to mono, resample to target_sr.

    Returns (samples_float32, sample_rate).
    Raises FileNotFoundError or RuntimeError on failure.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = Path(path).suffix.lower()

    if ext in {".mp3", ".m4a", ".aiff", ".aif"}:
        # soundfile can't read mp3/m4a natively; fall back to librosa which
        # uses audioread/ffmpeg under the hood.
        try:
            import librosa  # type: ignore
            audio, sr = librosa.load(path, sr=target_sr, mono=True)
            return audio.astype(np.float32), sr
        except ImportError:
            raise RuntimeError(
                "librosa is required to load MP3/M4A files. "
                "Install it with: pip install librosa"
            )
    else:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if data.shape[1] > 1:
            audio = data.mean(axis=1)
        else:
            audio = data[:, 0]
        if sr != target_sr:
            try:
                import librosa  # type: ignore
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except ImportError:
                pass  # use original sr; beat tracking may be less accurate
        return audio.astype(np.float32), sr


def build_song_from_path(path: str) -> Song:
    """Create a Song object from an audio file path (does NOT chunk yet)."""
    if not is_supported(path):
        raise ValueError(f"Unsupported audio format: {Path(path).suffix}")

    audio, sr = load_audio(path)
    duration = len(audio) / sr

    song = Song(
        path=os.path.abspath(path),
        name=Path(path).stem,
        sample_rate=sr,
        duration=duration,
        num_channels=1,
    )
    song.waveform_cache = _build_waveform_display(audio)
    return song


def _build_waveform_display(audio: np.ndarray, n_samples: int = WAVEFORM_DISPLAY_SAMPLES) -> list[float]:
    """Downsample audio to exactly n_samples amplitude envelope points."""
    if len(audio) == 0:
        return []
    n_samples = min(n_samples, len(audio))
    segments = np.array_split(np.abs(audio), n_samples)
    frames = [float(seg.max()) if len(seg) > 0 else 0.0 for seg in segments]
    peak = max(frames) if frames else 1.0
    if peak > 0:
        frames = [f / peak for f in frames]
    return frames


def compute_waveform_cache(song: Song) -> None:
    """(Re)compute the display waveform for a song and store it in-place."""
    audio, _ = load_audio(song.path, target_sr=song.sample_rate)
    song.waveform_cache = _build_waveform_display(audio)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_song_fixed(song: Song, chunk_duration: float) -> None:
    """Divide the song into fixed-duration chunks and store on song.chunks."""
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be positive")

    song.chunk_mode = "fixed"
    song.chunk_duration = chunk_duration
    song.chunks = _build_fixed_chunks(song.duration, chunk_duration)


def chunk_song_beats(song: Song, beats_per_chunk: int = 1) -> None:
    """Chunk the song at beat boundaries using librosa beat tracking.

    beats_per_chunk: how many detected beats to group into one chunk (default 1).
    E.g. beats_per_chunk=4 gives one chunk per bar (in 4/4 time).
    """
    if beats_per_chunk < 1:
        beats_per_chunk = 1

    try:
        import librosa  # type: ignore
    except ImportError:
        raise RuntimeError(
            "librosa is required for beat-aligned chunking. "
            "Install it with: pip install librosa"
        )

    audio, sr = load_audio(song.path)
    _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, units="time")
    all_beats: list[float] = [float(t) for t in beat_frames]

    if not all_beats or all_beats[0] > 0.01:
        all_beats.insert(0, 0.0)
    if all_beats[-1] < song.duration - 0.01:
        all_beats.append(song.duration)

    # Group beats: take every beats_per_chunk-th beat as a chunk boundary
    boundaries = all_beats[::beats_per_chunk]
    if boundaries[-1] < song.duration - 0.01:
        boundaries.append(song.duration)

    chunks = []
    for i in range(len(boundaries) - 1):
        chunks.append(Chunk(
            index=i,
            start=round(boundaries[i], 6),
            end=round(boundaries[i + 1], 6),
        ))

    song.chunk_mode = "beat"
    song.chunks = chunks


def _build_fixed_chunks(duration: float, chunk_duration: float) -> list[Chunk]:
    chunks = []
    start = 0.0
    i = 0
    while start < duration:
        end = min(start + chunk_duration, duration)
        if end - start < 0.05:
            break
        chunks.append(Chunk(index=i, start=round(start, 6), end=round(end, 6)))
        start = end
        i += 1
    return chunks


def get_chunk_audio(song: Song, chunk: Chunk, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load and return the audio samples for a single chunk."""
    audio, sr = load_audio(song.path, target_sr=target_sr)
    start_sample = int(chunk.start * sr)
    end_sample = int(chunk.end * sr)
    segment = audio[start_sample:end_sample]
    expected_len = int(chunk.duration * sr)
    if len(segment) < expected_len:
        segment = np.pad(segment, (0, expected_len - len(segment)))
    return segment
