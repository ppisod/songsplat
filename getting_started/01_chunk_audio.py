"""getting_started/01_chunk_audio.py

Load an audio file and split it into time-range chunks.
The result is saved as a .splatchunk file in the current directory.

Usage:
    python 01_chunk_audio.py path/to/song.mp3
"""

import sys
from pathlib import Path

from songsplat.audio.loader import load_audio, chunk_song_fixed, chunk_song_beats, build_song_from_path


def main(audio_path: str) -> None:
    print(f"Loading {Path(audio_path).name} ...")
    song = build_song_from_path(audio_path)
    print(f"  duration : {song.duration:.1f}s")
    print(f"  sample_rate : {song.sample_rate} Hz")

    chunk_song_fixed(song, chunk_duration=2.0)
    print(f"\nFixed chunking (2.0 s) -> {len(song.chunks)} chunks")
    for ck in song.chunks[:5]:
        print(f"  chunk {ck.index:3d}  {ck.start:7.3f}s - {ck.end:7.3f}s  ({ck.duration:.3f}s)")
    if len(song.chunks) > 5:
        print(f"  ... ({len(song.chunks)} total)")

    try:
        chunk_song_beats(song, beats_per_chunk=4)
        print(f"\nBeat chunking (4 beats/chunk) -> {len(song.chunks)} chunks")
        for ck in song.chunks[:5]:
            print(f"  chunk {ck.index:3d}  {ck.start:7.3f}s - {ck.end:7.3f}s")
        if len(song.chunks) > 5:
            print(f"  ... ({len(song.chunks)} total)")
    except RuntimeError as e:
        print(f"\nSkipping beat chunking: {e}")

    print("\nDone. Import this song in the GUI to start labeling.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 01_chunk_audio.py <audio_file>")
        sys.exit(1)
    main(sys.argv[1])
