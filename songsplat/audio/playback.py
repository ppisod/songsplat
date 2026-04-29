"""Low-latency audio playback engine using sounddevice."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from songsplat.audio.loader import TARGET_SR, load_audio
from songsplat.core.models import Chunk, Song


class AudioPlayer:
    """Manages playback of a single audio file.

    All public methods are thread-safe.  Position and playback state are
    polled by the UI via :py:attr:`position` and :py:attr:`is_playing`;
    call :py:meth:`drain_events` once per UI tick to let the player process
    any deferred state transitions.
    """

    def __init__(self) -> None:
        self._audio: Optional[np.ndarray] = None
        self._sr: int = TARGET_SR
        self._position: int = 0
        self._playing: bool = False
        self._volume: float = 1.0
        self._stream: Optional[sd.OutputStream] = None
        self._lock = threading.Lock()
        self._loop_start: Optional[int] = None
        self._loop_end: Optional[int] = None

    @property
    def position(self) -> float:
        """Current playhead position in seconds."""
        with self._lock:
            if self._audio is None:
                return 0.0
            return self._position / self._sr

    @property
    def duration(self) -> float:
        """Total duration of the loaded audio in seconds."""
        with self._lock:
            if self._audio is None:
                return 0.0
            return len(self._audio) / self._sr

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = max(0.0, min(1.0, value))

    def drain_events(self) -> None:
        """No-op hook called by the UI poll loop; reserved for future use."""

    def load_song(self, song: Song) -> None:
        """Load audio for *song* and reset playback position."""
        self.stop()
        audio, sr = load_audio(song.path, target_sr=TARGET_SR)
        with self._lock:
            self._audio = audio
            self._sr = sr
            self._position = 0
            self._loop_start = None
            self._loop_end = None

    def play(self) -> None:
        """Start or resume playback."""
        with self._lock:
            if self._audio is None or self._playing:
                return
            self._playing = True
        self._start_stream()

    def pause(self) -> None:
        """Pause playback without resetting position."""
        with self._lock:
            self._playing = False
        self._stop_stream()

    def stop(self) -> None:
        """Stop playback and reset position to zero."""
        with self._lock:
            self._playing = False
            self._position = 0
        self._stop_stream()

    def toggle_play_pause(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def seek(self, time_seconds: float) -> None:
        """Seek to *time_seconds*."""
        with self._lock:
            if self._audio is None:
                return
            self._position = max(0, min(int(time_seconds * self._sr), len(self._audio) - 1))

    def loop_chunk(self, chunk: Chunk) -> None:
        """Loop playback between the start and end of *chunk*."""
        with self._lock:
            if self._audio is None:
                return
            self._loop_start = int(chunk.start * self._sr)
            self._loop_end   = int(chunk.end   * self._sr)
            self._position   = self._loop_start

    def clear_loop(self) -> None:
        """Disable loop region."""
        with self._lock:
            self._loop_start = None
            self._loop_end   = None

    def _start_stream(self) -> None:
        self._stop_stream()
        stream = sd.OutputStream(
            samplerate=self._sr, channels=1, dtype="float32",
            callback=self._audio_callback,
            finished_callback=self._on_stream_finished,
            blocksize=1024,
        )
        with self._lock:
            self._stream = stream
        stream.start()

    def _stop_stream(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop(ignore_errors=True)
            stream.close(ignore_errors=True)

    def _audio_callback(self, outdata: np.ndarray, frames: int, _time, _status) -> None:
        with self._lock:
            if not self._playing or self._audio is None:
                outdata[:] = 0
                return
            loop_start = self._loop_start
            loop_end   = self._loop_end
            pos        = self._position
            audio      = self._audio
            vol        = self._volume

        buf = np.zeros(frames, dtype=np.float32)
        remaining, write_pos = frames, 0

        while remaining > 0:
            end_pos   = loop_end if loop_end is not None else len(audio)
            available = end_pos - pos
            if available <= 0:
                if loop_start is not None and loop_end is not None:
                    pos = loop_start
                    continue
                with self._lock:
                    self._playing = False
                break
            take = min(remaining, available)
            buf[write_pos:write_pos + take] = audio[pos:pos + take]
            pos += take
            write_pos += take
            remaining -= take
            if loop_start is not None and loop_end is not None and pos >= loop_end:
                pos = loop_start

        outdata[:, 0] = buf * vol
        with self._lock:
            self._position = pos

    def _on_stream_finished(self) -> None:
        pass
