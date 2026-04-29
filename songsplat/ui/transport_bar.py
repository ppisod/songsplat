"""Transport bar - playback controls, progress and volume."""

from __future__ import annotations

import tkinter as tk
from typing import Optional

from songsplat.audio.playback import AudioPlayer
from songsplat.core.models import Song
from songsplat.ui import theme as T

_TRACK_H  = 3   # px height of the slider track
_THUMB_R  = 5   # px radius of the thumb circle


class _Slider(tk.Canvas):
    """Minimal horizontal slider: thin track + circular thumb."""

    def __init__(self, master, length=200, **kw) -> None:
        h = _THUMB_R * 2 + 2
        super().__init__(master, width=length, height=h,
                         bg=T.SIDEBAR, highlightthickness=0, bd=0, **kw)
        self._value    = 0.0
        self._dragging = False
        self._press_cb = None
        self._release_cb = None
        self.bind("<Configure>",       lambda _e: self._draw())
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<B1-Motion>",       self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def get(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        self._value = max(0.0, min(1.0, float(value)))
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        actual_w = self.winfo_width()
        w = actual_w if actual_w > 1 else int(self["width"])
        h = int(self["height"])
        cy  = h // 2
        pad = _THUMB_R + 1
        # Track
        self.create_rectangle(pad, cy - _TRACK_H // 2,
                              w - pad, cy + _TRACK_H // 2,
                              fill=T.BORDER, outline="")
        # Filled portion
        tx = pad + int((w - pad * 2) * self._value)
        if tx > pad:
            self.create_rectangle(pad, cy - _TRACK_H // 2,
                                  tx, cy + _TRACK_H // 2,
                                  fill=T.FG, outline="")
        # Thumb
        self.create_oval(tx - _THUMB_R, cy - _THUMB_R,
                         tx + _THUMB_R, cy + _THUMB_R,
                         fill=T.FG, outline="")

    def _x_to_value(self, x: int) -> float:
        actual_w = self.winfo_width()
        w   = actual_w if actual_w > 1 else int(self["width"])
        pad = _THUMB_R + 1
        return max(0.0, min(1.0, (x - pad) / max(1, w - pad * 2)))

    def _on_press(self, e) -> None:
        self._dragging = True
        self.set(self._x_to_value(e.x))
        if self._press_cb:
            self._press_cb()

    def _on_drag(self, e) -> None:
        if self._dragging:
            self.set(self._x_to_value(e.x))

    def _on_release(self, e) -> None:
        self._dragging = False
        self.set(self._x_to_value(e.x))
        if self._release_cb:
            self._release_cb()


class TransportBar(tk.Frame):
    """Playback controls docked at the bottom of the main window."""

    def __init__(self, master, player: AudioPlayer) -> None:
        super().__init__(master, bg=T.SIDEBAR, height=52)
        self.pack_propagate(False)
        self._player = player
        self._song:  Optional[Song] = None
        self._build()
        self._poll_id = self.after(150, self._poll)

    def _build(self) -> None:
        self.grid_columnconfigure(3, weight=1)

        self._btn_play = T.btn(self, "▶", self._toggle_play)
        self._btn_play.configure(width=3, pady=6)
        self._btn_play.grid(row=0, column=0, padx=(12, 4), pady=8)

        btn_stop = T.btn(self, "■", self._stop, width=2)
        btn_stop.configure(pady=6)
        btn_stop.grid(row=0, column=1, padx=4, pady=8)

        self._lbl_time = tk.Label(
            self, text="0:00 / 0:00", bg=T.SIDEBAR, fg=T.FG_DIM,
            font=T.FONT_MONO, width=12,
        )
        self._lbl_time.grid(row=0, column=2, padx=(8, 4), pady=8)

        self._slider = _Slider(self)
        self._slider._press_cb   = lambda: setattr(self, "_dragging", True)
        self._slider._release_cb = self._on_slider_release
        self._slider.grid(row=0, column=3, sticky="ew", padx=12, pady=0)

        tk.Label(self, text="vol", bg=T.SIDEBAR, fg=T.FG_DIM,
                 font=T.FONT_SMALL).grid(row=0, column=4, padx=(4, 2), pady=8)

        self._vol = _Slider(self, length=80)
        self._vol.set(1.0)
        self._vol._release_cb = lambda: setattr(
            self._player, "volume", self._vol.get())
        self._vol.grid(row=0, column=5, padx=(2, 12), pady=0)

    def set_song(self, song: Optional[Song]) -> None:
        """Update the displayed song."""
        self._song = song
        self._slider.set(0)
        dur = song.duration if song else 0.0
        self._lbl_time.configure(text=f"0:00 / {_fmt(dur)}")

    def _toggle_play(self) -> None:
        self._player.toggle_play_pause()

    def _stop(self) -> None:
        self._player.stop()
        self._slider.set(0)
        dur = self._song.duration if self._song else 0.0
        self._lbl_time.configure(text=f"0:00 / {_fmt(dur)}")

    def _on_slider_release(self) -> None:
        self._dragging = False
        if self._song:
            self._player.seek(self._slider.get() * self._song.duration)

    def _poll(self) -> None:
        self._player.drain_events()
        self._btn_play.configure(text="⏸" if self._player.is_playing else "▶")
        if not getattr(self, "_dragging", False) and self._song and self._song.duration > 0:
            pos = self._player.position
            self._slider.set(min(1.0, pos / self._song.duration))
            self._lbl_time.configure(
                text=f"{_fmt(pos)} / {_fmt(self._song.duration)}")
        self._poll_id = self.after(150, self._poll)

    def destroy(self) -> None:
        try:
            self.after_cancel(self._poll_id)
        except Exception:
            pass
        super().destroy()


def _fmt(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"
