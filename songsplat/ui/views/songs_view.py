"""Songs view - import audio, manage songs, configure chunking."""

from __future__ import annotations

import os
import threading
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
from pathlib import Path
from typing import Optional

from songsplat.core.models import Project, Song
from songsplat.ui.app import BaseView
from songsplat.ui import theme as T


class SongsView(BaseView):
    """List of songs with import and chunk-configuration controls."""

    def __init__(self, master, app) -> None:
        super().__init__(master, app)
        self._selected: Optional[Song] = None
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        hdr = tk.Frame(self, bg=T.BG)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(20, 8))
        T.lbl(hdr, "Songs", title=True).pack(side="left")
        T.btn(hdr, "+ Import Songs", self._import, accent=True).pack(side="right")

        self._list_scroll = T.Scrollable(self, bg=T.BG2)
        self._list_scroll.grid(row=1, column=0, sticky="nsew", padx=(20, 8), pady=(0, 20))

        detail = T.Card(self)
        detail.grid(row=1, column=1, sticky="nsew", padx=(8, 20), pady=(0, 20))
        detail.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(1, minsize=280)
        self._build_detail(detail)

        self._rows: dict[str, _SongRow] = {}
        self._refresh()

    def _build_detail(self, p: tk.Frame) -> None:
        T.lbl(p, "Chunk Settings", bold=True, bg=T.BG2).grid(
            row=0, column=0, columnspan=2, padx=16, pady=(16, 8), sticky="w")

        def row(r, label, widget_fn):
            T.lbl(p, label, dim=True, bg=T.BG2).grid(row=r, column=0, padx=16, pady=4, sticky="w")
            w = widget_fn()
            w.grid(row=r, column=1, padx=16, pady=4, sticky="ew")
            return w

        self._mode_var = tk.StringVar(value="fixed")
        row(1, "Mode", lambda: T.dropdown(p, self._mode_var, ["fixed", "beat"],
                                           command=self._on_mode_change))

        self._dur_var = tk.StringVar(value="2.0")
        self._dur_row = tk.Frame(p, bg=T.BG2)
        self._dur_row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=2)
        T.lbl(self._dur_row, "Duration (s)", dim=True, bg=T.BG2).pack(side="left")
        T.entry(self._dur_row, textvariable=self._dur_var, width=8).pack(side="right")

        self._beats_var = tk.StringVar(value="4")
        self._beats_row = tk.Frame(p, bg=T.BG2)
        self._beats_row.grid(row=3, column=0, columnspan=2, sticky="ew", padx=16, pady=2)
        T.lbl(self._beats_row, "Beats/chunk", dim=True, bg=T.BG2).pack(side="left")
        T.entry(self._beats_row, textvariable=self._beats_var, width=8).pack(side="right")
        self._beats_row.grid_remove()

        T.separator(p).grid(row=4, column=0, columnspan=2, sticky="ew", padx=16, pady=8)

        self._btn_rechunk = T.btn(p, "Re-chunk Song", self._rechunk, accent=True)
        self._btn_rechunk.grid(row=5, column=0, columnspan=2, padx=16, pady=(4, 0), sticky="ew")
        self._btn_rechunk.configure(state="disabled")

        self._lbl_status = T.lbl(p, "", dim=True, bg=T.BG2)
        self._lbl_status.grid(row=6, column=0, columnspan=2, padx=16, sticky="w")
        self._lbl_status.grid_remove()

        T.separator(p).grid(row=7, column=0, columnspan=2, sticky="ew", padx=16, pady=8)

        self._btn_remove = T.btn(p, "Remove Song", self._remove, danger=True)
        self._btn_remove.grid(row=8, column=0, columnspan=2, padx=16, pady=4, sticky="ew")
        self._btn_remove.configure(state="disabled")

        T.btn(p, "Label this song", lambda: self.app._show_view("label")).grid(
            row=9, column=0, columnspan=2, padx=16, pady=(4, 16), sticky="ew")

    def _on_mode_change(self, value: str) -> None:
        if value == "fixed":
            self._dur_row.grid()
            self._beats_row.grid_remove()
        else:
            self._dur_row.grid_remove()
            self._beats_row.grid()
        if self._selected:
            self._selected.chunk_mode = value

    def _refresh(self) -> None:
        for w in self._list_scroll.inner.winfo_children():
            w.destroy()
        self._rows.clear()
        songs = self.project.songs if self.project else []
        if not songs:
            T.lbl(self._list_scroll.inner, "No songs. Click '+ Import Songs' to begin.",
                  dim=True, wraplength=280, bg=T.BG2).pack(padx=20, pady=40)
            return
        for song in songs:
            row = _SongRow(self._list_scroll.inner, song, lambda s=song: self._select(s))
            row.pack(fill="x", padx=4, pady=2)
            self._rows[song.id] = row
        self._highlight()

    def _select(self, song: Song) -> None:
        self._selected = song
        self.app.set_active_song(song)
        self._highlight()
        self._btn_rechunk.configure(state="normal")
        self._btn_remove.configure(state="normal")
        self._mode_var.set(song.chunk_mode)
        self._on_mode_change(song.chunk_mode)
        self._dur_var.set(str(song.chunk_duration))
        n = len(song.chunks)
        txt = f"{n} chunks  ({song.chunk_mode})" if n else "Not chunked yet"
        self._lbl_status.configure(text=txt)
        self._lbl_status.grid()

    def _highlight(self) -> None:
        sid = self._selected.id if self._selected else None
        for song_id, row in self._rows.items():
            row.configure(bg=T.BG3 if song_id == sid else T.BG2)
            for w in row.winfo_children():
                try:
                    w.configure(bg=T.BG3 if song_id == sid else T.BG2)
                except tk.TclError:
                    pass

    def _import(self) -> None:
        if not self.project:
            messagebox.showinfo("No project", "Open or create a project first.", parent=self)
            return
        paths = filedialog.askopenfilenames(
            title="Import Audio Files",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a *.aiff *.aif"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        from songsplat.audio.loader import build_song_from_path, is_supported
        added, errors = 0, []
        existing = {s.path for s in self.project.songs}
        for path in paths:
            if not is_supported(path):
                errors.append(f"Unsupported: {os.path.basename(path)}")
                continue
            if os.path.abspath(path) in existing:
                continue
            try:
                self.project.songs.append(build_song_from_path(path))
                added += 1
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
        if errors:
            messagebox.showwarning("Import errors", "\n".join(errors), parent=self)
        if added:
            self.app.mark_dirty()
            self._refresh()
            self._select(self.project.songs[-1])

    def _rechunk(self) -> None:
        if not self._selected:
            return
        song = self._selected
        mode = self._mode_var.get()
        song.chunk_mode = mode
        if mode == "fixed":
            try:
                dur = float(self._dur_var.get())
                assert dur > 0
            except Exception:
                messagebox.showerror("Invalid", "Enter a positive duration.", parent=self)
                return
            song.chunk_duration = dur
        else:
            try:
                bpc = int(self._beats_var.get())
                assert bpc >= 1
            except Exception:
                messagebox.showerror("Invalid", "Enter a positive integer for beats/chunk.", parent=self)
                return

        self._btn_rechunk.configure(state="disabled", text="Chunking...")
        self._lbl_status.configure(text="Processing...")
        self._lbl_status.grid()

        def _run():
            try:
                from songsplat.audio.loader import chunk_song_fixed, chunk_song_beats
                if mode == "fixed":
                    chunk_song_fixed(song, float(self._dur_var.get()))
                else:
                    chunk_song_beats(song, int(self._beats_var.get()))
                self.after(0, self._rechunk_done, None)
            except Exception as exc:
                self.after(0, self._rechunk_done, str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def _rechunk_done(self, error: Optional[str]) -> None:
        self._btn_rechunk.configure(state="normal", text="Re-chunk Song")
        if error:
            messagebox.showerror("Chunk error", error, parent=self)
            self._lbl_status.configure(text="Error")
            self._lbl_status.grid()
            return
        self.app.mark_dirty()
        song = self._selected
        self._lbl_status.configure(text=f"{len(song.chunks)} chunks  ({song.chunk_mode})")
        self._lbl_status.grid()
        lv = self.app._views.get("label")
        if lv and hasattr(lv, "refresh"):
            lv.refresh()

    def _remove(self) -> None:
        if not self._selected or not self.project:
            return
        if not messagebox.askyesno("Remove", f"Remove '{self._selected.name}'?", parent=self):
            return
        self.project.songs = [s for s in self.project.songs if s.id != self._selected.id]
        self._selected = None
        self._btn_rechunk.configure(state="disabled")
        self._btn_remove.configure(state="disabled")
        self.app.mark_dirty()
        self._refresh()

    def set_project(self, project) -> None:
        self._selected = None
        self._refresh()

    def on_show(self) -> None:
        self._refresh()


class _SongRow(T.Card):
    """Single row in the song list."""

    def __init__(self, master, song: Song, on_click) -> None:
        super().__init__(master)
        self.configure(cursor="hand2")
        tk.Label(self, text="♪", bg=T.BG2, fg=T.ACCENT,
                 font=("TkDefaultFont", 18), width=2).pack(side="left", padx=(8, 4))
        info = tk.Frame(self, bg=T.BG2)
        info.pack(side="left", fill="both", expand=True, pady=6)
        tk.Label(info, text=song.name, bg=T.BG2, fg=T.FG,
                 font=T.FONT_BOLD, anchor="w").pack(fill="x")
        dur = int(song.duration)
        detail = f"{dur // 60}:{dur % 60:02d}  |  {len(song.chunks)} chunks"
        tk.Label(info, text=detail, bg=T.BG2, fg=T.FG_DIM,
                 font=T.FONT_SMALL, anchor="w").pack(fill="x")
        for w in [self, *self.winfo_children(), *info.winfo_children()]:
            try:
                w.bind("<Button-1>", lambda _e: on_click())
            except tk.TclError:
                pass
