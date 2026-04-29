"""Predict view - run model inference on song chunks."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Optional

from songsplat.core.models import Song
from songsplat.ui.app import BaseView
from songsplat.ui import theme as T


class PredictView(BaseView):
    """Run the trained model and write predictions back to chunks."""

    def __init__(self, master, app) -> None:
        super().__init__(master, app)
        self._song: Optional[Song] = None
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        hdr = tk.Frame(self, bg=T.BG)
        hdr.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 8))
        T.lbl(hdr, "Predict", title=True).pack(side="left")

        card = T.Card(self)
        card.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(2, weight=1)

        top = tk.Frame(card, bg=T.BG2)
        top.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))

        T.lbl(top, "Song:", dim=True, bg=T.BG2).pack(side="left", padx=(0, 6))
        self._song_var  = tk.StringVar(value="(none)")
        self._song_menu = T.dropdown(top, self._song_var, ["(none)"],
                                      command=self._on_song_select)
        self._song_menu.pack(side="left", padx=4)

        self._btn_run = T.btn(top, "Run Predictions", self._run, accent=True)
        self._btn_run.pack(side="left", padx=12)

        self._progress = ttk.Progressbar(top, value=0, maximum=1.0, length=160)
        self._progress.pack(side="left", padx=4)

        self._lbl_status = T.lbl(top, "", dim=True, bg=T.BG2)
        self._lbl_status.pack(side="left", padx=4)

        T.lbl(card,
              "Predictions appear as dashed overlay lines in the Label view.\n"
              "Existing hand-labeled values are not overwritten.",
              dim=True, wraplength=600, justify="left", bg=T.BG2,
              ).grid(row=1, column=0, padx=16, pady=8, sticky="w")

        self._log = tk.Text(card, bg=T.BG3, fg=T.FG,
                            font=T.FONT_MONO, state="disabled",
                            relief="flat", padx=8, pady=8,
                            highlightthickness=1, highlightbackground=T.BORDER)
        self._log.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))

    def _log_line(self, msg: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _on_song_select(self, name: str) -> None:
        if not self.project:
            return
        for s in self.project.songs:
            if s.name == name:
                self._song = s
                break

    def _run(self) -> None:
        if not self.project:
            self._log_line("No project loaded.")
            return
        if not self.project.best_checkpoint:
            self._log_line("No trained model. Train a model first.")
            return
        if not self._song:
            self._log_line("Select a song first.")
            return
        if not self._song.chunks:
            self._log_line("Song has no chunks. Chunk it in the Songs view.")
            return

        self._btn_run.configure(state="disabled")
        self._progress.configure(value=0)
        self._lbl_status.configure(text="Running...")
        self._log_line(f"Running inference on '{self._song.name}'...")

        song  = self._song
        ckpt  = self.project.best_checkpoint
        splats = self.project.splats

        def _thread():
            try:
                from songsplat.ml.inference import run_inference
                run_inference(
                    song=song, checkpoint=ckpt, splats=splats,
                    progress_cb=lambda i, n: self.after(
                        0, self._progress.configure, {"value": i / n}),
                )
                self.after(0, self._done, None)
            except Exception as e:
                self.after(0, self._done, str(e))

        threading.Thread(target=_thread, daemon=True).start()

    def _done(self, error: Optional[str]) -> None:
        self._btn_run.configure(state="normal")
        self._progress.configure(value=0)
        if error:
            self._lbl_status.configure(text="Error")
            self._log_line(f"Error: {error}")
        else:
            self._lbl_status.configure(text="Done")
            self._log_line("Predictions written. View in the Label view.")
            self.app.mark_dirty()

    def _refresh_menu(self) -> None:
        songs = self.project.songs if self.project else []
        names = [s.name for s in songs] or ["(none)"]
        self._song_menu["menu"].delete(0, "end")
        for name in names:
            self._song_menu["menu"].add_command(
                label=name, command=lambda n=name: (
                    self._song_var.set(n), self._on_song_select(n)))
        if songs:
            self._song_var.set(songs[0].name)
            self._song = songs[0]
        else:
            self._song_var.set("(none)")
            self._song = None

    def set_project(self, project) -> None:
        self._song = None
        self._refresh_menu()

    def on_show(self) -> None:
        self._refresh_menu()
