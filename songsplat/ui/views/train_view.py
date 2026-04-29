"""Train view - configure and run model training."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

from songsplat.ui.app import BaseView
from songsplat.ui import theme as T


class TrainView(BaseView):
    """Training configuration panel with live log output."""

    def __init__(self, master, app) -> None:
        super().__init__(master, app)
        self._trainer = None
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        hdr = tk.Frame(self, bg=T.BG)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(20, 8))
        T.lbl(hdr, "Train", title=True).pack(side="left")

        cfg = T.Card(self)
        cfg.grid(row=1, column=0, sticky="nsew", padx=(20, 8), pady=(0, 20))
        cfg.grid_columnconfigure(1, weight=1)
        self._build_config(cfg)

        right = T.Card(self)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 20), pady=(0, 20))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        self._build_log(right)

    def _build_config(self, p: T.Card) -> None:
        T.lbl(p, "Configuration", bold=True, bg=T.BG2).grid(
            row=0, column=0, columnspan=2, padx=16, pady=(16, 8), sticky="w")

        def field(r, label, var, values=None):
            T.lbl(p, label, dim=True, bg=T.BG2).grid(row=r, column=0, padx=16, pady=5, sticky="w")
            if values:
                w = T.dropdown(p, var, values)
            else:
                w = T.entry(p, textvariable=var)
            w.grid(row=r, column=1, padx=16, pady=5, sticky="ew")

        self._arch_var   = tk.StringVar(value="pretrained")
        self._epochs_var = tk.StringVar(value="20")
        self._lr_var     = tk.StringVar(value="1e-3")
        self._batch_var  = tk.StringVar(value="16")

        field(1, "Architecture", self._arch_var, ["pretrained", "raw_transformer"])
        field(2, "Epochs",       self._epochs_var)
        field(3, "Learning rate", self._lr_var)
        field(4, "Batch size",   self._batch_var)

        T.separator(p).grid(row=5, column=0, columnspan=2, sticky="ew", padx=16, pady=8)

        self._btn_train = T.btn(p, "Start Training", self._start, accent=True)
        self._btn_train.grid(row=6, column=0, columnspan=2, padx=16, pady=4, sticky="ew")

        self._btn_stop = T.btn(p, "Stop", self._stop, danger=True)
        self._btn_stop.grid(row=7, column=0, columnspan=2, padx=16, pady=(4, 8), sticky="ew")
        self._btn_stop.configure(state="disabled")

        self._progress = ttk.Progressbar(p, value=0, maximum=1.0, length=200)
        self._progress.grid(row=8, column=0, columnspan=2, padx=16, pady=(4, 16), sticky="ew")

    def _build_log(self, p: T.Card) -> None:
        T.lbl(p, "Training Log", bold=True, bg=T.BG2).grid(
            row=0, column=0, padx=16, pady=(16, 8), sticky="w")
        self._log = tk.Text(p, bg=T.BG3, fg=T.FG,
                            font=T.FONT_MONO, state="disabled",
                            relief="flat", padx=8, pady=8,
                            highlightthickness=1, highlightbackground=T.BORDER)
        self._log.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

    def _log_line(self, msg: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _start(self) -> None:
        if not self.project:
            self._log_line("No project loaded.")
            return
        if not self.project.splats:
            self._log_line("No splats defined.")
            return
        labeled = sum(1 for s in self.project.songs
                      for c in s.chunks if c.labels)
        if labeled == 0:
            self._log_line("No labeled chunks. Label some chunks first.")
            return
        try:
            epochs = int(self._epochs_var.get())
            lr     = float(self._lr_var.get())
            batch  = int(self._batch_var.get())
        except ValueError:
            self._log_line("Invalid training parameters.")
            return

        self._btn_train.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._progress.configure(value=0)
        self._log_line(f"Starting: arch={self._arch_var.get()} epochs={epochs} lr={lr}")

        from songsplat.ml.trainer import Trainer, TrainerConfig
        cfg               = TrainerConfig()
        cfg.architecture  = self._arch_var.get()
        cfg.epochs        = epochs
        cfg.lr            = lr
        cfg.batch_size    = batch

        self._trainer = Trainer(project=self.project, config=cfg)
        self._trainer.on_epoch_end = lambda ep, tl, vl, _psl: self.after(
            0, self._on_epoch, ep + 1, epochs, vl)
        self._trainer.on_finished  = lambda ckpt: self.after(0, self._done, ckpt, None)
        self._trainer.on_error     = lambda msg:  self.after(0, self._done, None, msg)
        self._trainer.start()

    def _on_epoch(self, epoch: int, total: int, loss: float) -> None:
        self._progress.configure(value=epoch / total)
        self._log_line(f"  epoch {epoch}/{total}  loss={loss:.4f}")

    def _done(self, checkpoint, error: Optional[str]) -> None:
        self._btn_train.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        if error:
            self._progress.configure(value=0)
            self._log_line(f"Error: {error}")
            return
        self.project.best_checkpoint = checkpoint
        if checkpoint not in self.project.checkpoints:
            self.project.checkpoints.append(checkpoint)
        self.app.mark_dirty()
        self._progress.configure(value=1)
        self._log_line(f"Done. Best loss={checkpoint.loss:.4f}")

    def _stop(self) -> None:
        if self._trainer:
            self._trainer.stop()
        self._log_line("Stopping...")
