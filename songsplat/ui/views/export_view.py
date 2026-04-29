"""Export view - export trained splat models as .splat files."""

from __future__ import annotations

import json
import os
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
from typing import Optional

from songsplat.core.models import Splat
from songsplat.ui.app import BaseView
from songsplat.ui import theme as T

SPLAT_EXT = ".splat"


class ExportView(BaseView):
    """Select splats and export each as a portable .splat bundle."""

    def __init__(self, master, app) -> None:
        super().__init__(master, app)
        self._checks: dict[str, tk.BooleanVar] = {}
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        hdr = tk.Frame(self, bg=T.BG)
        hdr.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 8))
        T.lbl(hdr, "Export", title=True).pack(side="left")

        card = T.Card(self)
        card.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        T.lbl(card, "Select splats to export", bold=True, bg=T.BG2).grid(
            row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        self._checks_frame = T.Scrollable(card, bg=T.BG2)
        self._checks_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=4)

        btn_row = tk.Frame(card, bg=T.BG2)
        btn_row.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 4))
        T.btn(btn_row, "Select All", self._select_all).pack(side="left", padx=(0, 8))
        T.btn(btn_row, f"Export Selected ({SPLAT_EXT})", self._export, accent=True).pack(side="left")

        self._lbl_status = T.lbl(btn_row, "", dim=True, bg=T.BG2)
        self._lbl_status.pack(side="left", padx=12)

        T.lbl(card,
              "Each .splat file contains model weights and splat metadata.\n"
              "Load it with songsplat.cli.runner or the getting_started examples.",
              dim=True, justify="left", wraplength=500, bg=T.BG2,
              ).grid(row=3, column=0, padx=16, pady=(4, 16), sticky="w")

    def _refresh(self) -> None:
        for w in self._checks_frame.inner.winfo_children():
            w.destroy()
        self._checks.clear()
        splats = self.project.sorted_splats() if self.project else []
        if not splats:
            T.lbl(self._checks_frame.inner, "No splats defined.",
                  dim=True, bg=T.BG2).pack(padx=8, pady=20)
            return
        ckpt     = self.project.best_checkpoint if self.project else None
        has_model = ckpt is not None and os.path.isfile(ckpt.path)
        for splat in splats:
            var = tk.BooleanVar(value=True)
            self._checks[splat.id] = var
            row = tk.Frame(self._checks_frame.inner, bg=T.BG2)
            row.pack(fill="x", pady=2)
            tk.Checkbutton(row, variable=var, bg=T.BG2,
                           activebackground=T.BG2, cursor="hand2").pack(side="left")
            tk.Frame(row, bg=splat.color, width=10, height=10).pack(side="left", padx=4)
            tk.Label(row, text=splat.name, bg=T.BG2, fg=T.FG, font=T.FONT).pack(side="left")
            if not has_model:
                tk.Label(row, text="(no model - labels only)",
                         bg=T.BG2, fg=T.FG_DIM, font=T.FONT_SMALL).pack(side="left", padx=8)

    def _select_all(self) -> None:
        for var in self._checks.values():
            var.set(True)

    def _export(self) -> None:
        if not self.project:
            messagebox.showinfo("No project", "Open a project first.", parent=self)
            return
        selected = [s for s in self.project.splats
                    if self._checks.get(s.id, tk.BooleanVar(value=False)).get()]
        if not selected:
            messagebox.showinfo("Nothing selected", "Select at least one splat.", parent=self)
            return
        ckpt = self.project.best_checkpoint
        for splat in selected:
            path = filedialog.asksaveasfilename(
                title=f"Export '{splat.name}'",
                defaultextension=SPLAT_EXT,
                initialfile=splat.name.replace(" ", "_").lower() + SPLAT_EXT,
                filetypes=[(f"Splat files", f"*{SPLAT_EXT}"), ("All files", "*.*")],
            )
            if not path:
                continue
            try:
                _write_splat(path, splat, ckpt)
                self._lbl_status.configure(text=f"Exported {splat.name}")
            except Exception as e:
                messagebox.showerror("Export error", str(e), parent=self)

    def set_project(self, project) -> None:
        self._refresh()

    def on_show(self) -> None:
        self._refresh()


def _write_splat(path: str, splat: Splat, ckpt) -> None:
    """Write a .splat bundle (zip with meta.json + optional model weights)."""
    import zipfile
    meta = {
        "splat":        splat.to_dict(),
        "has_model":    ckpt is not None and os.path.isfile(ckpt.path) if ckpt else False,
        "architecture": ckpt.architecture if ckpt else None,
        "epoch":        ckpt.epoch        if ckpt else None,
        "loss":         ckpt.loss         if ckpt else None,
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta, indent=2))
        if ckpt and os.path.isfile(ckpt.path):
            zf.write(ckpt.path, "model.safetensors")
