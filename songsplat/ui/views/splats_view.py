"""Splats view - create and edit splat definitions."""

from __future__ import annotations

import tkinter as tk
import tkinter.colorchooser as colorchooser
import tkinter.messagebox as messagebox
from typing import Optional

from songsplat.core.models import Splat
from songsplat.core.undo import add_splat_action, delete_splat_action
from songsplat.ui.app import BaseView
from songsplat.ui import theme as T

PRESETS = [
    "#111111", "#333333", "#555555", "#777777",
    "#999999", "#BBBBBB", "#DDDDDD", "#FFFFFF",
]


def _set_bg_deep(widget, bg: str) -> None:
    """Recursively set bg, skipping widgets tagged as color-dot indicators."""
    if getattr(widget, "_is_color_dot", False):
        return
    try:
        widget.configure(bg=bg)
    except tk.TclError:
        pass
    for child in widget.winfo_children():
        _set_bg_deep(child, bg)


class SplatsView(BaseView):
    """List of splats with inline editor."""

    def __init__(self, master, app) -> None:
        super().__init__(master, app)
        self._selected: Optional[Splat] = None
        self._color                      = "#111111"
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        hdr = tk.Frame(self, bg=T.BG)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(20, 8))
        T.lbl(hdr, "Splats", title=True).pack(side="left")
        T.btn(hdr, "+ New Splat", self._add, accent=True).pack(side="right")

        self._scroll = T.Scrollable(self, bg=T.BG2)
        self._scroll.grid(row=1, column=0, sticky="nsew", padx=(20, 8), pady=(0, 20))

        panel = T.Card(self)
        panel.grid(row=1, column=1, sticky="nsew", padx=(8, 20), pady=(0, 20))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, minsize=280)
        self._build_panel(panel)

        self._rows: dict[str, tk.Frame] = {}
        self._refresh()

    def _build_panel(self, p: T.Card) -> None:
        # Placeholder – shown when no splat is selected
        self._panel_placeholder = T.lbl(p, "Select a splat", dim=True, bg=T.BG2)
        self._panel_placeholder.grid(row=0, column=0, pady=40)

        # Content frame – hidden until a splat is selected
        self._panel_frame = tk.Frame(p, bg=T.BG2)
        self._panel_frame.grid_columnconfigure(1, weight=1)
        f = self._panel_frame

        T.lbl(f, "Edit Splat", bold=True, bg=T.BG2).grid(
            row=0, column=0, columnspan=2, padx=16, pady=(16, 12), sticky="w")

        def field(row, label, var):
            T.lbl(f, label, dim=True, bg=T.BG2).grid(row=row, column=0, padx=16, pady=4, sticky="w")
            e = T.entry(f, textvariable=var)
            e.grid(row=row, column=1, padx=16, pady=4, sticky="ew")
            return e

        self._name_var = tk.StringVar()
        self._low_var  = tk.StringVar()
        self._high_var = tk.StringVar()
        field(1, "Name",       self._name_var)
        field(2, "Low label",  self._low_var)
        field(3, "High label", self._high_var)

        T.lbl(f, "Color", dim=True, bg=T.BG2).grid(row=4, column=0, padx=16, pady=4, sticky="w")
        cr = tk.Frame(f, bg=T.BG2)
        cr.grid(row=4, column=1, padx=16, pady=4, sticky="w")
        self._color_dot = tk.Frame(cr, bg=self._color, width=20, height=20)
        self._color_dot.pack(side="left", padx=(0, 6))
        T.btn(cr, "Pick", self._pick_color).pack(side="left")

        preset_row = tk.Frame(f, bg=T.BG2)
        preset_row.grid(row=5, column=0, columnspan=2, padx=16, pady=4, sticky="w")
        for c in PRESETS:
            dot = tk.Frame(preset_row, bg=c, width=18, height=18, cursor="hand2")
            dot.pack(side="left", padx=2)
            dot.bind("<Button-1>", lambda _e, col=c: self._set_color(col))

        T.separator(f).grid(row=6, column=0, columnspan=2, sticky="ew", padx=16, pady=8)
        T.btn(f, "Save changes", self._save_edit, accent=True).grid(
            row=7, column=0, columnspan=2, padx=16, pady=4, sticky="ew")
        T.btn(f, "Delete splat", self._delete, danger=True).grid(
            row=8, column=0, columnspan=2, padx=16, pady=(4, 16), sticky="ew")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _show_panel(self) -> None:
        self._panel_placeholder.grid_remove()
        self._panel_frame.grid(row=0, column=0, sticky="nsew")

    def _hide_panel(self) -> None:
        self._panel_frame.grid_remove()
        self._panel_placeholder.grid(row=0, column=0, pady=40)

    def _refresh(self) -> None:
        for w in self._scroll.inner.winfo_children():
            w.destroy()
        self._rows.clear()
        splats = self.project.sorted_splats() if self.project else []
        if not splats:
            T.lbl(self._scroll.inner, "No splats. Click '+ New Splat' to create one.",
                  dim=True, wraplength=280, bg=T.BG2).pack(padx=20, pady=40)
            return
        for splat in splats:
            row = self._make_row(splat)
            row.pack(fill="x", padx=4, pady=2)
            self._rows[splat.id] = row
        self._highlight()

    def _make_row(self, splat: Splat) -> tk.Frame:
        row = T.Card(self._scroll.inner)
        row.configure(cursor="hand2")
        dot = tk.Frame(row, bg=splat.color, width=10, height=10)
        dot._is_color_dot = True          # skip this frame in _set_bg_deep
        dot.pack(side="left", padx=(10, 6), pady=18)
        tk.Label(row, text=splat.name, bg=T.BG2, fg=T.FG,
                 font=T.FONT_BOLD, anchor="w").pack(side="left", pady=8)
        lo = splat.low_label or "0"
        hi = splat.high_label or "1"
        tk.Label(row, text=f"{lo} - {hi}", bg=T.BG2, fg=T.FG_DIM,
                 font=T.FONT_SMALL).pack(side="right", padx=12)
        for w in [row, *row.winfo_children()]:
            try:
                w.bind("<Button-1>", lambda _e, s=splat: self._select(s))
            except tk.TclError:
                pass
        return row

    def _select(self, splat: Splat) -> None:
        self._selected = splat
        self._highlight()
        self._name_var.set(splat.name)
        self._low_var.set(splat.low_label)
        self._high_var.set(splat.high_label)
        self._set_color(splat.color)
        self._show_panel()

    def _highlight(self) -> None:
        sid = self._selected.id if self._selected else None
        for k, row in self._rows.items():
            bg = T.BG3 if k == sid else T.BG2
            _set_bg_deep(row, bg)

    def _pick_color(self) -> None:
        color = colorchooser.askcolor(color=self._color, title="Pick color")[1]
        if color:
            self._set_color(color)

    def _set_color(self, color: str) -> None:
        self._color = color
        self._color_dot.configure(bg=color)

    def _save_edit(self) -> None:
        if not self._selected:
            return
        self._selected.name       = self._name_var.get().strip() or self._selected.name
        self._selected.low_label  = self._low_var.get()
        self._selected.high_label = self._high_var.get()
        self._selected.color      = self._color
        self.app.mark_dirty()
        self._refresh()

    def _add(self) -> None:
        if not self.project:
            return
        splat = Splat(
            name=f"Splat {len(self.project.splats) + 1}",
            order=len(self.project.splats),
            color=PRESETS[len(self.project.splats) % len(PRESETS)],
        )
        action = add_splat_action(self.project, splat)
        action.redo()
        self.app.undo_stack.push(action)
        self.app.mark_dirty()
        self._refresh()
        self._select(splat)

    def _delete(self) -> None:
        if not self._selected or not self.project:
            return
        if not messagebox.askyesno("Delete", f"Delete '{self._selected.name}'?", parent=self):
            return
        action = delete_splat_action(self.project, self._selected)
        action.redo()
        self.app.undo_stack.push(action)
        self._selected = None
        self.app.mark_dirty()
        self._refresh()
        self._hide_panel()

    def set_project(self, project) -> None:
        self._selected = None
        self._hide_panel()
        self._refresh()

    def on_show(self) -> None:
        self._refresh()
