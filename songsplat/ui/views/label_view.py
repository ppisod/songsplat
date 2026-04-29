"""Label view - waveform timeline and per-chunk splat labeling."""

from __future__ import annotations

import tkinter as tk
from typing import Optional

from songsplat.core.models import Chunk, CurvePoint, Song, Splat, SplatCurve
from songsplat.core.undo import set_chunk_label_action
from songsplat.ui.app import BaseView
from songsplat.ui import theme as T

WF_H       = 100
ROW_H      = 64
NUDGE_STEP = 0.05

WF_COLOR    = "#333333"
CHUNK_LINE  = T.BORDER
CHUNK_SEL   = T.ACCENT
PLAYHEAD    = T.FG
ACTIVE_FILL = T.SEL


class LabelView(BaseView):
    """Waveform + per-splat label rows with click and draw modes."""

    def __init__(self, master, app) -> None:
        super().__init__(master, app)
        self._song:             Optional[Song]  = None
        self._active_splat:     Optional[Splat] = None
        self._active_chunk_idx: int             = -1
        self._mode                              = "click"
        self._zoom                              = 1.0
        self._pan_x                             = 0.0
        self._build()
        self._poll_id = self.after(80, self._poll_playhead)

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        toolbar = tk.Frame(self, bg=T.BG2, height=46)
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.grid_propagate(False)
        toolbar.grid_columnconfigure(4, weight=1)
        self._build_toolbar(toolbar)

        wf_outer = tk.Frame(self, bg=T.BG2, height=WF_H + 4)
        wf_outer.grid(row=1, column=0, sticky="ew")
        wf_outer.grid_propagate(False)
        wf_outer.grid_columnconfigure(0, weight=1)
        wf_outer.grid_rowconfigure(0, weight=1)
        self._wf = tk.Canvas(wf_outer, bg=T.BG2, highlightthickness=0, height=WF_H)
        self._wf.grid(row=0, column=0, sticky="ew", padx=0, pady=2)
        self._wf.bind("<Configure>",  lambda _e: self._draw_waveform())
        self._wf.bind("<Button-1>",   self._wf_click)
        self._wf.bind("<B1-Motion>",  self._wf_click)
        self._wf.bind("<MouseWheel>", self._wf_scroll)
        self._wf.bind("<Button-4>",   self._wf_scroll)
        self._wf.bind("<Button-5>",   self._wf_scroll)

        self._rows_scroll = T.Scrollable(self, bg=T.BG)
        self._rows_scroll.grid(row=2, column=0, sticky="nsew")
        self._rows_scroll.inner.grid_columnconfigure(0, weight=1)

        self._splat_rows: dict[str, _LabelRow] = {}

    def _build_toolbar(self, tb: tk.Frame) -> None:
        T.lbl(tb, "Song:", dim=True, bg=T.BG2).grid(row=0, column=0, padx=(12, 4), pady=10)

        self._song_var = tk.StringVar(value="(none)")
        self._song_menu = T.dropdown(tb, self._song_var, ["(none)"],
                                      command=self._on_song_select)
        self._song_menu.grid(row=0, column=1, padx=4, pady=10)

        T.lbl(tb, "Splat:", dim=True, bg=T.BG2).grid(row=0, column=2, padx=(12, 4), pady=10)

        self._splat_var = tk.StringVar(value="(none)")
        self._splat_menu = T.dropdown(tb, self._splat_var, ["(none)"],
                                       command=self._on_splat_select)
        self._splat_menu.grid(row=0, column=3, padx=4, pady=10)

        # Spacer
        tk.Frame(tb, bg=T.BG2).grid(row=0, column=4, sticky="ew")

        # Mode buttons
        self._btn_click = T.btn(tb, "Click", lambda: self._set_mode("click"), accent=True)
        self._btn_click.grid(row=0, column=5, padx=4, pady=10)
        self._btn_draw = T.btn(tb, "Draw", lambda: self._set_mode("draw"))
        self._btn_draw.grid(row=0, column=6, padx=(4, 12), pady=10)

        # Zoom buttons
        T.btn(tb, "−", self._zoom_out).grid(row=0, column=7, padx=(4, 2), pady=10)
        T.btn(tb, "+", self._zoom_in).grid(row=0, column=8, padx=(2, 12), pady=10)

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._btn_click.configure(bg=T.ACCENT if mode == "click" else T.SEL)
        self._btn_draw.configure(bg=T.ACCENT if mode == "draw"  else T.SEL)

    def _zoom_in(self) -> None:
        self._zoom = min(50.0, self._zoom * 1.3)
        self._apply_view()

    def _zoom_out(self) -> None:
        self._zoom = max(1.0, self._zoom / 1.3)
        # Clamp pan so we don't go out of bounds
        vf = 1.0 / self._zoom
        self._pan_x = max(0.0, min(1.0 - vf, self._pan_x))
        self._apply_view()

    def _apply_view(self) -> None:
        self._draw_waveform()
        for row in self._splat_rows.values():
            row.set_view(self._pan_x, self._zoom)

    def set_project(self, project) -> None:
        self._active_splat     = None
        self._active_chunk_idx = -1
        self._refresh_menus()
        self._refresh_rows()

    def set_song(self, song: Optional[Song]) -> None:
        self._song = song
        if song:
            self._song_var.set(song.name)
        self._draw_waveform()
        self._refresh_rows()

    def on_show(self) -> None:
        self._refresh_menus()
        self._refresh_rows()
        self._draw_waveform()

    def refresh(self) -> None:
        """Refresh waveform and label rows (e.g. after re-chunking)."""
        self._draw_waveform()
        self._refresh_rows()

    def refresh_labels(self) -> None:
        """Redraw all label rows (called after undo/redo)."""
        for row in self._splat_rows.values():
            row.redraw()

    def _refresh_menus(self) -> None:
        songs  = self.project.songs          if self.project else []
        splats = self.project.sorted_splats() if self.project else []

        song_names = [s.name for s in songs] or ["(none)"]
        self._song_menu["menu"].delete(0, "end")
        for name in song_names:
            self._song_menu["menu"].add_command(
                label=name, command=lambda n=name: (
                    self._song_var.set(n), self._on_song_select(n)))
        if self._song and self._song in songs:
            self._song_var.set(self._song.name)
        elif songs:
            self._song_var.set(songs[0].name)
            self._song = songs[0]
        else:
            self._song_var.set("(none)")
            self._song = None

        splat_names = [s.name for s in splats] or ["(none)"]
        self._splat_menu["menu"].delete(0, "end")
        for name in splat_names:
            self._splat_menu["menu"].add_command(
                label=name, command=lambda n=name: (
                    self._splat_var.set(n), self._on_splat_select(n)))
        if splats:
            if self._active_splat and self._active_splat in splats:
                self._splat_var.set(self._active_splat.name)
            else:
                self._active_splat = splats[0]
                self._splat_var.set(splats[0].name)
        else:
            self._splat_var.set("(none)")
            self._active_splat = None

    def _refresh_rows(self) -> None:
        for w in self._rows_scroll.inner.winfo_children():
            w.destroy()
        self._splat_rows.clear()
        if not self.project or not self._song:
            return
        for i, splat in enumerate(self.project.sorted_splats()):
            row = _LabelRow(
                self._rows_scroll.inner, splat, self._song, self.app,
                self._on_click, self._on_draw_start, self._on_draw_motion,
                self._on_draw_end, lambda: self._mode,
            )
            row.grid(row=i, column=0, sticky="ew", padx=8, pady=3)
            row.set_view(self._pan_x, self._zoom)
            row.set_active(self._active_chunk_idx)
            self._splat_rows[splat.id] = row

    def _on_song_select(self, name: str) -> None:
        if not self.project:
            return
        for s in self.project.songs:
            if s.name == name:
                self.app.set_active_song(s)
                break

    def _on_splat_select(self, name: str) -> None:
        if not self.project:
            return
        for s in self.project.sorted_splats():
            if s.name == name:
                self._active_splat = s
                break

    def _draw_waveform(self) -> None:
        c = self._wf
        w, h = c.winfo_width(), c.winfo_height()
        if w < 2 or h < 2:
            return
        c.delete("all")
        if not self._song:
            c.create_text(w // 2, h // 2, text="No song loaded",
                          fill=T.FG_DIM, font=T.FONT)
            return
        wf  = self._song.waveform_cache or []
        dur = self._song.duration
        vf  = 1.0 / self._zoom
        v0, v1 = self._pan_x, min(1.0, self._pan_x + vf)

        if wf:
            total   = len(wf)
            i0, i1  = int(v0 * total), max(int(v0 * total) + 1, int(v1 * total))
            segment = wf[i0:i1]
            if segment:
                step = w / len(segment)
                mid  = h / 2
                for xi, amp in enumerate(segment):
                    x = xi * step
                    c.create_line(x, mid - amp * mid * 0.9,
                                  x, mid + amp * mid * 0.9,
                                  fill=WF_COLOR, width=1)

        if self._song.chunks and dur > 0:
            def tx(t):
                return ((t / dur) - v0) / (v1 - v0) * w

            for ck in self._song.chunks:
                if v0 <= ck.start / dur <= v1:
                    x     = tx(ck.start)
                    color = CHUNK_SEL if ck.index == self._active_chunk_idx else CHUNK_LINE
                    c.create_line(x, 0, x, h, fill=color, width=1)

            if 0 <= self._active_chunk_idx < len(self._song.chunks):
                ck = self._song.chunks[self._active_chunk_idx]
                xs = max(0,  tx(ck.start))
                xe = min(w,  tx(ck.end))
                c.create_rectangle(xs, 0, xe, h, fill=ACTIVE_FILL, outline="")

            pos    = self.app.player.position
            p_frac = pos / dur if dur > 0 else 0
            if v0 <= p_frac <= v1:
                px = tx(pos)
                c.create_line(px, 0, px, h, fill=PLAYHEAD, width=2)

    def _wf_click(self, event) -> None:
        if not self._song or not self._song.chunks:
            return
        vf   = 1.0 / self._zoom
        frac = self._pan_x + (event.x / self._wf.winfo_width()) * vf
        time = frac * self._song.duration
        for ck in self._song.chunks:
            if ck.start <= time < ck.end:
                self._active_chunk_idx = ck.index
                self.app.player.seek(time)
                self._draw_waveform()
                for row in self._splat_rows.values():
                    row.set_active(ck.index)
                break

    def _wf_scroll(self, event) -> None:
        delta = -1 if (event.num == 4 or getattr(event, "delta", 0) > 0) else 1
        if event.state & 0x1:
            # Shift + scroll → zoom
            self._zoom = max(1.0, min(50.0, self._zoom * (1.15 if delta < 0 else 0.87)))
            vf = 1.0 / self._zoom
            self._pan_x = max(0.0, min(1.0 - vf, self._pan_x))
        else:
            # Scroll → pan left/right
            vf = 1.0 / self._zoom
            self._pan_x = max(0.0, min(1.0 - vf, self._pan_x + delta * vf * 0.08))
        self._apply_view()

    def _poll_playhead(self) -> None:
        self._draw_waveform()
        self._poll_id = self.after(80, self._poll_playhead)

    def _on_click(self, splat: Splat, chunk: Chunk, value: float) -> None:
        action = set_chunk_label_action(chunk, splat.id, value)
        action.redo()
        self.app.undo_stack.push(action)
        self.app.mark_dirty()
        self._active_chunk_idx = chunk.index
        self._draw_waveform()
        for row in self._splat_rows.values():
            row.set_active(chunk.index)
        self._splat_rows[splat.id].redraw()

    def _on_draw_start(self, _splat: Splat) -> None:
        pass

    def _on_draw_motion(self, splat: Splat, time: float, value: float) -> None:
        if not self._song:
            return
        curve = self._song.curves.setdefault(splat.id, SplatCurve(splat_id=splat.id))
        eps   = 0.1
        curve.points = [p for p in curve.points if not (time - eps <= p.time <= time + eps)]
        curve.points.append(CurvePoint(time=time, value=value))

    def _on_draw_end(self, splat: Splat) -> None:
        if not self._song:
            return
        self._song.apply_curve_to_chunks(splat.id)
        self.app.mark_dirty()
        if splat.id in self._splat_rows:
            self._splat_rows[splat.id].redraw()

    def handle_key(self, action: str) -> None:
        """Handle keyboard navigation from the main window."""
        if not self._song or not self._song.chunks:
            return
        n = len(self._song.chunks)
        if action == "prev_chunk":
            self._active_chunk_idx = max(0, self._active_chunk_idx - 1)
            self._jump_to_active()
        elif action == "next_chunk":
            self._active_chunk_idx = min(n - 1, self._active_chunk_idx + 1)
            self._jump_to_active()
        elif action in ("nudge_up", "nudge_down") and self._active_splat:
            self._nudge(action == "nudge_up")

    def _jump_to_active(self) -> None:
        if 0 <= self._active_chunk_idx < len(self._song.chunks):
            ck  = self._song.chunks[self._active_chunk_idx]
            self.app.player.seek(ck.start)
            vf  = 1.0 / self._zoom
            tf  = ck.start / self._song.duration
            if tf < self._pan_x or tf > self._pan_x + vf:
                self._pan_x = max(0.0, min(1.0 - vf, tf - vf * 0.2))
            self._apply_view()
            for row in self._splat_rows.values():
                row.set_active(self._active_chunk_idx)

    def _nudge(self, up: bool) -> None:
        if self._active_chunk_idx < 0 or not self._active_splat:
            return
        chunk   = self._song.chunks[self._active_chunk_idx]
        splat   = self._active_splat
        current = chunk.labels.get(splat.id, 0.5)
        new_val = splat.clamp(current + (NUDGE_STEP if up else -NUDGE_STEP))
        action  = set_chunk_label_action(chunk, splat.id, new_val)
        action.redo()
        self.app.undo_stack.push(action)
        self.app.mark_dirty()
        if splat.id in self._splat_rows:
            self._splat_rows[splat.id].redraw()

    def destroy(self) -> None:
        try:
            self.after_cancel(self._poll_id)
        except Exception:
            pass
        super().destroy()


class _LabelRow(tk.Frame):
    """One splat's label track: name panel on the left, canvas on the right."""

    def __init__(self, master, splat: Splat, song: Song, app,
                 on_click, on_draw_start, on_draw_motion, on_draw_end, get_mode) -> None:
        super().__init__(master, bg=T.BG2, height=ROW_H)
        self.grid_propagate(False)
        self.grid_columnconfigure(1, weight=1)

        self._splat          = splat
        self._song           = song
        self._app            = app
        self._on_click       = on_click
        self._on_draw_start  = on_draw_start
        self._on_draw_motion = on_draw_motion
        self._on_draw_end    = on_draw_end
        self._get_mode       = get_mode
        self._active_chunk   = -1
        self._pan_x          = 0.0
        self._zoom           = 1.0

        # ── Left info panel ──────────────────────────────────────────────
        lf = tk.Frame(self, bg=T.BG2, width=130)
        lf.grid(row=0, column=0, sticky="nsew")
        lf.pack_propagate(False)

        top_row = tk.Frame(lf, bg=T.BG2)
        top_row.pack(side="top", fill="x", padx=6, pady=(8, 0))
        tk.Frame(top_row, bg=splat.color, width=8, height=8).pack(
            side="left", padx=(0, 6), pady=3)
        tk.Label(top_row, text=splat.name, bg=T.BG2, fg=T.FG,
                 font=T.FONT_BOLD, anchor="w").pack(side="left", fill="x", expand=True)

        lo = splat.low_label or "0"
        hi = splat.high_label or "1"
        tk.Label(lf, text=f"{lo} → {hi}", bg=T.BG2, fg=T.FG_DIM,
                 font=T.FONT_SMALL, anchor="w").pack(
                     side="top", fill="x", padx=6, pady=(2, 0))

        clear_lbl = tk.Label(lf, text="clear", bg=T.BG2, fg=T.FG_DIM,
                             font=T.FONT_SMALL, cursor="hand2", anchor="w")
        clear_lbl.pack(side="bottom", fill="x", padx=6, pady=(0, 6))
        clear_lbl.bind("<Button-1>", lambda _e: self._clear_labels())
        clear_lbl.bind("<Enter>",    lambda _e: clear_lbl.configure(fg=T.FG))
        clear_lbl.bind("<Leave>",    lambda _e: clear_lbl.configure(fg=T.FG_DIM))

        # ── Canvas ───────────────────────────────────────────────────────
        self._cv = tk.Canvas(self, bg=T.BG, highlightthickness=0, height=ROW_H - 4)
        self._cv.grid(row=0, column=1, sticky="nsew", padx=(4, 4), pady=2)
        self._cv.bind("<Configure>",       lambda _e: self.redraw())
        self._cv.bind("<Button-1>",        self._mouse_press)
        self._cv.bind("<B1-Motion>",       self._mouse_drag)
        self._cv.bind("<ButtonRelease-1>", self._mouse_release)

    def set_active(self, idx: int) -> None:
        self._active_chunk = idx
        self.redraw()

    def set_view(self, pan_x: float, zoom: float) -> None:
        self._pan_x = pan_x
        self._zoom  = zoom
        self.redraw()

    def redraw(self) -> None:
        """Repaint the chunk value canvas."""
        c = self._cv
        w, h = c.winfo_width(), c.winfo_height()
        if w < 2 or h < 2:
            return
        c.delete("all")
        song, splat = self._song, self._splat
        if not song or not song.chunks or song.duration <= 0:
            return
        dur = song.duration
        vf  = 1.0 / self._zoom
        v0, v1 = self._pan_x, min(1.0, self._pan_x + vf)

        def tx(t):
            return ((t / dur) - v0) / (v1 - v0) * w

        for ck in song.chunks:
            if ck.end / dur < v0 or ck.start / dur > v1:
                continue
            xs = max(0,  tx(ck.start))
            xe = min(w,  tx(ck.end))
            if xe - xs < 1:
                continue
            c.create_line(xs, 0, xs, h, fill=CHUNK_LINE, width=1)
            if splat.id in ck.labels:
                val  = ck.labels[splat.id]
                bh   = val * (h - 4)
                y0   = h - bh - 2
                c.create_rectangle(xs + 1, y0, xe - 1, h - 2,
                                   fill=T.ACCENT, outline="")
                if xe - xs > 28:
                    c.create_text((xs + xe) / 2, h - bh / 2 - 2,
                                  text=f"{val:.2f}", fill=T.BG, font=T.FONT_SMALL)
            else:
                c.create_rectangle(xs, 2, xe - 1, h - 2,
                                   fill=T.BG3, outline=CHUNK_LINE)
            if ck.index == self._active_chunk:
                c.create_rectangle(xs, 0, xe, h, outline=CHUNK_SEL, fill="", width=2)

        curve = song.curves.get(splat.id)
        if curve and curve.points:
            pts = [(tx(p.time), (1 - p.value) * h)
                   for p in curve.sorted_points()
                   if v0 <= p.time / dur <= v1]
            if len(pts) >= 2:
                flat = [coord for pt in pts for coord in pt]
                c.create_line(*flat, fill=T.FG_DIM, width=2, smooth=True)

        for ck in song.chunks:
            if splat.id in ck.predictions:
                if ck.end / dur < v0 or ck.start / dur > v1:
                    continue
                xs  = max(0, tx(ck.start))
                xe  = min(w, tx(ck.end))
                y   = (1 - ck.predictions[splat.id]) * h
                c.create_line(xs, y, xe, y, fill="#888888", width=1, dash=(2, 2))

    def _mouse_press(self, event) -> None:
        if self._get_mode() == "draw":
            self._on_draw_start(self._splat)
            self._handle_draw(event)
        else:
            self._handle_click(event)

    def _mouse_drag(self, event) -> None:
        if self._get_mode() == "draw":
            self._handle_draw(event)
        else:
            self._handle_click(event)

    def _mouse_release(self, event) -> None:
        if self._get_mode() == "draw":
            self._on_draw_end(self._splat)

    def _handle_click(self, event) -> None:
        chunk, value = self._event_to_chunk_value(event)
        if chunk is not None:
            self._on_click(self._splat, chunk, value)
            self.redraw()

    def _handle_draw(self, event) -> None:
        if not self._song or self._song.duration <= 0:
            return
        vf    = 1.0 / self._zoom
        w, h  = self._cv.winfo_width(), self._cv.winfo_height()
        time  = (self._pan_x + (event.x / w) * vf) * self._song.duration
        value = max(0.0, min(1.0, 1.0 - event.y / h))
        self._on_draw_motion(self._splat, time, value)
        self.redraw()

    def _event_to_chunk_value(self, event):
        if not self._song or self._song.duration <= 0:
            return None, None
        vf    = 1.0 / self._zoom
        w, h  = self._cv.winfo_width(), self._cv.winfo_height()
        time  = (self._pan_x + (event.x / w) * vf) * self._song.duration
        value = max(0.0, min(1.0, 1.0 - event.y / h))
        for ck in self._song.chunks:
            if ck.start <= time < ck.end:
                return ck, value
        return None, None

    def _clear_labels(self) -> None:
        import tkinter.messagebox as mb
        if not mb.askyesno("Confirm", f"Clear all '{self._splat.name}' labels?"):
            return
        for ck in self._song.chunks:
            ck.labels.pop(self._splat.id, None)
        self._app.mark_dirty()
        self.redraw()
