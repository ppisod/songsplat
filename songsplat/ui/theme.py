"""Shared monochrome palette and tkinter helper widgets."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

BG        = "#FFFFFF"
BG2       = "#F5F5F5"
BG3       = "#EEEEEE"
FG        = "#111111"
FG_DIM    = "#666666"
ACCENT    = "#111111"
ACCENT2   = "#000000"
SIDEBAR   = "#EEEEEE"
SEL       = "#E2E2E2"
SEL_HOVER = "#D0D0D0"
BORDER    = "#BBBBBB"
DANGER    = "#DDDDDD"
DANGER2   = "#C8C8C8"
RED       = "#555555"
GREEN     = "#333333"

FONT       = ("TkDefaultFont", 12)
FONT_BOLD  = ("TkDefaultFont", 12, "bold")
FONT_SMALL = ("TkDefaultFont", 10)
FONT_MONO  = ("TkFixedFont", 11)
FONT_TITLE = ("TkDefaultFont", 22, "bold")
FONT_H2    = ("TkDefaultFont", 15, "bold")

_RADIUS = 5  # button corner radius


def _is_dark(hex_color: str) -> bool:
    """Return True if *hex_color* is perceptually dark (luminance < 0.5)."""
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (r * 299 + g * 587 + b * 114) / 255000 < 0.5


def configure_ttk() -> None:
    """Apply monochrome overrides to ttk widgets (call once after Tk() is created)."""
    s = ttk.Style()
    s.theme_use("clam")
    s.configure("TProgressbar", troughcolor=SEL, background=ACCENT, thickness=6, borderwidth=0)
    s.configure("TScale", troughcolor=SEL, background=FG_DIM, sliderlength=12, borderwidth=0)
    s.map("TScale", background=[("active", ACCENT)])
    s.configure("Vertical.TScrollbar", background=BG3, troughcolor=BG2,
                arrowcolor=FG_DIM, borderwidth=0, relief="flat")
    s.map("Vertical.TScrollbar", background=[("active", SEL)])
    s.configure("TCombobox", fieldbackground=SEL, background=SEL, foreground=FG,
                selectbackground=ACCENT, selectforeground=BG,
                arrowcolor=FG, borderwidth=1, relief="flat")
    s.map("TCombobox", fieldbackground=[("readonly", SEL)], background=[("readonly", SEL)])


class _RoundedButton(tk.Canvas):
    """Canvas-backed button: 1-px border, slightly rounded corners, no system chrome."""

    def __init__(self, master, text: str, command: Callable,
                 bg: str, fg: str, hover: str, font,
                 padx: int = 10, pady: int = 5) -> None:
        self._bg    = bg
        self._fg    = fg
        self._hover = hover
        self._font  = font
        self._text  = text
        self._padx  = padx
        self._pady  = pady

        # Measure natural text size via a throwaway label.
        _m = tk.Label(master, text=text, font=font, padx=0, pady=0)
        _m.update_idletasks()
        self._tw = _m.winfo_reqwidth()
        self._th = _m.winfo_reqheight()
        _m.destroy()

        try:
            pbg = master.cget("bg")
        except Exception:
            pbg = BG

        super().__init__(master,
                         width=self._tw + padx * 2,
                         height=self._th + pady * 2,
                         highlightthickness=0, bd=0, bg=pbg, cursor="hand2")
        self.bind("<Enter>",     lambda _e: self._paint(self._hover))
        self.bind("<Leave>",     lambda _e: self._paint(self._bg))
        self.bind("<Button-1>",  lambda _e: command())
        self.bind("<Configure>", lambda _e: self._paint(self._bg))

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _paint(self, fill: str) -> None:
        self.delete("all")
        actual_w = self.winfo_width()
        w = max(4, actual_w if actual_w > 1 else int(self["width"]))
        h = max(4, int(self["height"]))
        r = _RADIUS
        # Smooth polygon → rounded rectangle with 1-px border
        pts = [
            r, 1,     w-r, 1,
            w-1, 1,   w-1, r,
            w-1, h-r, w-1, h-1,
            w-r, h-1, r,   h-1,
            1,   h-1, 1,   h-r,
            1,   r,   1,   1,
            r,   1,
        ]
        self.create_polygon(pts, smooth=True,
                            fill=fill, outline=BORDER, width=1)
        self.create_text(w // 2, h // 2,
                         text=self._text, fill=self._fg, font=self._font)

    # ------------------------------------------------------------------
    # configure / config — intercept Button-style kwargs
    # ------------------------------------------------------------------

    def configure(self, cnf=None, **kw) -> None:
        # tkinter may call configure({'key': value}) with a positional dict
        if cnf:
            kw.update(cnf)

        if "width" in kw:
            # Callers pass character-count widths (tk.Button convention).
            # Convert using the measured char height as a proxy for char width.
            char_w = int(kw.pop("width"))
            px_w = char_w * max(6, round(self._th * 0.55)) + self._padx * 2
            tk.Canvas.configure(self, width=px_w)

        if "pady" in kw:
            self._pady = int(kw.pop("pady"))
            tk.Canvas.configure(self, height=self._th + self._pady * 2)

        if "text" in kw:
            self._text = kw.pop("text")
            self._paint(self._bg)

        if "bg" in kw:
            self._bg = kw.pop("bg")
            # Auto-flip text colour to maintain contrast when bg changes
            if "fg" not in kw:
                self._fg = BG if _is_dark(self._bg) else FG
            self._paint(self._bg)

        if "fg" in kw:
            self._fg = kw.pop("fg")
            self._paint(self._bg)

        # Silently absorb Button-only kwargs that have no Canvas equivalent
        for k in ("padx", "relief", "activebackground", "activeforeground",
                  "highlightthickness", "highlightbackground",
                  "anchor", "justify"):
            kw.pop(k, None)

        if kw:
            tk.Canvas.configure(self, **kw)

    config = configure


def btn(parent, text: str, command: Callable, accent=False, danger=False,
        width: Optional[int] = None, **kw) -> _RoundedButton:
    """Rounded button with a 1-px border."""
    if accent:
        bg, hover, fg = ACCENT, ACCENT2, BG
    elif danger:
        bg, hover, fg = DANGER, DANGER2, FG
    else:
        bg, hover, fg = SEL, SEL_HOVER, FG

    b = _RoundedButton(parent, text=text, command=command,
                       bg=bg, fg=fg, hover=hover, font=FONT)
    if width is not None:
        b.configure(width=width)
    return b


def lbl(parent, text: str, dim=False, bold=False, title=False,
        bg: Optional[str] = None, **kw) -> tk.Label:
    """Styled label."""
    fg = FG_DIM if dim else FG
    font = FONT_TITLE if title else (FONT_BOLD if bold else FONT)
    return tk.Label(parent, text=text, bg=bg or BG, fg=fg, font=font, **kw)


def entry(parent, textvariable=None, **kw) -> tk.Entry:
    """Entry widget."""
    return tk.Entry(parent, textvariable=textvariable, bg=BG, fg=FG,
                    insertbackground=FG, relief="flat",
                    highlightthickness=1, highlightbackground=BORDER,
                    highlightcolor=ACCENT, font=FONT, **kw)


def dropdown(parent, variable: tk.StringVar, values: list[str],
             command: Optional[Callable] = None, **kw) -> tk.OptionMenu:
    """Monochrome option menu."""
    choices = values or ["(none)"]
    if variable.get() not in choices:
        variable.set(choices[0])
    m = tk.OptionMenu(parent, variable, *choices, command=command)
    m.configure(bg=SEL, fg=FG, activebackground=SEL_HOVER, activeforeground=FG,
                relief="flat", highlightthickness=1,
                highlightbackground=BORDER, cursor="hand2",
                font=FONT, indicatoron=True, **kw)
    m["menu"].configure(bg=BG, fg=FG, activebackground=ACCENT,
                        activeforeground=BG, relief="flat", bd=0, font=FONT)
    return m


def separator(parent, **kw) -> tk.Frame:
    """1-px horizontal separator."""
    return tk.Frame(parent, bg=BORDER, height=1, **kw)


class Card(tk.Frame):
    """Frame with secondary background colour."""

    def __init__(self, master, **kw):
        super().__init__(master, bg=BG2, **kw)


class Scrollable(tk.Frame):
    """Vertically scrollable container. Add child widgets to ``.inner``."""

    _BAR_W = 6  # scrollbar track width in pixels

    def __init__(self, master, bg: str = BG, **kw):
        super().__init__(master, bg=bg, **kw)

        self._cv = tk.Canvas(self, bg=bg, highlightthickness=0)
        self._cv.pack(side="left", fill="both", expand=True)

        # Minimal canvas-drawn scrollbar
        self._bar_cv = tk.Canvas(self, bg=BG2, width=self._BAR_W,
                                 highlightthickness=0, bd=0)
        self._bar_cv.pack(side="right", fill="y")

        self.inner = tk.Frame(self._cv, bg=bg)
        self._win  = self._cv.create_window(0, 0, anchor="nw", window=self.inner)

        self.inner.bind("<Configure>", lambda _e: self._update_scroll())
        self._cv.bind("<Configure>",   lambda  e: (
            self._cv.itemconfig(self._win, width=e.width),
            self._update_scroll(),
        ))
        self._bar_cv.bind("<Configure>", lambda _e: self._draw_thumb())
        self._bar_cv.bind("<ButtonPress-1>",   self._bar_click)
        self._bar_cv.bind("<B1-Motion>",       self._bar_drag)
        self._bar_cv.bind("<ButtonRelease-1>", lambda _e: None)

        for w in (self._cv, self.inner):
            w.bind("<MouseWheel>", self._on_scroll)
            w.bind("<Button-4>",   self._on_scroll)
            w.bind("<Button-5>",   self._on_scroll)

        self._yview = (0.0, 1.0)  # (top, bottom) fractions
        self._drag_start_y = 0
        self._drag_start_top = 0.0

    def _update_scroll(self) -> None:
        self._cv.configure(scrollregion=self._cv.bbox("all"))
        self._yview = self._cv.yview()
        self._draw_thumb()

    def _draw_thumb(self) -> None:
        self._bar_cv.delete("all")
        h = self._bar_cv.winfo_height()
        if h < 2:
            return
        top, bot = self._yview
        if bot - top >= 1.0:
            return  # content fits, no thumb needed
        ty = int(top * h)
        by = int(bot * h)
        r  = self._BAR_W // 2
        if by - ty < self._BAR_W:
            by = ty + self._BAR_W
        self._bar_cv.create_rectangle(1, ty, self._BAR_W - 1, by,
                                      fill=BORDER, outline="", width=0)

    def _on_scroll(self, e) -> None:
        if self._yview[0] <= 0.0 and self._yview[1] >= 1.0:
            return  # content fits; nothing to scroll
        delta = -1 if (e.num == 4 or getattr(e, "delta", 0) > 0) else 1
        self._cv.yview_scroll(delta * 3, "units")
        self._yview = self._cv.yview()
        self._draw_thumb()

    def _bar_click(self, e) -> None:
        self._drag_start_y   = e.y
        self._drag_start_top = self._yview[0]
        h = self._bar_cv.winfo_height()
        if h > 0:
            frac = e.y / h
            self._cv.yview_moveto(frac)
            self._yview = self._cv.yview()
            self._draw_thumb()

    def _bar_drag(self, e) -> None:
        h = self._bar_cv.winfo_height()
        if h <= 0:
            return
        delta = (e.y - self._drag_start_y) / h
        self._cv.yview_moveto(self._drag_start_top + delta)
        self._yview = self._cv.yview()
        self._draw_thumb()
