"""songsplat splatter - TUI for labeling chunks with splat values -> .splatdata

Controls:
  Left/Right arrows  : previous / next chunk
  Up/Down arrows     : increase / decrease value for current splat (step 0.05)
  Tab / Shift-Tab    : next / previous splat
  1-9                : jump to splat by number
  s                  : save
  q / Escape         : save and quit
  ?                  : toggle help
"""

from __future__ import annotations

import curses
import os
import sys
from pathlib import Path
from typing import Optional

from songsplat.cli.formats import SplatChunk, SplatData, SplatDef


NUDGE = 0.05


def run_splatter(chunk_path: str, output: str = "", existing_data: str = "") -> str:
    """Open the TUI labeler. Returns path to saved .splatdata file."""
    chunk_path = os.path.abspath(chunk_path)
    if not os.path.isfile(chunk_path):
        raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

    sc = SplatChunk.load(chunk_path)
    if not sc.chunks:
        raise RuntimeError("No chunks in file. Run 'songsplat chunk' first.")

    if existing_data and os.path.isfile(existing_data):
        sd = SplatData.load(existing_data)
    else:
        sd = SplatData(chunk_file=chunk_path)
        # Prompt for splat definitions before entering TUI
        sd.splats = _prompt_splats()

    if not sd.splats:
        print("No splats defined. Exiting.")
        sys.exit(1)

    if not output:
        output = str(Path(os.getcwd()) / (Path(chunk_path).stem + ".splatdata"))

    curses.wrapper(_tui_main, sc, sd, output)
    return output


def _prompt_splats() -> list[SplatDef]:
    """Simple text-mode splat setup before entering curses."""
    print("\nDefine splats (blank name to finish):")
    splats = []
    i = 1
    while True:
        name = input(f"  Splat {i} name: ").strip()
        if not name:
            break
        low = input(f"    Low label (e.g. 'calm'): ").strip()
        high = input(f"    High label (e.g. 'energetic'): ").strip()
        splats.append(SplatDef(name=name, low_label=low, high_label=high))
        i += 1
    return splats


def _tui_main(stdscr, sc: SplatChunk, sd: SplatData, output: str) -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()

    curses.init_pair(1, curses.COLOR_WHITE, -1)
    curses.init_pair(2, curses.COLOR_CYAN, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_GREEN, -1)
    curses.init_pair(5, curses.COLOR_RED, -1)
    curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(7, curses.COLOR_MAGENTA, -1)

    state = _State(sc=sc, sd=sd, n_chunks=len(sc.chunks), n_splats=len(sd.splats))
    show_help = False

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        if show_help:
            _draw_help(stdscr, h, w)
        else:
            _draw_main(stdscr, h, w, state, sc, sd)

        stdscr.refresh()
        key = stdscr.getch()

        if key in (ord("q"), 27):  # q or Escape
            _save(sd, output, stdscr, w)
            break
        elif key == ord("s"):
            _save(sd, output, stdscr, w)
        elif key == ord("?"):
            show_help = not show_help
        elif not show_help:
            if key == curses.KEY_RIGHT:
                state.chunk_idx = min(state.n_chunks - 1, state.chunk_idx + 1)
                state.scroll_chunks_to(state.chunk_idx, h)
            elif key == curses.KEY_LEFT:
                state.chunk_idx = max(0, state.chunk_idx - 1)
                state.scroll_chunks_to(state.chunk_idx, h)
            elif key == curses.KEY_UP:
                _nudge(sd, state, +NUDGE)
            elif key == curses.KEY_DOWN:
                _nudge(sd, state, -NUDGE)
            elif key == ord("\t"):
                state.splat_idx = (state.splat_idx + 1) % state.n_splats
            elif key == curses.KEY_BTAB:
                state.splat_idx = (state.splat_idx - 1) % state.n_splats
            elif ord("1") <= key <= ord("9"):
                n = key - ord("1")
                if n < state.n_splats:
                    state.splat_idx = n


class _State:
    def __init__(self, sc, sd, n_chunks, n_splats):
        self.sc = sc
        self.sd = sd
        self.chunk_idx = 0
        self.splat_idx = 0
        self.n_chunks = n_chunks
        self.n_splats = n_splats
        self.chunk_scroll = 0  # top visible chunk index

    def scroll_chunks_to(self, idx: int, h: int) -> None:
        visible = max(1, h - 12)
        if idx < self.chunk_scroll:
            self.chunk_scroll = idx
        elif idx >= self.chunk_scroll + visible:
            self.chunk_scroll = idx - visible + 1


def _nudge(sd: SplatData, state: _State, delta: float) -> None:
    splat = sd.splats[state.splat_idx]
    current = sd.get_label(state.chunk_idx, splat.id) or 0.5
    new_val = max(0.0, min(1.0, current + delta))
    sd.set_label(state.chunk_idx, splat.id, new_val)


def _save(sd: SplatData, output: str, stdscr, w: int) -> None:
    sd.save(output)
    msg = f" Saved to {output} "
    h, _ = stdscr.getmaxyx()
    stdscr.addstr(h - 1, 0, msg[:w - 1], curses.color_pair(4) | curses.A_BOLD)
    stdscr.refresh()
    curses.napms(800)


def _draw_main(stdscr, h: int, w: int, state: _State, sc: SplatChunk, sd: SplatData) -> None:
    C = curses.color_pair

    # Header
    title = f" songsplat splatter  |  {sc.song_name}  |  {len(sc.chunks)} chunks "
    stdscr.addstr(0, 0, title[:w].ljust(min(w, 80)), C(2) | curses.A_BOLD)

    # Current chunk info
    ck = sc.chunks[state.chunk_idx]
    chunk_info = (f" Chunk {state.chunk_idx + 1}/{state.n_chunks}"
                  f"  [{ck.start:.2f}s - {ck.end:.2f}s"
                  f"  dur={ck.end - ck.start:.2f}s] ")
    stdscr.addstr(1, 0, chunk_info[:w], C(3) | curses.A_BOLD)

    # Splat values for current chunk
    splat_panel_h = state.n_splats + 2
    stdscr.addstr(2, 0, " Splat values ", C(2))
    for i, splat in enumerate(sd.splats):
        val = sd.get_label(state.chunk_idx, splat.id)
        active = (i == state.splat_idx)
        row = 3 + i
        if row >= h - 3:
            break
        _draw_splat_row(stdscr, row, w, splat, i, val, active)

    sep_row = 3 + state.n_splats + 1
    if sep_row < h - 4:
        stdscr.addstr(sep_row, 0, "-" * min(w, 60), C(1))

    # Chunk list
    list_top = sep_row + 1
    list_h = h - list_top - 2
    if list_h > 0:
        stdscr.addstr(list_top - 1 if list_top > 0 else 0, 0,
                      " Chunks (Left/Right to navigate) ", C(2))
        for row_i in range(list_h):
            ci = state.chunk_scroll + row_i
            if ci >= state.n_chunks:
                break
            ck_i = sc.chunks[ci]
            labeled_count = sum(
                1 for sp in sd.splats
                if sd.get_label(ci, sp.id) is not None
            )
            total_splats = len(sd.splats)
            active = (ci == state.chunk_idx)
            attr = C(3) | curses.A_BOLD if active else C(1)
            prefix = "> " if active else "  "
            line = (f"{prefix}{ci + 1:4d}  [{ck_i.start:7.2f}s-{ck_i.end:7.2f}s]"
                    f"  {labeled_count}/{total_splats} labeled")
            y = list_top + row_i
            if y < h - 1:
                stdscr.addstr(y, 0, line[:w - 1], attr)

    # Status bar
    splat_name = sd.splats[state.splat_idx].name
    status = (f" Active splat: [{state.splat_idx + 1}] {splat_name}"
              f"  |  Up/Down: adjust  Tab: next splat  s: save  q: quit  ?: help ")
    if h > 1:
        stdscr.addstr(h - 1, 0, status[:w - 1], C(6))


def _draw_splat_row(stdscr, row: int, w: int, splat: SplatDef, idx: int,
                    val: Optional[float], active: bool) -> None:
    C = curses.color_pair
    attr = C(3) | curses.A_BOLD if active else C(1)
    marker = "*" if active else " "
    name_col = f"{marker}[{idx + 1}] {splat.name:<16}"

    if val is not None:
        bar_w = min(30, w - 50)
        filled = max(0, min(bar_w, int(val * bar_w)))
        bar = "[" + "#" * filled + "." * (bar_w - filled) + "]"
        labels = f"{splat.low_label or '0':>8} {bar} {splat.high_label or '1':<8}  {val:.3f}"
    else:
        labels = "  (unlabeled - Up/Down to set)"

    line = (name_col + labels)[:w - 1]
    stdscr.addstr(row, 0, line, attr)


def _draw_help(stdscr, h: int, w: int) -> None:
    C = curses.color_pair
    lines = [
        "",
        "  songsplat splatter - keyboard shortcuts",
        "",
        "  Left / Right      Previous / next chunk",
        "  Up / Down         Increase / decrease value (step 0.05)",
        "  Tab               Next splat",
        "  Shift-Tab         Previous splat",
        "  1-9               Jump to splat by number",
        "  s                 Save",
        "  q / Escape        Save and quit",
        "  ?                 Toggle this help",
        "",
        "  Press any key to return.",
    ]
    for i, line in enumerate(lines):
        if i < h - 1:
            stdscr.addstr(i, 0, line[:w - 1], C(2) if i == 1 else C(1))
