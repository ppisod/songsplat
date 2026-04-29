"""songsplat reinforce - review and correct model predictions interactively."""

from __future__ import annotations

import curses
import os
import sys
from pathlib import Path
from typing import Optional

from songsplat.cli.formats import SplatChunk, SplatData, SplatDef, SplatFile

NUDGE = 0.05


def run_reinforce(splat_path: str, chunk_paths: list[str]) -> None:
    """Load *splat_path*, run inference on *chunk_paths*, open review TUI."""
    splat_path = os.path.abspath(splat_path)
    if not os.path.isfile(splat_path):
        raise FileNotFoundError(f"Splat file not found: {splat_path}")

    sf = SplatFile.load(splat_path)
    splat_defs = [SplatDef(**s) for s in sf.splat_defs]

    chunk_files = []
    for p in chunk_paths:
        p = os.path.abspath(p)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Chunk file not found: {p}")
        chunk_files.append(SplatChunk.load(p))

    if not chunk_files:
        raise ValueError("No chunk files provided")

    print("Running inference...", flush=True)
    predictions = _run_inference(sf, splat_defs, chunk_files)

    sd = SplatData(chunk_file=chunk_files[0].source_path, splats=splat_defs)
    for fi, sc in enumerate(chunk_files):
        for chunk in sc.chunks:
            for j, sp in enumerate(splat_defs):
                pred = predictions.get((fi, chunk.index), {}).get(sp.id)
                if pred is not None:
                    sd.set_label(chunk.index, sp.id, pred)

    output = str(Path(os.getcwd()) / (
        Path(chunk_files[0].source_path).stem + ".reinforced.splatdata"))
    curses.wrapper(_tui, chunk_files[0], sd, output)


def _run_inference(sf: SplatFile, splat_defs: list[SplatDef],
                   chunk_files: list[SplatChunk]) -> dict:
    """Return ``{(file_idx, chunk_idx): {splat_id: value}}``."""
    if not os.path.isfile(sf.model_path):
        print("Warning: no model weights - using zeros", file=sys.stderr)
        return {}

    import torch
    from songsplat.audio.loader import TARGET_SR, get_chunk_audio
    from songsplat.core.models import Chunk as MC
    from songsplat.core.models import Song as MS
    from songsplat.ml.models import build_model

    model = build_model(sf.architecture, num_splats=len(splat_defs))
    state = torch.load(sf.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    results = {}
    total   = sum(len(sc.chunks) for sc in chunk_files)
    done    = 0

    with torch.no_grad():
        for fi, sc in enumerate(chunk_files):
            song = MS(path=sc.source_path, sample_rate=sc.sample_rate, duration=sc.duration)
            for ci in sc.chunks:
                audio = get_chunk_audio(song, MC(index=ci.index, start=ci.start, end=ci.end),
                                        target_sr=TARGET_SR)
                preds = model(torch.from_numpy(audio).unsqueeze(0))
                results[(fi, ci.index)] = {
                    sp.id: float(preds[0, j].item())
                    for j, sp in enumerate(splat_defs)
                }
                done += 1
                print(f"\r  {done}/{total}", end="", flush=True)
    print()
    return results


def _tui(stdscr, sc: SplatChunk, sd: SplatData, output: str) -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    for i, (fg, bg) in enumerate([
        (curses.COLOR_WHITE,   -1),
        (curses.COLOR_CYAN,    -1),
        (curses.COLOR_YELLOW,  -1),
        (curses.COLOR_GREEN,   -1),
        (curses.COLOR_RED,     -1),
        (curses.COLOR_BLACK,   curses.COLOR_WHITE),
    ], 1):
        curses.init_pair(i, fg, bg)

    C         = curses.color_pair
    chunk_idx = 0
    splat_idx = 0
    n_chunks  = len(sc.chunks)
    n_splats  = len(sd.splats)

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        ck    = sc.chunks[chunk_idx]

        stdscr.addstr(0, 0,
            f" songsplat reinforce  |  {sc.song_name}  |  "
            f"chunk {chunk_idx + 1}/{n_chunks} "[:w], C(2) | curses.A_BOLD)
        stdscr.addstr(1, 0, f" [{ck.start:.2f}s - {ck.end:.2f}s] "[:w], C(3))
        stdscr.addstr(2, 0, " Predictions (Up/Down to correct): "[:w], C(2))

        for i, splat in enumerate(sd.splats):
            val    = sd.get_label(chunk_idx, splat.id) or 0.0
            active = (i == splat_idx)
            attr   = C(3) | curses.A_BOLD if active else C(1)
            bar_w  = min(25, w - 50)
            filled = max(0, min(bar_w, int(val * bar_w)))
            bar    = "[" + "#" * filled + "." * (bar_w - filled) + "]"
            marker = "*" if active else " "
            line   = (f"{marker}[{i+1}] {splat.name:<16} "
                      f"{splat.low_label or '0':>8} {bar} "
                      f"{splat.high_label or '1':<8}  {val:.3f}")
            if 3 + i < h - 2:
                stdscr.addstr(3 + i, 0, line[:w - 1], attr)

        stdscr.addstr(h - 1, 0,
            " Left/Right:chunk  Up/Down:adjust  Tab:splat  s:save  q:quit "[:w - 1], C(6))
        stdscr.refresh()
        key = stdscr.getch()

        if key in (ord("q"), 27):
            sd.save(output)
            break
        elif key == ord("s"):
            sd.save(output)
            stdscr.addstr(h - 1, 0, f" Saved to {output} "[:w - 1], C(4) | curses.A_BOLD)
            stdscr.refresh()
            curses.napms(600)
        elif key == curses.KEY_RIGHT:
            chunk_idx = min(n_chunks - 1, chunk_idx + 1)
        elif key == curses.KEY_LEFT:
            chunk_idx = max(0, chunk_idx - 1)
        elif key == curses.KEY_UP:
            sp  = sd.splats[splat_idx]
            cur = sd.get_label(chunk_idx, sp.id) or 0.0
            sd.set_label(chunk_idx, sp.id, min(1.0, cur + NUDGE))
        elif key == curses.KEY_DOWN:
            sp  = sd.splats[splat_idx]
            cur = sd.get_label(chunk_idx, sp.id) or 0.0
            sd.set_label(chunk_idx, sp.id, max(0.0, cur - NUDGE))
        elif key == ord("\t"):
            splat_idx = (splat_idx + 1) % n_splats
        elif key == curses.KEY_BTAB:
            splat_idx = (splat_idx - 1) % n_splats
        elif ord("1") <= key <= ord("9"):
            n = key - ord("1")
            if n < n_splats:
                splat_idx = n
