"""songsplat test - run a .splat model on a chunk file or audio file."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from songsplat.cli.formats import SplatChunk, SplatFile


def run_test(splat_path: str, input_path: str) -> None:
    """Run *splat_path* against *input_path* (.splatchunk or raw audio)."""
    splat_path = os.path.abspath(splat_path)
    input_path = os.path.abspath(input_path)

    if not os.path.isfile(splat_path):
        raise FileNotFoundError(f"Splat file not found: {splat_path}")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    sf = SplatFile.load(splat_path)

    cleanup = False
    if Path(input_path).suffix.lower() != ".splatchunk":
        from songsplat.cli.chunker import run_chunker
        tmp = str(Path(os.getcwd()) / (Path(input_path).stem + ".tmp.splatchunk"))
        print("Auto-chunking input audio...", flush=True)
        run_chunker(input_path, output=tmp)
        input_path, cleanup = tmp, True

    sc = SplatChunk.load(input_path)
    print(f"Running inference on {len(sc.chunks)} chunks of '{sc.song_name}'...")

    results = _infer(sf, sc)
    _print_results(results, sc, sf.splat_defs)

    if cleanup:
        try:
            os.remove(input_path)
        except Exception:
            pass


def _infer(sf: SplatFile, sc: SplatChunk) -> list[dict]:
    """Return ``[{splat_name: value, ...}]`` for every chunk in *sc*."""
    if not os.path.isfile(sf.model_path):
        print("No model weights - showing zeros", file=sys.stderr)
        return [{sp["name"]: 0.0 for sp in sf.splat_defs} for _ in sc.chunks]

    import torch
    from songsplat.audio.loader import TARGET_SR, get_chunk_audio
    from songsplat.core.models import Chunk as ModelChunk
    from songsplat.core.models import Song as ModelSong
    from songsplat.ml.models import build_model

    model = build_model(sf.architecture, num_splats=len(sf.splat_defs))
    state = torch.load(sf.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    song = ModelSong(path=sc.source_path, sample_rate=sc.sample_rate, duration=sc.duration)
    results = []
    with torch.no_grad():
        for ci in sc.chunks:
            audio  = get_chunk_audio(song, ModelChunk(index=ci.index, start=ci.start, end=ci.end),
                                     target_sr=TARGET_SR)
            preds  = model(torch.from_numpy(audio).unsqueeze(0))
            results.append({
                sp["name"]: round(float(preds[0, j].item()), 4)
                for j, sp in enumerate(sf.splat_defs)
            })
    return results


def _print_results(results: list[dict], sc: SplatChunk, splat_defs: list) -> None:
    if not results:
        return
    names = [sp["name"] for sp in splat_defs]
    col_w = max(10, max(len(n) for n in names) + 2)
    header = f"{'Chunk':>6}  {'Time':>12}  " + "  ".join(n.ljust(col_w) for n in names)
    print(header)
    print("-" * len(header))
    for i, (chunk, row) in enumerate(zip(sc.chunks, results)):
        time_str = f"{chunk.start:.2f}-{chunk.end:.2f}s"
        vals = "  ".join(f"{row.get(n, 0.0):.3f}".ljust(col_w) for n in names)
        print(f"{i + 1:>6}  {time_str:>12}  {vals}")
    print(f"\n{len(results)} chunks processed.")
