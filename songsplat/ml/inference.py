"""Run model inference on a song's chunks."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from songsplat.audio.loader import get_chunk_audio, TARGET_SR
from songsplat.core.models import ModelCheckpoint, Song, Splat
from songsplat.ml.models import build_model


def run_inference(
    song: Song,
    checkpoint: ModelCheckpoint,
    splats: list[Splat],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict[str, list[float]]:
    """Run the model on all chunks of a song.

    Writes predictions into chunk.predictions[splat_id] and returns a dict
    mapping splat_id -> list of predicted values (one per chunk).
    """
    if not song.chunks:
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_splats = len(splats)
    splat_ids = [s.id for s in splats]

    model = build_model(checkpoint.architecture, num_splats=num_splats)
    state = torch.load(checkpoint.path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    results: dict[str, list[float]] = {sid: [] for sid in splat_ids}
    n = len(song.chunks)

    with torch.no_grad():
        for i, chunk in enumerate(song.chunks):
            audio = get_chunk_audio(song, chunk, target_sr=TARGET_SR)
            t = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, T)
            preds = model(t)  # (1, num_splats)
            for j, sid in enumerate(splat_ids):
                val = float(preds[0, j].item())
                chunk.predictions[sid] = val
                results[sid].append(val)

            if progress_cb:
                progress_cb(i + 1, n)

    return results
