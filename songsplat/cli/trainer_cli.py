"""songsplat train - train a neural network from .splatdata files -> .splat"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from songsplat.cli.formats import SplatChunk, SplatData


def run_train(
    splatdata_paths: list[str],
    output: str = "",
    architecture: str = "pretrained",
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 8,
) -> str:
    """Train a model from .splatdata files. Returns path to output .splat file."""
    if not splatdata_paths:
        raise ValueError("Provide at least one .splatdata file")

    all_sd = []
    for p in splatdata_paths:
        p = os.path.abspath(p)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")
        all_sd.append(SplatData.load(p))

    project = _build_project(all_sd)
    if not project.splats:
        raise RuntimeError("No splats found in data files")

    labeled = sum(
        1 for s in project.songs for c in s.chunks if c.labels
    )
    if labeled == 0:
        raise RuntimeError("No labeled chunks. Label chunks with 'songsplat splatter' first.")

    print(f"Training on {labeled} labeled chunks across {len(project.songs)} songs...")
    print(f"Architecture: {architecture}  Epochs: {epochs}  LR: {lr}  Batch: {batch_size}")

    from songsplat.ml.trainer import Trainer, TrainerConfig
    cfg = TrainerConfig()
    cfg.architecture = architecture
    cfg.epochs = epochs
    cfg.lr = lr
    cfg.batch_size = batch_size

    last_epoch = [0]
    last_loss = [float("inf")]

    def on_epoch(ep, tl, vl, psl):
        last_epoch[0] = ep + 1
        last_loss[0] = vl
        bar_w = 30
        filled = int((ep + 1) / epochs * bar_w)
        bar = "#" * filled + "." * (bar_w - filled)
        print(f"\r  [{bar}] {ep + 1}/{epochs}  val_loss={vl:.4f}", end="", flush=True)

    def on_finished(ckpt):
        print(f"\nTraining done. Best val_loss={ckpt.loss:.4f}  saved to {ckpt.path}")
        _export_splat(ckpt, project.splats, output or _default_output(splatdata_paths[0]))

    def on_error(msg):
        print(f"\nTraining error: {msg}", file=sys.stderr)
        sys.exit(1)

    trainer = Trainer(project=project, config=cfg)
    trainer.on_epoch_end = on_epoch
    trainer.on_finished = on_finished
    trainer.on_error = on_error
    trainer.start()
    if trainer._thread:
        trainer._thread.join()

    out = output or _default_output(splatdata_paths[0])
    return out


def _build_project(all_sd: list[SplatData]):
    """Convert .splatdata files into a Project for the trainer."""
    from songsplat.core.models import Project, Splat, Song, Chunk

    # Collect splats (from first file that has them)
    splats_by_id: dict[str, Splat] = {}
    for sd in all_sd:
        for sdef in sd.splats:
            if sdef.id not in splats_by_id:
                splats_by_id[sdef.id] = Splat(
                    id=sdef.id,
                    name=sdef.name,
                    low_label=sdef.low_label,
                    high_label=sdef.high_label,
                    order=len(splats_by_id),
                )

    project = Project(splats=list(splats_by_id.values()))

    for sd in all_sd:
        sc = SplatChunk.load(sd.chunk_file) if os.path.isfile(sd.chunk_file) else None
        if sc is None:
            print(f"Warning: chunk file not found: {sd.chunk_file}", file=sys.stderr)
            continue

        song = Song(
            path=sc.source_path,
            name=sc.song_name,
            sample_rate=sc.sample_rate,
            duration=sc.duration,
        )
        for ci in sc.chunks:
            chunk = Chunk(index=ci.index, start=ci.start, end=ci.end)
            row = sd.labels.get(str(ci.index), {})
            chunk.labels = dict(row)
            song.chunks.append(chunk)

        project.songs.append(song)

    return project


def _default_output(splatdata_path: str) -> str:
    return str(Path(os.getcwd()) / (Path(splatdata_path).stem + ".splat"))


def _export_splat(ckpt, splats, output: str) -> None:
    import json, zipfile
    meta = {
        "splats": [{"id": s.id, "name": s.name,
                    "low_label": s.low_label, "high_label": s.high_label}
                   for s in splats],
        "architecture": ckpt.architecture,
        "epoch": ckpt.epoch,
        "loss": ckpt.loss,
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta, indent=2))
        if os.path.isfile(ckpt.path):
            zf.write(ckpt.path, "model.pt")
    print(f"Exported to {output}")
