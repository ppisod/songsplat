"""getting_started/04_run_inference.py

Load a trained project and run inference on all chunks of a song.
Predictions are written into chunk.predictions[splat_id].

Usage:
    python 04_run_inference.py demo_project.splatject
"""

import sys
from songsplat.core.project_io import load_project
from songsplat.ml.inference import run_inference


def main(project_path: str) -> None:
    project = load_project(project_path)
    print(f"Project: {project.name}")

    if not project.best_checkpoint:
        print("No trained checkpoint found. Run 03_train_model.py first.")
        sys.exit(1)

    if not project.songs:
        print("No songs in project.")
        sys.exit(1)

    song = project.songs[0]
    if not song.chunks:
        print("Song has no chunks. Chunk it first.")
        sys.exit(1)

    print(f"Running inference on '{song.name}' ({len(song.chunks)} chunks) ...")
    results = run_inference(
        song=song,
        checkpoint=project.best_checkpoint,
        splats=project.splats,
        progress_cb=lambda i, n: print(f"\r  {i}/{n}", end="", flush=True),
    )
    print()

    splat_names = {s.id: s.name for s in project.splats}
    print(f"\n{'Chunk':>6}  {'Time':>12}  " +
          "  ".join(f"{splat_names.get(sid, sid):<12}" for sid in results))
    for ck in song.chunks[:10]:
        vals = "  ".join(
            f"{ck.predictions.get(sid, 0.0):<12.3f}" for sid in results)
        print(f"{ck.index:>6}  {ck.start:6.2f}-{ck.end:<6.2f}  {vals}")
    if len(song.chunks) > 10:
        print(f"  ... ({len(song.chunks)} total chunks)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 04_run_inference.py <project.splatject>")
        sys.exit(1)
    main(sys.argv[1])
