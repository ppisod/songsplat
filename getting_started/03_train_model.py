"""getting_started/03_train_model.py

Train a splat prediction model from labeled chunks inside a .splatject project.

Requires: torch, torchaudio, transformers
Usage:
    python 03_train_model.py demo_project.splatject
"""

import sys
from songsplat.core.project_io import load_project
from songsplat.ml.trainer import Trainer, TrainerConfig


def main(project_path: str) -> None:
    project = load_project(project_path)
    print(f"Project: {project.name}")

    labeled = sum(1 for s in project.songs for c in s.chunks if c.labels)
    if labeled == 0:
        print("No labeled chunks found. Label some chunks first (GUI or script).")
        sys.exit(1)
    print(f"Training on {labeled} labeled chunks, {len(project.splats)} splats")

    cfg               = TrainerConfig()
    cfg.architecture  = "pretrained"
    cfg.epochs        = 5
    cfg.lr            = 1e-3
    cfg.batch_size    = 4

    trainer = Trainer(project=project, config=cfg)

    trainer.on_epoch_end = lambda ep, tl, vl, _psl: print(
        f"  epoch {ep + 1}/{cfg.epochs}  train={tl:.4f}  val={vl:.4f}")
    trainer.on_error     = lambda msg: (print(f"Error: {msg}"), sys.exit(1))

    best_ckpt = None

    def on_finished(ckpt):
        nonlocal best_ckpt
        best_ckpt = ckpt
        print(f"Training complete. Best val_loss={ckpt.loss:.4f}  -> {ckpt.path}")

    trainer.on_finished = on_finished
    trainer.start()
    if trainer._thread:
        trainer._thread.join()

    if best_ckpt:
        project.best_checkpoint = best_ckpt
        from songsplat.core.project_io import save_project
        save_project(project, project_path)
        print(f"Checkpoint saved to project: {project_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 03_train_model.py <project.splatject>")
        sys.exit(1)
    main(sys.argv[1])
