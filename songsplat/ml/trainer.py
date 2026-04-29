"""Training loop - runs in a background thread, emits progress via callbacks."""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from songsplat.core.models import ModelCheckpoint, Project
from songsplat.ml.dataset import ChunkDataset
from songsplat.ml.models import build_model

# Checkpoint directory
_CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), ".songsplat", "checkpoints")


class TrainerConfig:
    architecture: str = "pretrained"   # or "raw_transformer"
    lr: float = 1e-4
    batch_size: int = 8
    epochs: int = 30
    val_split: float = 0.1
    chunk_samples: int = 44100         # ~2 s at 22050 Hz


def _nan_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss ignoring NaN targets (unlabeled splats)."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return nn.functional.mse_loss(pred[mask], target[mask])


class Trainer:
    """Runs training in a background thread.

    Callbacks (called from the training thread - wire to Qt slots via signals):
        on_epoch_end(epoch, train_loss, val_loss, loss_per_splat)
        on_finished(checkpoint: ModelCheckpoint)
        on_error(message: str)
    """

    def __init__(
        self,
        project: Project,
        config: Optional[TrainerConfig] = None,
    ) -> None:
        self._project = project
        self._config = config or TrainerConfig()
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        self._pause_flag.set()  # not paused initially

        self.on_epoch_end: Optional[Callable] = None
        self.on_finished: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._pause_flag.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        self._pause_flag.set()  # unblock if paused

    def pause(self) -> None:
        self._pause_flag.clear()

    def resume(self) -> None:
        self._pause_flag.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Training loop (runs in background thread)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        cfg = self._config
        try:
            dataset = ChunkDataset(
                self._project,
                chunk_samples=cfg.chunk_samples,
            )
            if len(dataset) == 0:
                raise RuntimeError("No labeled chunks found. Label some chunks before training.")

            n_val = max(1, int(len(dataset) * cfg.val_split))
            n_train = len(dataset) - n_val
            train_ds, val_ds = random_split(dataset, [n_train, n_val])

            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = build_model(cfg.architecture, num_splats=dataset.num_splats)
            model.to(device)

            if cfg.architecture == "pretrained":
                params = [p for p in model.heads.parameters()]
            else:
                params = list(model.parameters())

            optimizer = torch.optim.AdamW(params, lr=cfg.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

            best_val = float("inf")
            best_path: Optional[str] = None
            loss_history: dict[str, list[float]] = {sid: [] for sid in dataset.splat_ids}

            os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
            project_ckpt_dir = os.path.join(_CHECKPOINT_DIR, self._project.id)
            os.makedirs(project_ckpt_dir, exist_ok=True)

            for epoch in range(cfg.epochs):
                if self._stop_flag.is_set():
                    break
                self._pause_flag.wait()

                # Train
                model.train()
                train_loss = 0.0
                for audio, targets in train_loader:
                    if self._stop_flag.is_set():
                        break
                    audio = audio.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad()
                    preds = model(audio)
                    loss = _nan_mse_loss(preds, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= max(1, len(train_loader))

                model.eval()
                val_loss = 0.0
                per_splat_loss = [0.0] * dataset.num_splats
                with torch.no_grad():
                    for audio, targets in val_loader:
                        audio = audio.to(device)
                        targets = targets.to(device)
                        preds = model(audio)
                        val_loss += _nan_mse_loss(preds, targets).item()
                        for i in range(dataset.num_splats):
                            mask = ~torch.isnan(targets[:, i])
                            if mask.sum() > 0:
                                per_splat_loss[i] += nn.functional.mse_loss(
                                    preds[:, i][mask], targets[:, i][mask]
                                ).item()
                val_loss /= max(1, len(val_loader))

                for i, sid in enumerate(dataset.splat_ids):
                    per_splat_loss[i] /= max(1, len(val_loader))
                    loss_history[sid].append(per_splat_loss[i])

                scheduler.step()

                ckpt_path = os.path.join(project_ckpt_dir, f"epoch_{epoch:04d}.pt")
                torch.save(model.state_dict(), ckpt_path)

                if val_loss < best_val:
                    best_val = val_loss
                    best_path = os.path.join(project_ckpt_dir, "best.pt")
                    torch.save(model.state_dict(), best_path)

                if self.on_epoch_end:
                    self.on_epoch_end(epoch, train_loss, val_loss, per_splat_loss)

            if best_path and os.path.isfile(best_path):
                checkpoint = ModelCheckpoint(
                    path=best_path,
                    epoch=cfg.epochs - 1,
                    loss=best_val,
                    architecture=cfg.architecture,
                    loss_history=loss_history,
                )
                if self.on_finished:
                    self.on_finished(checkpoint)

        except Exception as e:
            if self.on_error:
                self.on_error(str(e))


# ---------------------------------------------------------------------------
# Active learning micro-update
# ---------------------------------------------------------------------------

def micro_update(
    model: nn.Module,
    audio_tensors: list[torch.Tensor],
    target_tensors: list[torch.Tensor],
    lr: float = 5e-5,
    steps: int = 5,
) -> None:
    """Fine-tune a model on a small batch of newly accepted/corrected labels."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        for audio, target in zip(audio_tensors, target_tensors):
            audio = audio.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            optimizer.zero_grad()
            pred = model(audio)
            loss = _nan_mse_loss(pred, target)
            loss.backward()
            optimizer.step()
    model.eval()
