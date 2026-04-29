"""PyTorch dataset for labeled chunks."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from songsplat.audio.loader import get_chunk_audio, TARGET_SR
from songsplat.core.models import Project, Splat


class ChunkDataset(Dataset):
    """Dataset of (audio_segment, {splat_id: value}) pairs from a project."""

    def __init__(
        self,
        project: Project,
        target_sr: int = TARGET_SR,
        chunk_samples: int = 44100,  # pad/truncate to this length (~2s at 22050)
    ) -> None:
        self._project = project
        self._target_sr = target_sr
        self._chunk_samples = chunk_samples
        self._splat_ids: list[str] = [s.id for s in project.sorted_splats()]
        self._items: list[tuple] = []  # (song, chunk, labels_dict)
        self._build()

    def _build(self) -> None:
        self._items = []
        for song in self._project.songs:
            for chunk in song.chunks:
                if chunk.is_labeled():
                    self._items.append((song, chunk, chunk.labels))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        song, chunk, labels = self._items[idx]
        audio = get_chunk_audio(song, chunk, target_sr=self._target_sr)

        # Pad or truncate to fixed length
        if len(audio) < self._chunk_samples:
            audio = np.pad(audio, (0, self._chunk_samples - len(audio)))
        else:
            audio = audio[: self._chunk_samples]

        audio_tensor = torch.from_numpy(audio).float()

        # Build target vector: one value per splat (NaN if unlabeled)
        targets = []
        for splat_id in self._splat_ids:
            splat = self._project.splat_by_id(splat_id)
            val = labels.get(splat_id)
            if val is not None and splat is not None:
                targets.append(splat.normalize(val))  # normalize to [0,1]
            else:
                targets.append(float("nan"))

        target_tensor = torch.tensor(targets, dtype=torch.float32)
        return audio_tensor, target_tensor

    @property
    def splat_ids(self) -> list[str]:
        return list(self._splat_ids)

    @property
    def num_splats(self) -> int:
        return len(self._splat_ids)
