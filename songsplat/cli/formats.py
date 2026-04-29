"""Serializable file formats used by the songsplat CLI.

File types
----------
.splatchunk
    Audio source path, sample rate, duration, and the list of time-range chunks.
.splatdata
    Chunk file reference, splat definitions, and the chunk->splat label map.
.splat
    Zip bundle containing ``meta.json`` (splat list + architecture info) and
    optionally ``model.pt`` (PyTorch weights).
"""

from __future__ import annotations

import json
import os
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkInfo:
    """Time range for a single chunk."""

    index: int
    start: float
    end:   float

    def to_dict(self) -> dict:
        return {"index": self.index, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkInfo":
        return cls(index=d["index"], start=d["start"], end=d["end"])


@dataclass
class SplatChunk:
    """Contents of a ``.splatchunk`` file."""

    id:              str        = field(default_factory=lambda: str(uuid.uuid4()))
    source_path:     str        = ""
    song_name:       str        = ""
    sample_rate:     int        = 22050
    duration:        float      = 0.0
    chunk_mode:      str        = "fixed"
    chunk_duration:  float      = 2.0
    beats_per_chunk: int        = 4
    chunks:          list[ChunkInfo] = field(default_factory=list)

    def save(self, path: str) -> None:
        """Write this object to *path* as JSON."""
        data = {
            "id": self.id, "source_path": self.source_path,
            "song_name": self.song_name, "sample_rate": self.sample_rate,
            "duration": self.duration, "chunk_mode": self.chunk_mode,
            "chunk_duration": self.chunk_duration,
            "beats_per_chunk": self.beats_per_chunk,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SplatChunk":
        """Load a ``.splatchunk`` file from *path*."""
        with open(path) as f:
            d = json.load(f)
        obj = cls(
            id=d.get("id", str(uuid.uuid4())),
            source_path=d["source_path"],
            song_name=d.get("song_name", ""),
            sample_rate=d.get("sample_rate", 22050),
            duration=d.get("duration", 0.0),
            chunk_mode=d.get("chunk_mode", "fixed"),
            chunk_duration=d.get("chunk_duration", 2.0),
            beats_per_chunk=d.get("beats_per_chunk", 4),
        )
        obj.chunks = [ChunkInfo.from_dict(c) for c in d.get("chunks", [])]
        return obj


@dataclass
class SplatDef:
    """Definition of one splat dimension stored inside a ``.splatdata`` file."""

    id:         str = field(default_factory=lambda: str(uuid.uuid4()))
    name:       str = "Unnamed"
    low_label:  str = ""
    high_label: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name,
                "low_label": self.low_label, "high_label": self.high_label}

    @classmethod
    def from_dict(cls, d: dict) -> "SplatDef":
        return cls(**d)


@dataclass
class SplatData:
    """Contents of a ``.splatdata`` file."""

    id:         str = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_file: str = ""
    splats:     list[SplatDef]              = field(default_factory=list)
    labels:     dict[str, dict[str, float]] = field(default_factory=dict)

    def set_label(self, chunk_index: int, splat_id: str, value: float) -> None:
        """Record *value* for *splat_id* on *chunk_index*."""
        self.labels.setdefault(str(chunk_index), {})[splat_id] = value

    def get_label(self, chunk_index: int, splat_id: str) -> Optional[float]:
        """Return the label for *splat_id* on *chunk_index*, or ``None``."""
        return self.labels.get(str(chunk_index), {}).get(splat_id)

    def save(self, path: str) -> None:
        """Write this object to *path* as JSON."""
        with open(path, "w") as f:
            json.dump({
                "id": self.id, "chunk_file": self.chunk_file,
                "splats": [s.to_dict() for s in self.splats],
                "labels": self.labels,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SplatData":
        """Load a ``.splatdata`` file from *path*."""
        with open(path) as f:
            d = json.load(f)
        obj = cls(
            id=d.get("id", str(uuid.uuid4())),
            chunk_file=d.get("chunk_file", ""),
        )
        obj.splats = [SplatDef.from_dict(s) for s in d.get("splats", [])]
        obj.labels = d.get("labels", {})
        return obj


@dataclass
class SplatFile:
    """In-memory representation of an extracted ``.splat`` bundle."""

    meta:       dict
    model_path: str

    @classmethod
    def load(cls, splat_path: str) -> "SplatFile":
        """Extract the bundle at *splat_path* and return a :class:`SplatFile`."""
        extract_dir = splat_path + "_extracted"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(splat_path) as zf:
            zf.extractall(extract_dir)
        with open(os.path.join(extract_dir, "meta.json")) as f:
            meta = json.load(f)
        return cls(meta=meta, model_path=os.path.join(extract_dir, "model.pt"))

    @property
    def splat_defs(self) -> list[dict]:
        """List of splat definition dicts from the bundle metadata."""
        return self.meta.get("splats", [])

    @property
    def architecture(self) -> str:
        return self.meta.get("architecture", "pretrained")
