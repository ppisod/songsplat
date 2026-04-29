"""Core data models for Songsplat."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Splat:
    """A user-defined perceptual dimension (e.g. 'bounciness', 'darkness')."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Unnamed Splat"
    min_val: float = 0.0
    max_val: float = 1.0
    low_label: str = ""
    high_label: str = ""
    color: str = "#111111"
    order: int = 0

    def clamp(self, value: float) -> float:
        return max(self.min_val, min(self.max_val, value))

    def normalize(self, value: float) -> float:
        """Map value to [0, 1] within the splat's range."""
        span = self.max_val - self.min_val
        if span == 0:
            return 0.0
        return (value - self.min_val) / span

    def denormalize(self, t: float) -> float:
        """Map t in [0, 1] back to the splat's range."""
        return self.min_val + t * (self.max_val - self.min_val)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "low_label": self.low_label,
            "high_label": self.high_label,
            "color": self.color,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Splat":
        return cls(**d)


@dataclass
class Chunk:
    """A time-bounded slice of a song that can be labeled with splat values."""

    index: int
    start: float
    end: float
    labels: dict[str, float] = field(default_factory=dict)
    predictions: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0

    def is_labeled(self, splat_id: Optional[str] = None) -> bool:
        if splat_id is not None:
            return splat_id in self.labels
        return len(self.labels) > 0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "labels": self.labels,
            "predictions": self.predictions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            index=d["index"],
            start=d["start"],
            end=d["end"],
            labels=d.get("labels", {}),
            predictions=d.get("predictions", {}),
        )


@dataclass
class CurvePoint:
    """A single control point on a splat curve."""

    time: float
    value: float

    def to_dict(self) -> dict:
        return {"time": self.time, "value": self.value}

    @classmethod
    def from_dict(cls, d: dict) -> "CurvePoint":
        return cls(time=d["time"], value=d["value"])


@dataclass
class SplatCurve:
    """A freehand curve drawn over a song's timeline for one splat."""

    splat_id: str
    points: list[CurvePoint] = field(default_factory=list)

    def sorted_points(self) -> list[CurvePoint]:
        return sorted(self.points, key=lambda p: p.time)

    def sample_at(self, time: float) -> Optional[float]:
        """Linear interpolation of the curve at the given time."""
        pts = self.sorted_points()
        if not pts:
            return None
        if time <= pts[0].time:
            return pts[0].value
        if time >= pts[-1].time:
            return pts[-1].value
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if a.time <= time <= b.time:
                t = (time - a.time) / (b.time - a.time) if b.time != a.time else 0
                return a.value + t * (b.value - a.value)
        return None

    def to_dict(self) -> dict:
        return {
            "splat_id": self.splat_id,
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SplatCurve":
        return cls(
            splat_id=d["splat_id"],
            points=[CurvePoint.from_dict(p) for p in d.get("points", [])],
        )


@dataclass
class Song:
    """An imported audio file with its chunks and curves."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    path: str = ""
    name: str = ""
    sample_rate: int = 44100
    duration: float = 0.0
    num_channels: int = 1
    chunk_mode: str = "fixed"
    chunk_duration: float = 2.0
    chunks: list[Chunk] = field(default_factory=list)
    curves: dict[str, SplatCurve] = field(default_factory=dict)
    waveform_cache: Optional[list[float]] = field(default=None, repr=False)

    def labeled_chunks(self, splat_id: Optional[str] = None) -> list[Chunk]:
        return [c for c in self.chunks if c.is_labeled(splat_id)]

    def unlabeled_chunks(self, splat_id: Optional[str] = None) -> list[Chunk]:
        return [c for c in self.chunks if not c.is_labeled(splat_id)]

    def apply_curve_to_chunks(self, splat_id: str) -> None:
        """Sample a drawn curve at each chunk center and write into chunk labels."""
        curve = self.curves.get(splat_id)
        if curve is None:
            return
        for chunk in self.chunks:
            val = curve.sample_at(chunk.center)
            if val is not None:
                chunk.labels[splat_id] = val

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "name": self.name,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "num_channels": self.num_channels,
            "chunk_mode": self.chunk_mode,
            "chunk_duration": self.chunk_duration,
            "chunks": [c.to_dict() for c in self.chunks],
            "curves": {k: v.to_dict() for k, v in self.curves.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Song":
        song = cls(
            id=d["id"],
            path=d["path"],
            name=d.get("name", ""),
            sample_rate=d.get("sample_rate", 44100),
            duration=d.get("duration", 0.0),
            num_channels=d.get("num_channels", 1),
            chunk_mode=d.get("chunk_mode", "fixed"),
            chunk_duration=d.get("chunk_duration", 2.0),
        )
        song.chunks = [Chunk.from_dict(c) for c in d.get("chunks", [])]
        song.curves = {
            k: SplatCurve.from_dict(v)
            for k, v in d.get("curves", {}).items()
        }
        return song


@dataclass
class ModelCheckpoint:
    """Reference to a saved model checkpoint on disk."""

    path: str
    epoch: int = 0
    loss: float = float("inf")
    architecture: str = "pretrained"
    loss_history: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "epoch": self.epoch,
            "loss": self.loss,
            "architecture": self.architecture,
            "loss_history": self.loss_history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelCheckpoint":
        return cls(
            path=d["path"],
            epoch=d.get("epoch", 0),
            loss=d.get("loss", float("inf")),
            architecture=d.get("architecture", "pretrained"),
            loss_history=d.get("loss_history", {}),
        )


@dataclass
class FeedbackRecord:
    """One accept/override/nudge action from the active learning loop."""

    song_id: str
    chunk_index: int
    splat_id: str
    action: str
    original_pred: float
    final_value: float

    def to_dict(self) -> dict:
        return {
            "song_id": self.song_id,
            "chunk_index": self.chunk_index,
            "splat_id": self.splat_id,
            "action": self.action,
            "original_pred": self.original_pred,
            "final_value": self.final_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackRecord":
        return cls(**d)


@dataclass
class Project:
    """Top-level container for a Songsplat project."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Project"
    version: str = "0.1.0"
    splats: list[Splat] = field(default_factory=list)
    songs: list[Song] = field(default_factory=list)
    best_checkpoint: Optional[ModelCheckpoint] = None
    checkpoints: list[ModelCheckpoint] = field(default_factory=list)
    feedback_history: list[FeedbackRecord] = field(default_factory=list)
    drift_history: dict[str, list[float]] = field(default_factory=dict)
    nudge_delta: float = 0.05

    def splat_by_id(self, splat_id: str) -> Optional[Splat]:
        for s in self.splats:
            if s.id == splat_id:
                return s
        return None

    def song_by_id(self, song_id: str) -> Optional[Song]:
        for s in self.songs:
            if s.id == song_id:
                return s
        return None

    def sorted_splats(self) -> list[Splat]:
        return sorted(self.splats, key=lambda s: s.order)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "splats": [s.to_dict() for s in self.splats],
            "songs": [s.to_dict() for s in self.songs],
            "best_checkpoint": self.best_checkpoint.to_dict() if self.best_checkpoint else None,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "feedback_history": [f.to_dict() for f in self.feedback_history],
            "drift_history": self.drift_history,
            "nudge_delta": self.nudge_delta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        proj = cls(
            id=d.get("id", str(uuid.uuid4())),
            name=d.get("name", "Untitled Project"),
            version=d.get("version", "0.1.0"),
            nudge_delta=d.get("nudge_delta", 0.05),
        )
        proj.splats = [Splat.from_dict(s) for s in d.get("splats", [])]
        proj.songs = [Song.from_dict(s) for s in d.get("songs", [])]
        if d.get("best_checkpoint"):
            proj.best_checkpoint = ModelCheckpoint.from_dict(d["best_checkpoint"])
        proj.checkpoints = [ModelCheckpoint.from_dict(c) for c in d.get("checkpoints", [])]
        proj.feedback_history = [FeedbackRecord.from_dict(f) for f in d.get("feedback_history", [])]
        proj.drift_history = d.get("drift_history", {})
        return proj
