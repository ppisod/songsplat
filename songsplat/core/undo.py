"""Simple undo/redo stack for reversible user actions."""

from __future__ import annotations

from typing import Callable, Optional


class Action:
    """An undoable/redoable operation."""

    def __init__(self, undo_fn: Callable, redo_fn: Callable, description: str = "") -> None:
        self._undo_fn = undo_fn
        self._redo_fn = redo_fn
        self.description = description

    def undo(self) -> None:
        self._undo_fn()

    def redo(self) -> None:
        self._redo_fn()


class UndoStack:
    """Maintains an undo/redo history of Actions.

    Max size prevents unbounded memory growth.
    """

    def __init__(self, max_size: int = 200) -> None:
        self._stack: list[Action] = []
        self._index: int = -1  # points to the last executed action
        self._max_size = max_size
        self.on_change: Optional[Callable] = None

    def push(self, action: Action) -> None:
        """Record a new action. Clears any redo history beyond current position."""
        self._stack = self._stack[: self._index + 1]
        self._stack.append(action)
        if len(self._stack) > self._max_size:
            self._stack = self._stack[-self._max_size:]
        self._index = len(self._stack) - 1
        if self.on_change:
            self.on_change()

    def undo(self) -> bool:
        """Undo the most recent action. Returns True if successful."""
        if self._index < 0:
            return False
        self._stack[self._index].undo()
        self._index -= 1
        if self.on_change:
            self.on_change()
        return True

    def redo(self) -> bool:
        """Redo the next action. Returns True if successful."""
        if self._index >= len(self._stack) - 1:
            return False
        self._index += 1
        self._stack[self._index].redo()
        if self.on_change:
            self.on_change()
        return True

    def can_undo(self) -> bool:
        return self._index >= 0

    def can_redo(self) -> bool:
        return self._index < len(self._stack) - 1

    def clear(self) -> None:
        self._stack.clear()
        self._index = -1
        if self.on_change:
            self.on_change()


# ---------------------------------------------------------------------------
# Factory helpers for common actions
# ---------------------------------------------------------------------------

def set_chunk_label_action(chunk, splat_id: str, new_value: float) -> Action:
    """Create an undoable action for setting a splat label on a chunk."""
    old_value = chunk.labels.get(splat_id)

    def _undo():
        if old_value is None:
            chunk.labels.pop(splat_id, None)
        else:
            chunk.labels[splat_id] = old_value

    def _redo():
        chunk.labels[splat_id] = new_value

    return Action(_undo, _redo, f"label {splat_id}={new_value:.3f}")


def delete_chunk_label_action(chunk, splat_id: str) -> Action:
    """Create an undoable action for removing a label from a chunk."""
    old_value = chunk.labels.get(splat_id)

    def _undo():
        if old_value is not None:
            chunk.labels[splat_id] = old_value

    def _redo():
        chunk.labels.pop(splat_id, None)

    return Action(_undo, _redo, f"delete label {splat_id}")


def add_splat_action(project, splat) -> Action:
    def _undo():
        project.splats = [s for s in project.splats if s.id != splat.id]

    def _redo():
        project.splats.append(splat)

    return Action(_undo, _redo, f"add splat {splat.name}")


def delete_splat_action(project, splat) -> Action:
    old_splats = list(project.splats)

    def _undo():
        project.splats = list(old_splats)

    def _redo():
        project.splats = [s for s in project.splats if s.id != splat.id]

    return Action(_undo, _redo, f"delete splat {splat.name}")
