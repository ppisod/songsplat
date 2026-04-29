"""Save and load .splatject files (zip archives containing JSON + assets)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from songsplat.core.models import Project

PROJECT_EXTENSION = ".splatject"
MANIFEST_FILENAME = "project.json"
FORMAT_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Recent projects registry (stored in ~/.songsplat/recents.json)
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path.home() / ".songsplat"
_RECENTS_FILE = _CONFIG_DIR / "recents.json"
MAX_RECENTS = 10


def _ensure_config_dir() -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def get_recent_projects() -> list[str]:
    """Return list of recent project file paths (most recent first)."""
    _ensure_config_dir()
    if not _RECENTS_FILE.exists():
        return []
    try:
        data = json.loads(_RECENTS_FILE.read_text(encoding="utf-8"))
        return [p for p in data.get("recents", []) if os.path.exists(p)]
    except (json.JSONDecodeError, OSError):
        return []


def _push_recent(path: str) -> None:
    _ensure_config_dir()
    recents = get_recent_projects()
    recents = [p for p in recents if p != path]
    recents.insert(0, path)
    recents = recents[:MAX_RECENTS]
    _RECENTS_FILE.write_text(
        json.dumps({"recents": recents}, indent=2), encoding="utf-8"
    )


def remove_recent(path: str) -> None:
    _ensure_config_dir()
    recents = get_recent_projects()
    recents = [p for p in recents if p != path]
    _RECENTS_FILE.write_text(
        json.dumps({"recents": recents}, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_project(project: Project, path: str) -> None:
    """Serialize project to a .songsplatproj zip file at *path*.

    Any model checkpoint files referenced by the project are copied into the
    archive under a `checkpoints/` subdirectory, and the stored paths are
    updated to relative form so the project is portable.
    """
    path = str(path)
    if not path.endswith(PROJECT_EXTENSION):
        path += PROJECT_EXTENSION

    tmp_dir = tempfile.mkdtemp()
    try:
        project_dict = project.to_dict()
        project_dict["format_version"] = FORMAT_VERSION

        checkpoint_relpaths: dict[str, str] = {}
        for i, cp in enumerate(project_dict.get("checkpoints", [])):
            abs_path = cp.get("path", "")
            if abs_path and os.path.isfile(abs_path):
                rel = f"checkpoints/checkpoint_{i}_{os.path.basename(abs_path)}"
                checkpoint_relpaths[abs_path] = rel
                cp["path"] = rel  # store relative inside archive

        if project_dict.get("best_checkpoint"):
            abs_path = project_dict["best_checkpoint"].get("path", "")
            if abs_path and os.path.isfile(abs_path):
                if abs_path not in checkpoint_relpaths:
                    rel = f"checkpoints/best_{os.path.basename(abs_path)}"
                    checkpoint_relpaths[abs_path] = rel
                project_dict["best_checkpoint"]["path"] = checkpoint_relpaths[abs_path]

        manifest_path = os.path.join(tmp_dir, MANIFEST_FILENAME)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(project_dict, f, indent=2, ensure_ascii=False)

        tmp_zip = os.path.join(tmp_dir, "out.zip")
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, MANIFEST_FILENAME)
            for abs_path, rel in checkpoint_relpaths.items():
                zf.write(abs_path, rel)

        shutil.move(tmp_zip, path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _push_recent(os.path.abspath(path))


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_project(path: str, extract_dir: Optional[str] = None) -> Project:
    """Deserialize a .songsplatproj file.

    Checkpoint weights inside the archive are extracted to *extract_dir*
    (defaults to ~/.songsplat/checkpoints/<project_id>/).
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Project file not found: {path}")

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if MANIFEST_FILENAME not in names:
            raise ValueError(f"Invalid project file: missing {MANIFEST_FILENAME}")

        manifest_bytes = zf.read(MANIFEST_FILENAME)
        project_dict = json.loads(manifest_bytes.decode("utf-8"))

        _check_format_version(project_dict.get("format_version", "0.0.0"))

        checkpoint_names = [n for n in names if n.startswith("checkpoints/")]
        if checkpoint_names:
            project_id = project_dict.get("id", "unknown")
            if extract_dir is None:
                extract_dir = str(_CONFIG_DIR / "checkpoints" / project_id)
            os.makedirs(extract_dir, exist_ok=True)
            for name in checkpoint_names:
                zf.extract(name, extract_dir)

            def _abs(rel: str) -> str:
                if rel and not os.path.isabs(rel):
                    return os.path.join(extract_dir, rel)
                return rel

            for cp in project_dict.get("checkpoints", []):
                cp["path"] = _abs(cp.get("path", ""))
            if project_dict.get("best_checkpoint"):
                project_dict["best_checkpoint"]["path"] = _abs(
                    project_dict["best_checkpoint"].get("path", "")
                )

    project = Project.from_dict(project_dict)
    _push_recent(path)
    return project


def _check_format_version(stored: str) -> None:
    """Warn (don't crash) on version mismatch."""
    stored_parts = tuple(int(x) for x in stored.split(".") if x.isdigit())
    current_parts = tuple(int(x) for x in FORMAT_VERSION.split(".") if x.isdigit())
    if stored_parts > current_parts:
        import warnings
        warnings.warn(
            f"Project was saved with a newer format version ({stored}). "
            f"Some features may not load correctly.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# New project helper
# ---------------------------------------------------------------------------

def new_project(name: str = "Untitled Project") -> Project:
    return Project(name=name)
