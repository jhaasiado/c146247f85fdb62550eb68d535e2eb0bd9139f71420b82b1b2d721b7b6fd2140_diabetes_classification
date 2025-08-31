from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create a directory if it doesn't already exist."""
    path.mkdir(parents=True, exist_ok=True)
