"""Crash-safe checkpoint writes shared by every agent's save()."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    """Write `obj` via torch.save() without ever exposing a partial file at `path`.

    torch.save(obj, path) writes straight to the final path; a reader that
    opens it while the write is still in progress (e.g. this same checkpoint
    being reloaded moments after a training loop saved it) sees a truncated
    zip archive and fails with a low-level torch.load error, not anything
    that points back at "the file was mid-write". Saving to a temp file in
    the same directory first and only then atomically replacing the target
    (os.replace(), a single filesystem rename) means any reader either sees
    the complete previous file or the complete new one -- never a partial
    write, and same-directory keeps the rename on one filesystem so it's
    guaranteed atomic on POSIX.
    """
    atomic_save(path, lambda tmp_path: torch.save(obj, tmp_path))


def atomic_save(path: str | Path, writer: Callable[[str], None]) -> None:
    """Call `writer(tmp_path)` to populate a temp file, then atomically
    replace `path` with it. Shared by atomic_torch_save() and any saver
    (e.g. Stable Baselines3's own .save()) that writes to a path itself
    rather than returning bytes.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(fd)
    try:
        writer(tmp_name)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
