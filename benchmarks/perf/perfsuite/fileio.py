"""Writing results so that a reader never sees half a file."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path


def write_atomic(path, data, *, retry_s=5.0) -> Path:
    """Write `data` (str or bytes) to `path` so a reader sees the old file or the new one, never a mix.

    The data goes to a temporary file of its own in the same directory (two writers do not share
    one), which then replaces `path`. On Windows the replace fails while another process has `path`
    open, so it is retried for `retry_s` seconds; the temporary file is removed if it never lands.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data if isinstance(data, bytes) else data.encode("utf-8")
    # a name of its own, created as open() creates files: mode 0666 less the process umask
    tmp = str(path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp"))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        if path.exists():  # a replaced file keeps its mode
            os.chmod(tmp, path.stat().st_mode & 0o777)
        deadline = time.monotonic() + retry_s
        while True:
            try:
                os.replace(tmp, path)
                return path
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
