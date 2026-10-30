"""Keep one dictation process per user with a process-lifetime OS lock.

The lock file is deliberately left in place. Removing it would allow another
process to lock a new file while the original file is still locked. The OS
releases the actual lock even if the owner crashes or is forcibly stopped.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path
from typing import BinaryIO


class SingleInstance:
    """Hold ``lock_path`` until ``release()`` or process exit.

    ``acquire()`` returns False when another instance owns the lock. Other
    filesystem errors propagate so an unavailable lock cannot silently allow
    multiple dictation processes to start.
    """

    def __init__(self, lock_path: str | os.PathLike[str]):
        self.path = Path(lock_path)
        self._file: BinaryIO | None = None

    def acquire(self) -> bool:
        if self._file is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.path.open("a+b", buffering=0)
        try:
            # msvcrt locks a byte range, including the first byte in this file.
            # Never truncate an existing file, which another process may own.
            if lock_file.seek(0, os.SEEK_END) == 0:
                lock_file.write(b"\0")
            lock_file.seek(0)
            try:
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    return False
                raise
            self._file = lock_file
            return True
        finally:
            if self._file is not lock_file:
                lock_file.close()

    def release(self) -> None:
        lock_file, self._file = self._file, None
        if lock_file is None:
            return
        try:
            lock_file.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
