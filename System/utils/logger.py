"""Centralized logging for AI Camera System.

Directory structure:
  logs/
  ├── system/
  │   ├── 20260526.log   <- today's active log
  │   ├── 20260525.log   <- yesterday (kept up to LOG_BACKUP_COUNT files)
  │   └── ...
  └── cam_{id}/
      ├── 20260526.log
      ├── 20260525.log
      └── ...

Rotation: automatic at midnight on first write of the new day.
Cleanup: keeps at most LOG_BACKUP_COUNT daily files per folder.
"""

import glob
import logging
import os
import threading
import time
import datetime
from typing import Optional

_lock = threading.Lock()
_system_logger: Optional[logging.Logger] = None
_cam_loggers: dict = {}
_base_log_dir: str = ""

LOG_BACKUP_COUNT = 10

_throttle_lock = threading.Lock()
_throttle_state: dict = {}   # key -> {"window_start": float, "count": int, "last_message": str}

_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)-5s] [%(name)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class _DailyFileHandler(logging.Handler):
    """Write to YYYYMMDD.log inside log_dir.

    On the first write after midnight the file rolls over automatically.
    Old files exceeding backup_count are deleted.
    """

    def __init__(self, log_dir: str, backup_count: int = LOG_BACKUP_COUNT):
        super().__init__()
        self._log_dir = log_dir
        self._backup_count = backup_count
        self._handler_lock = threading.Lock()
        self._current_date: str = ""
        self._stream = None
        os.makedirs(log_dir, exist_ok=True)
        self._open_stream()

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _today() -> str:
        return datetime.date.today().strftime("%Y%m%d")

    def _log_path(self, date_str: str) -> str:
        return os.path.join(self._log_dir, f"{date_str}.log")

    def _open_stream(self) -> None:
        today = self._today()
        self._current_date = today
        self._stream = open(self._log_path(today), "a", encoding="utf-8")

    def _rollover(self) -> None:
        """Close current stream, open new date file, prune old files."""
        if self._stream:
            self._stream.flush()
            self._stream.close()
        self._open_stream()
        self._prune_old_files()

    def _prune_old_files(self) -> None:
        """Delete oldest files when count exceeds backup_count."""
        pattern = os.path.join(self._log_dir, "????????.log")
        files = sorted(glob.glob(pattern))          # sorted = oldest first
        while len(files) > self._backup_count:
            try:
                os.remove(files.pop(0))
            except OSError:
                break

    # ── logging.Handler interface ─────────────────────────────────────────

    def emit(self, record: logging.LogRecord) -> None:
        with self._handler_lock:
            # Roll over if the date changed
            if self._today() != self._current_date:
                self._rollover()
            try:
                msg = self.format(record) + "\n"
                self._stream.write(msg)
                self._stream.flush()
            except Exception:
                self.handleError(record)

    def close(self) -> None:
        with self._handler_lock:
            if self._stream:
                try:
                    self._stream.flush()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        super().close()


# ── Public API ────────────────────────────────────────────────────────────────

def init_logging(base_log_dir: str) -> None:
    """Initialize logging. Call once from main() before starting workers."""
    global _base_log_dir, _system_logger
    _base_log_dir = base_log_dir

    sys_dir = os.path.join(base_log_dir, "system")
    os.makedirs(sys_dir, exist_ok=True)

    logger = logging.getLogger("system")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        h = _DailyFileHandler(sys_dir)
        h.setFormatter(_FORMATTER)
        logger.addHandler(h)

    _system_logger = logger


def get_system_logger() -> logging.Logger:
    """Return the system-wide logger (startup, GPU, DB, API sync events)."""
    if _system_logger is None:
        raise RuntimeError("Call init_logging() before using loggers.")
    return _system_logger


def get_cam_logger(cam_id: int) -> logging.Logger:
    """Return a per-camera logger. Creates folder + handler on first call. Thread-safe."""
    if cam_id not in _cam_loggers:
        with _lock:
            if cam_id not in _cam_loggers:
                if not _base_log_dir:
                    raise RuntimeError("Call init_logging() before using loggers.")
                cam_dir = os.path.join(_base_log_dir, f"cam_{cam_id}")
                os.makedirs(cam_dir, exist_ok=True)

                logger = logging.getLogger(f"cam_{cam_id}")
                logger.setLevel(logging.DEBUG)
                logger.propagate = False

                h = _DailyFileHandler(cam_dir)
                h.setFormatter(_FORMATTER)
                logger.addHandler(h)
                _cam_loggers[cam_id] = logger

    return _cam_loggers[cam_id]


def remove_cam_logger(cam_id: int) -> None:
    """Close and remove a camera logger when camera is removed from the system."""
    with _lock:
        logger = _cam_loggers.pop(cam_id, None)
        if logger:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


def log_throttled(
    logger: logging.Logger,
    level: int,
    key: str,
    message: str,
    *,
    interval: float = 60.0,
) -> None:
    """Log `message` at `level`, collapsing repeats sharing `key` into at most
    one line per `interval` seconds, plus a summary line when the window
    reopens ("... repeated N more time(s) in the last Ms").

    `key` MUST be a stable identifier for the call-site/error class (e.g.
    "yolo_thread_unexpected_error") — NOT the interpolated message itself,
    since messages containing exception text differ on every call and would
    defeat deduplication if used as the key. Thread-safe (single short-held
    lock; safe to call from any worker thread).
    """
    now = time.monotonic()
    with _throttle_lock:
        st = _throttle_state.get(key)
        if st is None or (now - st["window_start"]) >= interval:
            if st is not None and st["count"] > 0:
                logger.log(
                    level,
                    f"(throttled) previous message repeated {st['count']} more "
                    f"time(s) in the last {now - st['window_start']:.0f}s: {st['last_message']}",
                )
            _throttle_state[key] = {"window_start": now, "count": 0, "last_message": message}
            logger.log(level, message)
        else:
            st["count"] += 1
            st["last_message"] = message
