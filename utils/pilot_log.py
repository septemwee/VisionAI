"""Session log file that mirrors console output for pilot diagnostics.

``start_session_log`` tees ``sys.stdout``/``sys.stderr`` into a timestamped
log file under ``runtime/logs`` while keeping the original console behaviour,
so pilot runs leave a diagnosable trail without changing what the operator
sees.

All writes are drained by a background writer thread with a bounded queue:
console sinks can stall indefinitely (a terminal with a text selection or a
paused pipe traps blocking writes), and a stalled sink must never block the
inspection pipeline or the GUI thread. When the queue overflows, lines are
dropped instead of freezing the application.
"""

import atexit
import queue
import sys
import threading
import time
from pathlib import Path

from utils.paths import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "runtime" / "logs"
FALLBACK_LOG_DIR = Path.home() / ".visionai" / "logs"
MAX_SESSION_LOGS = 20
MAX_PENDING_WRITES = 500
# A factory session runs for days; without a per-file cap the active log
# grows unbounded (per-tick prints with a flush per line).
MAX_LOG_BYTES = 20 * 1024 * 1024


class _LogWriter(threading.Thread):
    """Queued writer for console + session log; never blocks producers."""

    def __init__(self, original, log_file):
        super().__init__(daemon=True, name="pilot-log-writer")
        self.original = original
        self.log_file = log_file
        self.log_path = Path(log_file.name)
        self.rotation_count = 0
        self.queue = queue.Queue(maxsize=MAX_PENDING_WRITES)
        self.dropped = 0
        self._closing = threading.Event()
        atexit.register(self.close)
        self.start()

    # File-like API used by sys.stdout/sys.stderr consumers ---------------

    def write(self, text):
        self._enqueue(text)

    def flush(self):
        self._enqueue("")

    def isatty(self):
        return False

    def fileno(self):
        if self.original is not None and hasattr(self.original, "fileno"):
            return self.original.fileno()
        raise OSError("fileno is not available for this stream")

    # Internals -----------------------------------------------------------

    def _enqueue(self, text):
        try:
            self.queue.put_nowait(text)
        except queue.Full:
            self.dropped += 1

    def close(self):
        if self._closing.is_set():
            return
        self._closing.set()
        try:
            self.queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        self.join(timeout=2.0)

    def run(self):
        while True:
            text = self.queue.get()
            if text is None:
                break

            if text:
                self._write_original(text)
                self._write_file(text)
            self._report_dropped()

        if self.log_file is not None:
            try:
                self.log_file.close()
            except OSError:
                pass

    def _write_original(self, text):
        if self.original is None:
            return
        try:
            self.original.write(text)
        except (OSError, ValueError):
            self.original = None

    def _write_file(self, text):
        try:
            if self.log_file.tell() >= MAX_LOG_BYTES:
                self._rotate()

            if self.log_file is None:
                return

            self.log_file.write(text)
            self.log_file.flush()
        except (OSError, ValueError):
            self.log_file = None

    def _rotate(self):
        """Start a continuation log once the active file hits the size cap."""
        try:
            self.log_file.close()
        except (OSError, ValueError):
            pass

        self.rotation_count += 1
        rotated_path = self.log_path.with_name(
            f"{self.log_path.stem}-{self.rotation_count}{self.log_path.suffix}"
        )
        try:
            self.log_file = open(rotated_path, "a", encoding="utf-8", buffering=1)
            self.log_path = rotated_path
        except OSError:
            # Rotation failed (read-only dir, disk full): stop file writes
            # rather than blocking producers; console output keeps flowing.
            self.log_file = None

    def _report_dropped(self):
        if not self.dropped:
            return
        dropped, self.dropped = self.dropped, 0
        try:
            self.log_file.write(f"[LOG] dropped {dropped} lines (slow sink)\n")
            self.log_file.flush()
        except (OSError, ValueError):
            pass


class Tee:
    """File-like object that forwards writes to the background writer."""

    def __init__(self, original, writer):
        self.original = original
        self.writer = writer

    def write(self, text):
        self.writer.write(text)

    def flush(self):
        self.writer.flush()

    def isatty(self):
        return self.writer.isatty()

    def fileno(self):
        return self.writer.fileno()


def _prune_session_logs(log_dir):
    """Delete the oldest session logs, keeping at most MAX_SESSION_LOGS."""
    try:
        session_logs = sorted(
            log_dir.glob("pilot-*.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for old_log in session_logs[MAX_SESSION_LOGS:]:
            old_log.unlink(missing_ok=True)
    except OSError:
        pass


def _open_session_file(log_dir):
    """Create the log directory and open a new timestamped log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pilot-{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    _prune_session_logs(log_dir)
    return log_file, log_path


def start_session_log():
    """Tee console output into a timestamped log file and report its path.

    Falls back to a log directory under the user profile when the project
    directory is read-only. If neither location is writable, logging is
    silently disabled and ``None`` is returned; no exception escapes.
    """
    try:
        try:
            log_file, log_path = _open_session_file(LOG_DIR)
        except Exception:
            log_file, log_path = _open_session_file(FALLBACK_LOG_DIR)
    except Exception:
        return None

    writer = _LogWriter(sys.stdout, log_file)

    sys.stdout = Tee(sys.stdout, writer)
    sys.stderr = Tee(sys.stderr, writer)

    print(f"Session log: {log_path}")
    return writer
