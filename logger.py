"""
Detection event logger.
Daily-rotating JSONL files + in-memory ring buffer for dashboard.
"""

import json
import os
import glob
import time
import threading
from collections import deque
from config import LOG_DIR


class DetectionLogger:
    def __init__(self, max_mb: int = 50, keep_days: int = 7, maxlen: int = 500):
        self.log_dir   = LOG_DIR
        self.max_bytes = max_mb * 1024 * 1024
        self.keep_days = keep_days
        self._buf      = deque(maxlen=maxlen)
        self._lock     = threading.Lock()
        self._subs: list[deque] = []
        self._fh       = None
        self._cur_path = None
        os.makedirs(self.log_dir, exist_ok=True)
        self._open_file()
        self._cleanup_old()

    def _today_base(self) -> str:
        return os.path.join(self.log_dir, f"detections_{time.strftime('%Y-%m-%d')}")

    def _open_file(self):
        if self._fh:
            self._fh.close()
        base = self._today_base()
        path = base + ".jsonl"
        n = 1
        while os.path.exists(path) and os.path.getsize(path) >= self.max_bytes:
            path = f"{base}_{n}.jsonl"; n += 1
        self._cur_path = path
        self._fh = open(path, "a", buffering=1)

    def _rotate_if_needed(self):
        if (not self._cur_path.startswith(self._today_base())
                or os.path.getsize(self._cur_path) >= self.max_bytes):
            self._open_file()

    def _cleanup_old(self):
        cutoff = time.time() - self.keep_days * 86400
        for f in glob.glob(os.path.join(self.log_dir, "detections_*.jsonl")):
            if os.path.getmtime(f) < cutoff:
                os.remove(f)

    @property
    def log_file(self) -> str:
        return self._cur_path

    def log(self, event: dict):
        with self._lock:
            self._rotate_if_needed()
            self._fh.write(json.dumps(event) + "\n")
            self._buf.append(event)
            for q in self._subs:
                q.append(event)

    def recent(self, n: int = 100) -> list:
        with self._lock:
            return list(self._buf)[-n:]

    def subscribe(self) -> deque:
        q = deque(maxlen=200)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: deque):
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass
