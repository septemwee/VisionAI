"""Bounded handoff between detection, inspection and the GUI."""
from threading import Lock


class LatestFrame:
    def __init__(self):
        self._lock = Lock()
        self._value = None
        self._version = 0

    def put(self, value):
        with self._lock:
            self._version += 1
            self._value = value
            return self._version

    def read(self):
        with self._lock:
            return self._version, self._value
