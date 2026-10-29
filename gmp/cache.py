"""Small thread-safe TTL cache with single-flight loading.

FastAPI runs sync endpoints in a thread pool, so the old module-level dicts
could be read and mutated at the same time. This cache also makes sure two
requests for the same key don't both hit Yahoo at once.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

T = TypeVar("T")


class TTLCache:
    def __init__(self, ttl_seconds: float, max_entries: int = 256) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._data: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._key_locks: dict[Hashable, threading.Lock] = {}

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            stored_at, value = hit
            if time.monotonic() - stored_at > self.ttl:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def get_or_load(self, key: Hashable, loader: Callable[[], T]) -> T:
        value = self.get(key)
        if value is not None:
            return value
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            value = self.get(key)  # another thread may have loaded it
            if value is not None:
                return value
            value = loader()
            if value is not None:
                self.set(key, value)
            return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
