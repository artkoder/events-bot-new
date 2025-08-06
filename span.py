import logging
import time
from contextlib import asynccontextmanager


class _Span:
    def __init__(self) -> None:
        self._thresholds: dict[str, float] = {}

    def configure(self, thresholds: dict[str, int | float]) -> None:
        self._thresholds.update(thresholds)

    @asynccontextmanager
    async def __call__(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - start) * 1000
            if dt >= self._thresholds.get(label, 1000):
                logging.debug("%s took %.0f ms", label, dt)


span = _Span()
