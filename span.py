import logging
import time
from contextlib import asynccontextmanager

import psutil


class _Span:
    def __init__(self) -> None:
        self._thresholds: dict[str, float] = {}
        self._process = psutil.Process()

    def configure(self, thresholds: dict[str, int | float]) -> None:
        self._thresholds.update(thresholds)

    @asynccontextmanager
    async def __call__(self, label: str, **fields):
        start = time.perf_counter()
        start_rss = self._process.memory_info().rss // 2**20
        try:
            yield
        finally:
            dt = (time.perf_counter() - start) * 1000
            end_rss = self._process.memory_info().rss // 2**20
            if dt >= self._thresholds.get(label, 1000):
                extra = " ".join(f"{k}={v}" for k, v in fields.items())
                if extra:
                    extra = " " + extra
                logging.debug(
                    "%s took %.0f ms rss=%d MB Δ%+d MB%s",
                    label,
                    dt,
                    end_rss,
                    end_rss - start_rss,
                    extra,
                )


span = _Span()
