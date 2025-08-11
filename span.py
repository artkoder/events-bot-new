import logging
import time
from typing import Any


class _Span:
    def __init__(self) -> None:
        self._thresholds: dict[str, float] = {}
        self.extra: dict[str, Any] = {}
        self._stack: list[dict[str, Any]] = []

    def configure(self, thresholds: dict[str, int | float]) -> None:
        self._thresholds.update(thresholds)

    def __call__(self, label: str | None = None, **meta):
        if label is None and "task" in meta:
            label = str(meta.pop("task"))
        if label is None:
            raise TypeError("label is required")

        span = self

        class _Ctx:
            def __init__(self, label: str, meta: dict[str, Any]):
                self.label = label
                self.meta = meta
                self.start = 0.0

            def _enter(self) -> None:
                self.start = time.perf_counter()
                span._stack.append(span.extra)
                span.extra = {**span.extra, **self.meta}
                logging.debug("SPAN_START label=%s meta=%s", self.label, self.meta)

            def _exit(self) -> None:
                dt = (time.perf_counter() - self.start) * 1000
                logging.debug("SPAN_DONE label=%s dur_ms=%.0f", self.label, dt)
                span.extra = span._stack.pop()

            def __enter__(self):
                self._enter()
                return self

            def __exit__(self, exc_type, exc, tb):
                self._exit()
                return False

            async def __aenter__(self):
                self._enter()
                return self

            async def __aexit__(self, exc_type, exc, tb):
                self._exit()
                return False

        return _Ctx(label, meta)


span = _Span()


if __name__ == "__main__":
    from io import StringIO
    import asyncio

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = [handler]

    # Sync span
    with span("sync", foo=1):
        assert span.extra["foo"] == 1
    assert span.extra == {}
    log = stream.getvalue()
    assert "SPAN_START label=sync" in log
    assert "meta={'foo': 1}" in log
    assert "SPAN_DONE label=sync" in log
    stream.truncate(0)
    stream.seek(0)

    # Async span with task label
    async def main() -> None:
        async with span(task="async", bar=2):
            assert span.extra["bar"] == 2
        assert span.extra == {}

    asyncio.run(main())
    log = stream.getvalue()
    assert "SPAN_START label=async" in log
    assert "meta={'bar': 2}" in log
    assert "SPAN_DONE label=async" in log

    print("OK")
