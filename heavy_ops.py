from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import AsyncIterator, Literal

from runtime import require_main_attr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeavyOpMeta:
    kind: str
    trigger: str
    started_monotonic: float
    run_id: str | None = None
    operator_id: int | None = None
    chat_id: int | None = None


_HEAVY_DEPTH: ContextVar[int] = ContextVar("heavy_ops_depth", default=0)
_FALLBACK_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(1)
_CURRENT_META: HeavyOpMeta | None = None


def _resolve_semaphore() -> asyncio.Semaphore:
    try:
        sem = require_main_attr("HEAVY_SEMAPHORE")
    except Exception:
        return _FALLBACK_SEMAPHORE
    if isinstance(sem, asyncio.Semaphore):
        return sem
    # Defensive fallback: keep process alive even if main defines a different type.
    if hasattr(sem, "acquire") and hasattr(sem, "release"):
        return sem  # type: ignore[return-value]
    return _FALLBACK_SEMAPHORE


def current_heavy_meta() -> HeavyOpMeta | None:
    return _CURRENT_META


def describe_heavy_meta(meta: HeavyOpMeta | None) -> str:
    if not meta:
        return "неизвестно (meta недоступна)"
    took_sec = max(0.0, float(time.monotonic() - meta.started_monotonic))
    took_min = int(took_sec // 60)
    if took_min >= 60:
        took_txt = f"{took_min // 60}ч {took_min % 60}м"
    elif took_min:
        took_txt = f"{took_min}м"
    else:
        took_txt = f"{took_sec:.0f}с"
    run_id_part = f", run_id={meta.run_id}" if meta.run_id else ""
    operator_part = f", operator_id={meta.operator_id}" if meta.operator_id else ""
    return f"{meta.kind} ({meta.trigger}), идёт уже {took_txt}{run_id_part}{operator_part}"


def _is_try_mode(mode: str) -> bool:
    return (mode or "").strip().lower() in {"try", "skip", "nonblocking", "non-blocking"}


@asynccontextmanager
async def heavy_operation(
    *,
    kind: str,
    trigger: str,
    mode: Literal["wait", "try"] = "wait",
    timeout_sec: float = 0.2,
    run_id: str | None = None,
    operator_id: int | None = None,
    chat_id: int | None = None,
) -> AsyncIterator[bool]:
    """
    Global (process-wide) re-entrant gate for long-running operations.

    - Uses `main.HEAVY_SEMAPHORE` when available (without importing main).
    - Re-entrant within the same task/context: nested heavy ops do not deadlock.
    - In `mode='try'`, yields `False` when the semaphore is busy.
    """

    depth = int(_HEAVY_DEPTH.get() or 0)
    if depth > 0:
        token = _HEAVY_DEPTH.set(depth + 1)
        try:
            yield True
        finally:
            _HEAVY_DEPTH.reset(token)
        return

    sem = _resolve_semaphore()
    acquired = False
    try:
        if _is_try_mode(mode):
            if getattr(sem, "locked", None) and sem.locked():  # type: ignore[truthy-function]
                acquired = False
            else:
                # Use a tiny timeout to avoid blocking the scheduler, but still reduce races.
                effective_timeout = max(0.001, float(timeout_sec))
                await asyncio.wait_for(sem.acquire(), timeout=effective_timeout)
                acquired = True
        else:
            await sem.acquire()
            acquired = True
    except asyncio.TimeoutError:
        acquired = False

    if not acquired:
        yield False
        return

    global _CURRENT_META
    _CURRENT_META = HeavyOpMeta(
        kind=str(kind or "").strip() or "heavy_op",
        trigger=str(trigger or "").strip() or "manual",
        started_monotonic=float(time.monotonic()),
        run_id=(str(run_id).strip() if run_id else None),
        operator_id=int(operator_id) if operator_id is not None else None,
        chat_id=int(chat_id) if chat_id is not None else None,
    )
    token = _HEAVY_DEPTH.set(1)
    try:
        yield True
    finally:
        _HEAVY_DEPTH.reset(token)
        _CURRENT_META = None
        try:
            sem.release()
        except Exception:
            logger.warning("heavy_ops: failed to release semaphore", exc_info=True)
