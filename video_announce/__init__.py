"""Helpers for managing video announcement rendering pipeline."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

__all__ = ["handle_video_callback", "handle_video_command"]


def _lazy_handler(name: str) -> Callable[..., Awaitable[Any]]:
    def _call(*args: Any, **kwargs: Any) -> Awaitable[Any]:
        from . import handlers

        return getattr(handlers, name)(*args, **kwargs)

    return _call


handle_video_callback = _lazy_handler("handle_video_callback")
handle_video_command = _lazy_handler("handle_video_command")
