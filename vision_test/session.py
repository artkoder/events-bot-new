from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cachetools import TTLCache

DetailLevel = Literal["auto", "low", "high"]


@dataclass
class VisionSession:
    detail: DetailLevel = "auto"
    waiting_for_photo: bool = True
    last_texts: dict[str, str] = field(default_factory=dict)


_SESSIONS: TTLCache[int, VisionSession] = TTLCache(maxsize=128, ttl=30 * 60)


def _get_session(user_id: int) -> VisionSession | None:
    session = _SESSIONS.get(user_id)
    if session:
        _SESSIONS[user_id] = session  # refresh TTL
    return session


def get_session(user_id: int, *, create: bool = False) -> VisionSession | None:
    session = _get_session(user_id)
    if session is None and create:
        session = start_session(user_id)
    return session


def start_session(user_id: int) -> VisionSession:
    session = VisionSession()
    _SESSIONS[user_id] = session
    return session


def set_detail(user_id: int, detail: DetailLevel) -> VisionSession:
    session = get_session(user_id, create=True)
    session.detail = detail
    session.waiting_for_photo = True
    return session


def is_waiting(user_id: int) -> bool:
    session = _get_session(user_id)
    return bool(session and session.waiting_for_photo)


def finish_session(user_id: int) -> None:
    _SESSIONS.pop(user_id, None)


def reset_sessions() -> None:
    _SESSIONS.clear()


__all__ = [
    "DetailLevel",
    "VisionSession",
    "finish_session",
    "get_session",
    "is_waiting",
    "reset_sessions",
    "set_detail",
    "start_session",
]
