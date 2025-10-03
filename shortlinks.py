from __future__ import annotations

import logging
import re
from typing import Any, Optional, Protocol
from urllib.parse import urlparse

from db import Database
from models import Event

try:  # pragma: no cover - optional import for typing
    from sqlmodel.ext.asyncio.session import AsyncSession
except Exception:  # pragma: no cover - fallback for type checkers
    AsyncSession = Any  # type: ignore


if False:  # pragma: no cover - typing helper
    from aiogram import Bot  # noqa: F401


class VkApiCallable(Protocol):
    async def __call__(
        self,
        method: str,
        params: dict[str, Any],
        db: Optional[Database] = None,
        bot: Optional["Bot"] = None,
        **kwargs: Any,
    ) -> Any:
        ...


async def _persist_short_link(
    event: Event,
    db: Database | None,
    session: "AsyncSession" | None,
) -> None:
    if not event.id:
        return
    if session is not None:
        try:
            await session.commit()
        except Exception:  # pragma: no cover - log only
            logging.exception("vk_shortlink_session_commit_failed eid=%s", event.id)
        return
    if db is None:
        return
    try:
        async with db.get_session() as commit_session:
            stored = await commit_session.get(Event, event.id)
            if stored is None:
                return
            stored.vk_ticket_short_url = event.vk_ticket_short_url
            stored.vk_ticket_short_key = event.vk_ticket_short_key
            await commit_session.commit()
    except Exception:  # pragma: no cover - log only
        logging.exception("vk_shortlink_persist_failed eid=%s", event.id)


def _log_short_link_action(
    event: Event,
    ticket_link: str,
    short_url: str,
    *,
    action: str,
) -> None:
    logging.info(
        "vk_shortlink_%s eid=%s url=%s short_url=%s",
        action,
        event.id,
        ticket_link,
        short_url,
    )


def _log_short_link_fallback(event: Event, ticket_link: str, reason: str) -> None:
    logging.info(
        "vk_shortlink_fallback eid=%s url=%s reason=%s",
        event.id,
        ticket_link,
        reason,
    )


async def ensure_vk_short_ticket_link(
    event: Event,
    db: Database | None,
    *,
    session: "AsyncSession" | None = None,
    bot: "Bot" | None = None,
    vk_api_fn: VkApiCallable | None = None,
) -> tuple[str, str] | None:
    """Ensure that an event has a stored VK short ticket link.

    Returns the ``(short_url, key)`` tuple or ``None`` on failure.
    """

    ticket_link = (event.ticket_link or "").strip()
    if not ticket_link:
        _log_short_link_fallback(event, ticket_link, "empty_ticket_link")
        return None

    parsed = urlparse(ticket_link)
    host = parsed.netloc.lower()
    if host == "vk.cc":
        key = parsed.path.strip("/")
        if not key:
            logging.warning(
                "vk_shortlink_invalid_path eid=%s url=%s", event.id, ticket_link
            )
            _log_short_link_fallback(event, ticket_link, "vk_shortlink_invalid_path")
            return None
        short_url = f"https://vk.cc/{key}"
        if (
            event.vk_ticket_short_url != short_url
            or event.vk_ticket_short_key != key
        ):
            event.vk_ticket_short_url = short_url
            event.vk_ticket_short_key = key
            await _persist_short_link(event, db, session)
            _log_short_link_action(
                event,
                ticket_link,
                short_url,
                action="saved",
            )
        else:
            _log_short_link_action(
                event,
                ticket_link,
                short_url,
                action="reused",
            )
        return short_url, key

    if event.vk_ticket_short_url and event.vk_ticket_short_key:
        _log_short_link_action(
            event,
            ticket_link,
            event.vk_ticket_short_url,
            action="reused",
        )
        return event.vk_ticket_short_url, event.vk_ticket_short_key

    if vk_api_fn is None:
        logging.warning("vk_shortlink_no_api eid=%s url=%s", event.id, ticket_link)
        _log_short_link_fallback(event, ticket_link, "vk_shortlink_no_api")
        return None

    params = {"url": ticket_link}
    try:
        response = await vk_api_fn("utils.getShortLink", params, db, bot)
    except Exception:
        logging.exception("vk_shortlink_fetch_failed eid=%s", event.id)
        return None

    if isinstance(response, dict):
        payload = response.get("response", response)
    else:
        payload = response
    if not isinstance(payload, dict):
        logging.error("vk_shortlink_invalid_response eid=%s resp=%r", event.id, response)
        _log_short_link_fallback(event, ticket_link, "vk_shortlink_invalid_response")
        return None

    key = payload.get("key")
    short_url = payload.get("short_url")
    if not short_url and key:
        short_url = f"https://vk.cc/{key}"
    if not short_url or not key:
        logging.error(
            "vk_shortlink_missing_data eid=%s resp=%r", event.id, payload
        )
        _log_short_link_fallback(event, ticket_link, "vk_shortlink_missing_data")
        return None

    event.vk_ticket_short_url = short_url
    event.vk_ticket_short_key = key
    await _persist_short_link(event, db, session)
    _log_short_link_action(
        event,
        ticket_link,
        short_url,
        action="saved",
    )
    return short_url, key


def format_vk_short_url(short_url: str) -> str:
    """Return the VK short URL without the scheme prefix."""

    return re.sub(r"^https?://", "", short_url, flags=re.IGNORECASE)
