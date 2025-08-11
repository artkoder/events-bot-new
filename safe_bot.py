import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from cachetools import TTLCache

BACKOFF_DELAYS = [0.5, 1, 2, 4, 8]
NO_RETRY_MESSAGES = (
    "message is not modified",
    "message can't be deleted",
    "message to delete not found",
)


class SafeBot(Bot):
    """Bot wrapper with retry logic and basic no-op protection."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # cache chat_id/message_id -> (text, markup_repr, date)
        self._cache: TTLCache = TTLCache(maxsize=64, ttl=3 * 24 * 3600)

    async def _call(self, func, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt, delay in enumerate(BACKOFF_DELAYS, start=1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 1 and last_exc:
                    logging.warning(
                        "tg call %s retried=%d last_error=%s",
                        func.__name__,
                        attempt - 1,
                        last_exc,
                    )
                return result
            except TelegramBadRequest as e:  # non-retriable?
                msg = (e.message or "").lower()
                if any(m in msg for m in NO_RETRY_MESSAGES):
                    logging.info("tg no-retry error: %s", e.message)
                    return None
                last_exc = e
            except TelegramRetryAfter as e:
                last_exc = e
                delay = max(delay, int(getattr(e, "retry_after", delay)))
            except Exception as e:  # pragma: no cover - network issues
                last_exc = e
            if attempt == len(BACKOFF_DELAYS):
                logging.warning(
                    "tg call %s failed after %d attempts: %s",
                    func.__name__,
                    attempt,
                    last_exc,
                )
                raise last_exc
            await asyncio.sleep(delay)

    async def send_message(self, chat_id: int, text: str, **kwargs):
        result = await self._call(super().send_message, chat_id, text, **kwargs)
        if result:
            key = (result.chat.id, result.message_id)
            self._cache[key] = (
                result.text,
                repr(result.reply_markup) if result.reply_markup else None,
                datetime.now(timezone.utc),
            )
        return result

    async def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: Any | None = None,
        **kwargs,
    ):
        key = (chat_id, message_id)
        cached = self._cache.get(key)
        markup_repr = repr(reply_markup) if reply_markup else None
        if cached and cached[0] == text and cached[1] == markup_repr:
            logging.info("skip edit_message_text no changes")
            return None
        result = await self._call(
            super().edit_message_text,
            text=text,
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=reply_markup,
            **kwargs,
        )
        if result:
            self._cache[key] = (
                result.text,
                repr(result.reply_markup) if result.reply_markup else None,
                datetime.now(timezone.utc),
            )
        return result

    async def edit_message_reply_markup(
        self,
        chat_id: int,
        message_id: int,
        reply_markup: Any | None = None,
        **kwargs,
    ):
        key = (chat_id, message_id)
        cached = self._cache.get(key)
        markup_repr = repr(reply_markup) if reply_markup else None
        if cached and cached[1] == markup_repr:
            logging.info("skip edit_message_reply_markup no changes")
            return None
        result = await self._call(
            super().edit_message_reply_markup,
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=reply_markup,
            **kwargs,
        )
        if result and key in self._cache:
            text, _, date = self._cache[key]
            self._cache[key] = (text, markup_repr, date)
        return result

    async def delete_message(self, chat_id: int, message_id: int, **kwargs):
        key = (chat_id, message_id)
        cached = self._cache.get(key)
        if cached:
            _, _, ts = cached
            if datetime.now(timezone.utc) - ts > timedelta(days=2):
                logging.info("skip delete_message too old")
                return None
        result = await self._call(
            super().delete_message, chat_id=chat_id, message_id=message_id, **kwargs
        )
        if result is not None and key in self._cache:
            del self._cache[key]
        return result
