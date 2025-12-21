from __future__ import annotations

import html
import logging
from datetime import date
from typing import Callable

from aiogram import types
from models import Event, User
from .kaggle_client import KaggleClient

from db import Database
from .scenario import VideoAnnounceScenario, handle_prefix_action

logger = logging.getLogger(__name__)


async def _load_user(db: Database, user_id: int) -> User | None:
    async with db.get_session() as session:
        return await session.get(User, user_id)


async def handle_video_command(message: types.Message, db: Database, bot) -> None:
    scenario = VideoAnnounceScenario(db, bot, message.chat.id, message.from_user.id)
    await scenario.show_menu()


async def handle_kaggle_test(message: types.Message, db: Database, bot) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not user or not user.is_superadmin:
        await bot.send_message(message.chat.id, "Not authorized")
        return
    client = KaggleClient()
    try:
        title = client.kaggle_test()
    except Exception:
        logger.exception("kaggletest failed")
        await bot.send_message(message.chat.id, "Kaggle API error")
        return
    await bot.send_message(message.chat.id, f"Kaggle OK: {title}")


async def _rerender_events(
    db: Database,
    bot,
    callback: types.CallbackQuery,
    event: Event,
    *,
    build_events_message: Callable,
    get_tz_offset: Callable,
    offset_to_timezone: Callable,
    creator_filter: int | None,
) -> None:
    try:
        tz_offset = await get_tz_offset(db)
        tz = offset_to_timezone(tz_offset)
    except Exception:
        logger.exception("video_announce: failed to fetch timezone, fallback to UTC")
        from datetime import timezone

        tz = timezone.utc
    try:
        target_day = date.fromisoformat(event.date.split("..", 1)[0])
    except Exception:
        await callback.answer("Не удалось обновить список", show_alert=True)
        return
    text, markup = await build_events_message(db, target_day, tz, creator_filter)
    await callback.message.edit_text(text, reply_markup=markup)
    await callback.answer(f"Видео: {event.video_include_count}")


async def handle_video_count(
    callback: types.CallbackQuery,
    db: Database,
    bot,
    *,
    build_events_message: Callable,
    get_tz_offset: Callable,
    offset_to_timezone: Callable,
) -> bool:
    parts = callback.data.split(":")
    if len(parts) < 2:
        return False
    try:
        event_id = int(parts[1])
    except ValueError:
        await callback.answer("Некорректный идентификатор", show_alert=True)
        return True

    async with db.get_session() as session:
        user = await session.get(User, callback.from_user.id)
        event = await session.get(Event, event_id)
        if not user or (user.blocked or (user.is_partner and event and event.creator_id != user.user_id)):
            await callback.answer("Not authorized", show_alert=True)
            return True
        if not event:
            await callback.answer("Событие не найдено", show_alert=True)
            return True
        event.video_include_count = ((event.video_include_count or 0) + 1) % 6
        await session.commit()

    creator_filter = user.user_id if user and user.is_partner else None
    await _rerender_events(
        db,
        bot,
        callback,
        event,
        build_events_message=build_events_message,
        get_tz_offset=get_tz_offset,
        offset_to_timezone=offset_to_timezone,
        creator_filter=creator_filter,
    )
    return True


async def handle_video_callback(
    callback: types.CallbackQuery,
    db: Database,
    bot,
    *,
    build_events_message: Callable,
    get_tz_offset: Callable,
    offset_to_timezone: Callable,
) -> None:
    if not callback.data:
        return
    data = callback.data
    if data.startswith("vidcnt:"):
        handled = await handle_video_count(
            callback,
            db,
            bot,
            build_events_message=build_events_message,
            get_tz_offset=get_tz_offset,
            offset_to_timezone=offset_to_timezone,
        )
        if handled:
            return

    scenario = VideoAnnounceScenario(db, bot, callback.message.chat.id, callback.from_user.id)
    prefix = data.split(":", 1)[0]
    handled = await handle_prefix_action(prefix, callback, scenario)
    if handled:
        return
    await callback.answer("Неизвестное действие", show_alert=False)
