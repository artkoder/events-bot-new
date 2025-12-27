from __future__ import annotations

import asyncio
import html
import logging
from datetime import date
from typing import Callable

from aiogram import types
from models import Event, User, VideoAnnounceSession
from main import TELEGRAPH_TOKEN_FILE, get_telegraph_token_info
from .kaggle_client import KaggleClient
from .poller import remember_status_message, update_status_message

from db import Database
from .scenario import (
    PendingInstruction,
    PendingIntroText,
    VideoAnnounceScenario,
    handle_prefix_action,
    is_waiting_instruction,
    is_waiting_intro_text,
    take_pending_instruction,
    take_pending_intro_text,
)

logger = logging.getLogger(__name__)


async def _load_user(db: Database, user_id: int) -> User | None:
    async with db.get_session() as session:
        return await session.get(User, user_id)


async def handle_video_command(message: types.Message, db: Database, bot) -> None:
    token_info = get_telegraph_token_info(create_if_missing=False)
    logger.info(
        "telegraph_token_diagnostics",
        extra={
            "env_present": token_info.env_present,
            "token_file_path": TELEGRAPH_TOKEN_FILE,
            "token_file_exists": token_info.token_file_exists,
            "token_file_readable": token_info.token_file_readable,
            "token_source": token_info.source,
        },
    )
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
    except Exception as e:
        logger.exception("kaggletest failed")
        await bot.send_message(
            message.chat.id, f"Kaggle API error: {type(e).__name__}: {e}"
        )
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
    scenario = VideoAnnounceScenario(db, bot, callback.message.chat.id, callback.from_user.id)
    if data.startswith("vidkstat:"):
        try:
            _, raw_session_id = data.split(":", 1)
            session_id = int(raw_session_id)
        except ValueError:
            await callback.answer("Некорректный идентификатор", show_alert=True)
            return
        async with db.get_session() as session:
            session_obj = await session.get(VideoAnnounceSession, session_id)
        if not session_obj:
            await callback.answer("Сессия не найдена", show_alert=True)
            return
        client = KaggleClient()
        status: dict | None = {}
        if session_obj.kaggle_kernel_ref:
            try:
                status = await asyncio.to_thread(
                    client.get_kernel_status, session_obj.kaggle_kernel_ref
                )
            except Exception:
                logger.exception("video_announce: manual status fetch failed")
                status = {}
        remember_status_message(
            session_id, callback.message.chat.id, callback.message.message_id
        )
        await update_status_message(
            bot,
            session_obj,
            status,
            chat_id=callback.message.chat.id,
            message_id=callback.message.message_id,
            allow_send=True,
        )
        await callback.answer("Обновлено")
        return
    if data.startswith("vidsel:"):
        try:
            _, session_id, action = data.split(":", 2)
            msg = await scenario.adjust_selection_params(
                int(session_id), action, callback.message
            )
        except Exception:
            logger.exception("video_announce: selection adjust failed")
            msg = "Ошибка"
        await callback.answer(msg or "Обновлено", show_alert=msg not in {"Обновлено"})
        return
    if data.startswith("vidtoggle:"):
        try:
            _, session_id, event_id = data.split(":", 2)
            msg = await scenario.toggle_item(int(session_id), int(event_id), callback.message)
        except Exception:
            logger.exception("video_announce: toggle failed")
            msg = "Ошибка"
        await callback.answer(msg, show_alert=msg != "Обновлено")
        return
    if data.startswith("vidintro:"):
        try:
            _, session_id, action = data.split(":", 2)
            if action != "edit":
                await callback.answer("Неизвестное действие", show_alert=True)
                return
            msg = await scenario.prompt_intro_override(int(session_id))
        except Exception:
            logger.exception("video_announce: intro prompt failed")
            msg = "Ошибка"
        await callback.answer(msg or "Готово", show_alert=msg not in {"Ожидаю интро", "Готово"})
        return
    if data.startswith("vidjson:"):
        try:
            _, session_id = data.split(":", 1)
            msg = await scenario.preview_json(int(session_id)) or ""
        except Exception:
            logger.exception("video_announce: json preview failed")
            msg = "Ошибка"
        await callback.answer(msg or "Готово", show_alert=bool(msg and msg != "Сформировано"))
        return
    if data.startswith("vidrender:"):
        try:
            _, session_id = data.split(":", 1)
            # Old: msg = await scenario.start_render(int(session_id), message=callback.message)
            # New: Show kernel selection first
            msg = await scenario.show_kernel_selection(int(session_id), message=callback.message)
        except Exception:
            logger.exception("video_announce: render start failed")
            msg = "Ошибка"
        await callback.answer(msg or "Готово", show_alert=msg != "Выбор kernel")
        return
    if data.startswith("vidkernel:"):
        try:
            _, session_id, kernel_ref = data.split(":", 2)
            msg = await scenario.save_kernel_and_start(int(session_id), kernel_ref, message=callback.message)
        except Exception:
            logger.exception("video_announce: kernel save failed")
            msg = "Ошибка"
        await callback.answer(msg or "Готово", show_alert=msg not in {"Рендеринг запущен", "Готово"})
        return
    # --- Pattern Selection Callbacks ---
    if data.startswith("vidpat:"):
        try:
            _, session_id, pattern = data.split(":", 2)
            msg = await scenario.switch_pattern(int(session_id), pattern, callback.message)
        except Exception:
            logger.exception("video_announce: pattern switch failed")
            msg = "Ошибка"
        await callback.answer(msg or "Обновлено", show_alert=msg not in {"Паттерн обновлён", "Обновлено", "Выбор паттерна"})
        return
    if data.startswith("vidpatconfirm:"):
        try:
            _, session_id = data.split(":", 1)
            msg = await scenario.confirm_pattern(int(session_id), callback.message)
        except Exception:
            logger.exception("video_announce: pattern confirm failed")
            msg = "Ошибка"
        await callback.answer(msg or "Готово", show_alert=msg not in {"Выбор kernel", "Готово"})
        return
    if data.startswith("vidpatshow:"):
        try:
            _, session_id = data.split(":", 1)
            msg = await scenario.show_pattern_selection(int(session_id), message=None)
        except Exception:
            logger.exception("video_announce: pattern show failed")
            msg = "Ошибка"
        await callback.answer(msg or "Выбор паттерна", show_alert=msg not in {"Выбор паттерна", "Готово"})
        return
    # --- Sorting Callbacks ---
    if data.startswith("vidsort:"):
        try:
            parts = data.split(":")
            session_id = int(parts[1])
            action = parts[2] if len(parts) > 2 else "show"
            
            if action == "show":
                msg = await scenario.show_sort_screen(session_id, callback.message)
            elif action == "noop":
                msg = ""
            elif action == "done":
                msg = await scenario.finish_sorting(session_id, callback.message)
            elif action in ("up", "down"):
                event_id = int(parts[3]) if len(parts) > 3 else 0
                msg = await scenario.move_item(session_id, event_id, action, callback.message)
            else:
                msg = "Неизвестное действие"
        except Exception:
            logger.exception("video_announce: sort callback failed")
            msg = "Ошибка"
        await callback.answer(msg or "OK", show_alert=msg not in {"Сортировка", "Перемещено", "Готово", "OK", ""})
        return
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

    prefix = data.split(":", 1)[0]
    handled = await handle_prefix_action(prefix, callback, scenario)
    if handled:
        return
    await callback.answer("Неизвестное действие", show_alert=False)


async def handle_instruction_message(
    message: types.Message, db: Database, bot
) -> None:
    pending: PendingInstruction | None = take_pending_instruction(message.from_user.id)
    if not pending:
        return
    scenario = VideoAnnounceScenario(db, bot, message.chat.id, message.from_user.id)
    text = (message.text or message.caption or "").strip()
    msg = await scenario.apply_instruction(
        pending.session_id,
        text or None,
        reuse_candidates=pending.reuse_candidates,
        pending=pending,
    )
    await bot.send_message(message.chat.id, msg or "Готово")


async def handle_intro_message(message: types.Message, db: Database, bot) -> None:
    pending: PendingIntroText | None = take_pending_intro_text(message.from_user.id)
    if not pending:
        return
    scenario = VideoAnnounceScenario(db, bot, message.chat.id, message.from_user.id)
    text = (message.text or message.caption or "").strip()
    msg = await scenario.save_intro_override(pending.session_id, text or None)
    await bot.send_message(message.chat.id, msg or "Готово")
