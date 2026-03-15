from __future__ import annotations

import asyncio
import html
import logging

from aiogram import F, Bot, Router, types
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from db import Database
from source_parsing.telegram.commands import _get_user

from .service import (
    GUIDE_DIGEST_TARGET_CHAT,
    build_guide_digest_preview,
    publish_guide_digest,
    render_guide_sources_summary,
    run_guide_monitor,
)

guide_router = Router()
guide_excursions_router = guide_router
logger = logging.getLogger(__name__)


def get_guide_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🚀 Полный скан", callback_data="guide:scan:full")],
            [InlineKeyboardButton(text="🪶 Лёгкий скан", callback_data="guide:scan:light")],
            [
                InlineKeyboardButton(text="👁 Новые", callback_data="guide:preview:new_occurrences"),
                InlineKeyboardButton(text="⚠️ Last call", callback_data="guide:preview:last_call"),
            ],
            [InlineKeyboardButton(text="📣 Опубликовать новые", callback_data="guide:publish:new_occurrences")],
            [InlineKeyboardButton(text="🗂 Источники", callback_data="guide:sources")],
        ]
    )


@guide_router.message(Command("guide_excursions"))
async def cmd_guide_excursions(message: types.Message, command: CommandObject):
    import main

    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return
    await message.answer(
        (
            "🧭 <b>Guide Excursions Control</b>\n"
            f"Целевой канал публикации: <code>{html.escape(GUIDE_DIGEST_TARGET_CHAT)}</code>"
        ),
        parse_mode="HTML",
        reply_markup=get_guide_keyboard(),
        disable_web_page_preview=True,
    )


@guide_router.message(Command("guide_sources"))
async def cmd_guide_sources(message: types.Message):
    import main

    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return
    await message.answer(await render_guide_sources_summary(db), disable_web_page_preview=True)


@guide_router.message(Command("guide_recent"))
async def cmd_guide_recent(message: types.Message):
    import main

    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return
    preview = await build_guide_digest_preview(db, family="new_occurrences")
    for text in preview["texts"]:
        await message.answer(text, parse_mode="HTML", disable_web_page_preview=True)


@guide_router.message(Command("guide_digest"))
async def cmd_guide_digest(message: types.Message):
    import main

    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return
    res = await publish_guide_digest(
        db,
        message.bot,
        family="new_occurrences",
        chat_id=message.chat.id,
    )
    if not res.get("published"):
        await message.answer("Пока нечего публиковать.")
        return
    await message.answer(
        (
            "✅ Дайджест экскурсий опубликован.\n"
            f"issue_id={res['issue_id']}\n"
            f"target={html.escape(str(res.get('target_chat') or ''))}"
        ),
        parse_mode="HTML",
        disable_web_page_preview=True,
    )


async def _run_scan_task(bot: Bot, db: Database, *, chat_id: int, operator_id: int, mode: str) -> None:
    result = await run_guide_monitor(
        db,
        bot,
        chat_id=chat_id,
        operator_id=operator_id,
        trigger="manual",
        mode=mode,
        send_progress=False,
    )
    lines = [
        "✅ Мониторинг экскурсий завершён" if not result.errors else "⚠️ Мониторинг экскурсий завершён с ошибками",
        f"run_id={result.run_id}",
        f"Источников: {result.metrics['sources_scanned']}",
        f"Постов: {result.metrics['posts_scanned']}",
        f"После prefilter: {result.metrics['posts_prefiltered']}",
        f"Новых выходов: {result.metrics['occurrences_created']}",
        f"Обновлений: {result.metrics['occurrences_updated']}",
        "",
        "Дальше можно открыть «👁 Новые» или сразу нажать «📣 Опубликовать новые».",
    ]
    if result.errors:
        lines.append("")
        lines.append("Ошибки:")
        lines.extend(f"- {err}"[:350] for err in result.errors[:5])
    try:
        await bot.send_message(
            chat_id,
            "\n".join(lines),
            reply_markup=get_guide_keyboard(),
            disable_web_page_preview=True,
        )
    except Exception:
        logger.warning("guide_commands: failed to send post-scan menu", exc_info=True)


@guide_router.callback_query(F.data.startswith("guide:"))
async def handle_guide_callback(callback: CallbackQuery):
    import main

    db = main.get_db()
    if not db:
        await callback.answer("Database not initialized", show_alert=True)
        return
    if not main.has_admin_access(await _get_user(db, callback.from_user.id)):
        await callback.answer("⛔ Access denied", show_alert=True)
        return

    payload = str(callback.data or "")
    parts = payload.split(":")
    action = parts[1] if len(parts) > 1 else ""
    value = parts[2] if len(parts) > 2 else ""

    if action == "scan":
        await callback.answer()
        await callback.message.answer(
            f"🧭 Запускаю {value or 'full'} scan мониторинга экскурсий…",
            disable_web_page_preview=True,
        )
        asyncio.create_task(
            _run_scan_task(
                callback.bot,
                db,
                chat_id=callback.message.chat.id,
                operator_id=callback.from_user.id,
                mode=value or "full",
            )
        )
        return

    if action == "preview":
        await callback.answer()
        family = value or "new_occurrences"
        preview = await build_guide_digest_preview(db, family=family)
        if not preview["items"]:
            await callback.message.answer(
                "Пока нет подходящих карточек для этого дайджеста.",
                disable_web_page_preview=True,
                reply_markup=get_guide_keyboard(),
            )
            return
        header = (
            f"👁 Предпросмотр {family}\nissue_id={preview['issue_id']}\n"
            f"items={len(preview['items'])}, media={len(preview['media_items'])}"
        )
        await callback.message.answer(header, disable_web_page_preview=True)
        for text in preview["texts"]:
            await callback.message.answer(text, parse_mode="HTML", disable_web_page_preview=True)
        return

    if action == "publish":
        await callback.answer()
        res = await publish_guide_digest(
            db,
            callback.bot,
            family=value or "new_occurrences",
            chat_id=callback.message.chat.id,
        )
        if not res.get("published"):
            await callback.message.answer("Пока нечего публиковать.", reply_markup=get_guide_keyboard())
            return
        await callback.message.answer(
            (
                "✅ Публикация завершена.\n"
                f"issue_id={res['issue_id']}\n"
                f"target={html.escape(str(res.get('target_chat') or ''))}\n"
                f"messages={len(res.get('message_ids') or [])}"
            ),
            parse_mode="HTML",
            disable_web_page_preview=True,
            reply_markup=get_guide_keyboard(),
        )
        return

    if action == "sources":
        await callback.answer()
        await callback.message.answer(await render_guide_sources_summary(db), disable_web_page_preview=True)
        return

    await callback.answer("Неизвестное действие", show_alert=True)
