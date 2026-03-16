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
    GUIDE_RECENT_CHANGES_DEFAULT_HOURS,
    build_guide_future_occurrences_message,
    build_guide_templates_message,
    build_guide_digest_preview,
    delete_guide_occurrence,
    delete_guide_template,
    publish_guide_digest,
    render_guide_occurrence_facts,
    render_guide_occurrence_log,
    render_guide_recent_changes,
    render_guide_run_report,
    render_guide_runs_summary,
    render_guide_sources_summary,
    render_guide_template_detail,
    run_guide_monitor,
)

guide_router = Router()
guide_excursions_router = guide_router
logger = logging.getLogger(__name__)


def _split_plain_text(text: str, *, limit: int = 3800) -> list[str]:
    lines = str(text or "").splitlines() or [str(text or "")]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        candidate_len = current_len + len(line) + 1
        if current and candidate_len > limit:
            chunks.append("\n".join(current).strip())
            current = [line]
            current_len = len(line) + 1
        else:
            current.append(line)
            current_len = candidate_len
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


async def _answer_chunks(message: types.Message, chunks: list[str], *, parse_mode: str | None = None) -> None:
    for chunk in chunks:
        await message.answer(
            chunk,
            parse_mode=parse_mode,
            disable_web_page_preview=True,
        )


def get_guide_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🚀 Полный скан", callback_data="guide:scan:full")],
            [InlineKeyboardButton(text="🪶 Лёгкий скан", callback_data="guide:scan:light")],
            [
                InlineKeyboardButton(text="📄 Последний run", callback_data="guide:report:latest"),
                InlineKeyboardButton(text="🕘 Runs 48ч", callback_data="guide:runs:48"),
            ],
            [
                InlineKeyboardButton(text="🗓 Будущие", callback_data="guide:future:1"),
                InlineKeyboardButton(text="🧩 Шаблоны", callback_data="guide:templates:1"),
            ],
            [InlineKeyboardButton(text="📈 Изменения 24ч", callback_data="guide:changes:24")],
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


@guide_router.message(Command("guide_events"))
async def cmd_guide_events(message: types.Message, command: CommandObject):
    import main

    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return
    raw = (command.args or "").strip()
    if raw and not raw.isdigit():
        await message.answer("Использование: /guide_events [page]")
        return
    page = int(raw) if raw else 1
    text, markup = await build_guide_future_occurrences_message(db, page=page)
    await message.answer(text, reply_markup=markup, disable_web_page_preview=True)


@guide_router.message(Command("guide_templates"))
async def cmd_guide_templates(message: types.Message, command: CommandObject):
    import main

    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return
    raw = (command.args or "").strip()
    if raw and not raw.isdigit():
        await message.answer("Использование: /guide_templates [page]")
        return
    page = int(raw) if raw else 1
    text, markup = await build_guide_templates_message(db, page=page)
    await message.answer(text, reply_markup=markup, disable_web_page_preview=True)


@guide_router.message(Command("guide_template"))
async def cmd_guide_template(message: types.Message, command: CommandObject):
    import main

    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return
    raw = (command.args or "").strip()
    if not raw or not raw.isdigit():
        await message.answer("Использование: /guide_template <template_id>")
        return
    await _answer_chunks(
        message,
        _split_plain_text(await render_guide_template_detail(db, template_id=int(raw))),
    )


@guide_router.message(Command("guide_recent_changes"))
async def cmd_guide_recent_changes(message: types.Message, command: CommandObject):
    import main

    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return
    raw = (command.args or "").strip()
    if raw:
        if not raw.isdigit():
            await message.answer("Использование: /guide_recent_changes [hours]")
            return
        hours = int(raw)
        if hours < 1 or hours > 720:
            await message.answer("❌ Укажите окно от 1 до 720 часов. Пример: /guide_recent_changes 24")
            return
    else:
        hours = GUIDE_RECENT_CHANGES_DEFAULT_HOURS
    await _answer_chunks(message, await render_guide_recent_changes(db, hours=hours))


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
    if preview["items"]:
        lines = ["🧾 Recent guide findings"]
        for item in preview["items"][:12]:
            lines.append(
                f"- #{int(item['id'])} {str(item.get('date') or '—')} {html.escape(str(item.get('canonical_title') or ''))}"
            )
        lines.append("")
        lines.append("Факты по карточке: /guide_facts <id>")
        lines.append("Лог источников: /guide_log <id>")
        await message.answer("\n".join(lines), parse_mode="HTML", disable_web_page_preview=True)
    for text in preview["texts"]:
        await message.answer(text, parse_mode="HTML", disable_web_page_preview=True)


@guide_router.message(Command("guide_runs"))
async def cmd_guide_runs(message: types.Message, command: CommandObject):
    import main

    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return
    raw = (command.args or "").strip()
    if raw:
        if not raw.isdigit():
            await message.answer("Использование: /guide_runs [hours]")
            return
        hours = int(raw)
        if hours < 1 or hours > 720:
            await message.answer("❌ Укажите окно от 1 до 720 часов. Пример: /guide_runs 48")
            return
    else:
        hours = 48
    await _answer_chunks(message, await render_guide_runs_summary(db, hours=hours))


@guide_router.message(Command("guide_report"))
async def cmd_guide_report(message: types.Message, command: CommandObject):
    import main

    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return
    raw = (command.args or "").strip()
    if raw and not raw.isdigit():
        await message.answer("Использование: /guide_report [ops_run_id]")
        return
    ops_run_id = int(raw) if raw else None
    await _answer_chunks(message, await render_guide_run_report(db, ops_run_id))


@guide_router.message(Command("guide_facts"))
async def cmd_guide_facts(message: types.Message, command: CommandObject):
    import main

    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return
    raw = (command.args or "").strip()
    if not raw or not raw.isdigit():
        await message.answer("Использование: /guide_facts <occurrence_id>")
        return
    text = await render_guide_occurrence_facts(db, int(raw))
    await _answer_chunks(message, _split_plain_text(text))


@guide_router.message(Command("guide_log"))
async def cmd_guide_log(message: types.Message, command: CommandObject):
    import main

    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return
    raw = (command.args or "").strip()
    if not raw or not raw.isdigit():
        await message.answer("Использование: /guide_log <occurrence_id>")
        return
    await _answer_chunks(message, _split_plain_text(await render_guide_occurrence_log(db, int(raw))))


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
        send_progress=True,
    )
    try:
        if result.ops_run_id:
            for chunk in await render_guide_run_report(db, result.ops_run_id):
                await bot.send_message(
                    chat_id,
                    chunk,
                    disable_web_page_preview=True,
                )
        await bot.send_message(
            chat_id,
            (
                "🧭 Меню обновлено после скана.\n"
                "Дальше можно открыть «👁 Новые», посмотреть «⚠️ Last call», "
                "открыть «📄 Последний run» или сразу нажать «📣 Опубликовать новые»."
            ),
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
            f"items={len(preview['items'])}, media={len(preview['media_items'])}\n"
            f"facts=/guide_facts <id>\n"
            f"log=/guide_log <id>"
        )
        await callback.message.answer(header, disable_web_page_preview=True)
        if preview["items"]:
            lines = ["🧾 IDs in preview:"]
            for item in preview["items"][:12]:
                lines.append(
                    f"- #{int(item['id'])} {str(item.get('date') or '—')} {html.escape(str(item.get('canonical_title') or ''))}"
                )
            await callback.message.answer("\n".join(lines), parse_mode="HTML", disable_web_page_preview=True)
        for text in preview["texts"]:
            await callback.message.answer(text, parse_mode="HTML", disable_web_page_preview=True)
        return

    if action == "future":
        await callback.answer()
        page = int(value) if value.isdigit() else 1
        text, markup = await build_guide_future_occurrences_message(db, page=page)
        await callback.message.edit_text(text, reply_markup=markup, disable_web_page_preview=True)
        return

    if action == "templates":
        await callback.answer()
        page = int(value) if value.isdigit() else 1
        text, markup = await build_guide_templates_message(db, page=page)
        await callback.message.edit_text(text, reply_markup=markup, disable_web_page_preview=True)
        return

    if action == "tplshow":
        await callback.answer()
        template_id = int(value) if value.isdigit() else 0
        if template_id <= 0:
            await callback.answer("Некорректный template_id", show_alert=True)
            return
        await _answer_chunks(
            callback.message,
            _split_plain_text(await render_guide_template_detail(db, template_id=template_id)),
        )
        return

    if action == "changes":
        await callback.answer()
        hours = int(value) if value.isdigit() else GUIDE_RECENT_CHANGES_DEFAULT_HOURS
        await _answer_chunks(callback.message, await render_guide_recent_changes(db, hours=hours))
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

    if action == "report":
        await callback.answer()
        raw = value.strip()
        ops_run_id = None if raw in {"", "latest"} else int(raw) if raw.isdigit() else None
        await _answer_chunks(callback.message, await render_guide_run_report(db, ops_run_id))
        return

    if action == "runs":
        await callback.answer()
        hours = int(value) if value.isdigit() else 48
        await _answer_chunks(callback.message, await render_guide_runs_summary(db, hours=hours))
        return

    if action == "occfacts":
        await callback.answer()
        occurrence_id = int(value) if value.isdigit() else 0
        if occurrence_id <= 0:
            await callback.answer("Некорректный occurrence_id", show_alert=True)
            return
        await _answer_chunks(
            callback.message,
            _split_plain_text(await render_guide_occurrence_facts(db, occurrence_id)),
        )
        return

    if action == "occlog":
        await callback.answer()
        occurrence_id = int(value) if value.isdigit() else 0
        if occurrence_id <= 0:
            await callback.answer("Некорректный occurrence_id", show_alert=True)
            return
        await _answer_chunks(
            callback.message,
            _split_plain_text(await render_guide_occurrence_log(db, occurrence_id)),
        )
        return

    if action == "occdel":
        page = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 1
        occurrence_id = int(value) if value.isdigit() else 0
        if occurrence_id <= 0:
            await callback.answer("Некорректный occurrence_id", show_alert=True)
            return
        result = await delete_guide_occurrence(db, occurrence_id)
        if not result.get("deleted"):
            await callback.answer("Экскурсия не найдена", show_alert=True)
        else:
            await callback.answer(f"Удалена экскурсия #{occurrence_id}")
        text, markup = await build_guide_future_occurrences_message(db, page=page)
        await callback.message.edit_text(text, reply_markup=markup, disable_web_page_preview=True)
        return

    if action == "tpldel":
        page = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 1
        template_id = int(value) if value.isdigit() else 0
        if template_id <= 0:
            await callback.answer("Некорректный template_id", show_alert=True)
            return
        result = await delete_guide_template(db, template_id)
        if not result.get("deleted"):
            await callback.answer("Шаблон не найден", show_alert=True)
        else:
            await callback.answer(f"Удалён шаблон #{template_id}")
        text, markup = await build_guide_templates_message(db, page=page)
        await callback.message.edit_text(text, reply_markup=markup, disable_web_page_preview=True)
        return

    await callback.answer("Неизвестное действие", show_alert=True)
