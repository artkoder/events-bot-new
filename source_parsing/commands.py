"""Command handler for /parse command.

This module contains the Telegram command handler for source parsing.
Follows project architecture by keeping handlers as separate services.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import asyncio

from aiogram import Bot, types
from aiogram.filters import Command

from db import Database
from models import User
from source_parsing.handlers import (
    run_source_parsing,
    format_parsing_report,
    escape_md,
    run_diagnostic_parse,
)
from net import http_call

logger = logging.getLogger(__name__)

MAX_TG_MESSAGE_LEN = 3800
PARSE_LOCK = asyncio.Lock()

SOURCE_PARSING_GUARD_PATH = Path("/data/parse_debug/source_parsing_guard.json")
SOURCE_PARSING_GUARD_URLS = {
    "dramteatr": "https://dramteatr39.ru/afisha",
    "muzteatr": "https://muzteatr39.ru/action/cat/afisha/",
    "sobor": "https://sobor39.ru/events/concerts/night/",
    "tretyakov": "https://kaliningrad.tretyakovgallery.ru/events/",
    "philharmonia": "https://filarmonia39.ru/?event",
    "qtickets": "https://kaliningrad.qtickets.events",
}


def _load_source_parsing_guard() -> dict:
    if not SOURCE_PARSING_GUARD_PATH.exists():
        return {}
    try:
        return json.loads(SOURCE_PARSING_GUARD_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("source_parsing: guard read failed: %s", e)
        return {}


def _save_source_parsing_guard(signatures: dict[str, str]) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "signatures": signatures,
    }
    try:
        SOURCE_PARSING_GUARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        SOURCE_PARSING_GUARD_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning("source_parsing: guard write failed: %s", e)


async def _collect_source_parsing_signatures() -> dict[str, str]:
    signatures: dict[str, str] = {}
    for name, url in SOURCE_PARSING_GUARD_URLS.items():
        try:
            response = await http_call(
                f"source_parsing_guard_{name}",
                "GET",
                url,
                timeout=20,
                retries=2,
                backoff=1.0,
                headers={"User-Agent": "events-bot/1.0"},
            )
            if response.status_code >= 400:
                raise RuntimeError(f"status={response.status_code}")
            signatures[name] = hashlib.sha256(response.content).hexdigest()
        except Exception as e:
            logger.warning(
                "source_parsing: guard fetch failed source=%s error=%s", name, e
            )
            return {}
    return signatures


async def _update_source_parsing_guard(signatures: dict[str, str] | None = None) -> None:
    if signatures is None:
        signatures = await _collect_source_parsing_signatures()
    if signatures:
        _save_source_parsing_guard(signatures)


def _format_added_events_lines(added_events) -> list[str]:
    source_labels = {
        "dramteatr": "Драмтеатр",
        "muzteatr": "Музтеатр",
        "sobor": "Собор",
        "tretyakov": "Третьяковка",
        "philharmonia": "Филармония",
        "qtickets": "Qtickets",
    }
    lines = [f"📌 **Добавленные события:** {len(added_events)}", ""]
    for item in added_events:
        title = escape_md(item.title or "Без названия")
        url = item.telegraph_url
        if url:
            line = f"• [{title}]({url})"
        else:
            line = f"• {title} — телеграф не создан"
        suffix_parts = []
        if item.date:
            suffix_parts.append(escape_md(item.date))
        if item.time:
            suffix_parts.append(escape_md(item.time))
        source_label = source_labels.get(item.source or "", item.source or "")
        if source_label:
            suffix_parts.append(escape_md(source_label))
        if suffix_parts:
            line += f" — {', '.join(suffix_parts)}"
        lines.append(line)
    return lines


def _format_updated_events_lines(updated_events) -> list[str]:
    """Format updated events list for display."""
    source_labels = {
        "dramteatr": "Драмтеатр",
        "muzteatr": "Музтеатр",
        "sobor": "Собор",
        "tretyakov": "Третьяковка",
        "philharmonia": "Филармония",
        "qtickets": "Qtickets",
    }
    lines = [f"🔄 **Обновлённые события:** {len(updated_events)}", ""]
    for item in updated_events:
        title = escape_md(item.title or "Без названия")
        url = item.telegraph_url
        if url:
            line = f"• [{title}]({url})"
        else:
            line = f"• {title}"
        suffix_parts = []
        if item.date:
            suffix_parts.append(escape_md(item.date))
        if item.time:
            suffix_parts.append(escape_md(item.time))
        source_label = source_labels.get(item.source or "", item.source or "")
        if source_label:
            suffix_parts.append(escape_md(source_label))
        if suffix_parts:
            line += f" — {', '.join(suffix_parts)}"
        lines.append(line)
    return lines


def _chunk_lines(lines: list[str], max_len: int = MAX_TG_MESSAGE_LEN) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_len:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


async def handle_parse_check_callback(callback_query: types.CallbackQuery, bot: Bot) -> None:
    """Handle callback for diagnostic parse selection."""
    source_map = {
        "parse_check_dramteatr": "dramteatr",
        "parse_check_muzteatr": "muzteatr",
        "parse_check_sobor": "sobor",
        "parse_check_tretyakov": "tretyakov",
        "parse_check_philharmonia": "philharmonia",
    }
    source = source_map.get(callback_query.data)
    if not source:
        await callback_query.answer("Неизвестный источник")
        return
        
    await callback_query.answer()
    
    # Run diagnostic parse in background
    asyncio.create_task(
        run_diagnostic_parse(bot, callback_query.from_user.id, source)
    )


async def handle_parse_command(message: types.Message, db: Database, bot: Bot) -> None:
    """Handle /parse command to manually trigger source parsing from theatres."""
    logger.info("source_parsing: /parse command received from user_id=%s", message.from_user.id)
    
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not (user and user.is_superadmin):
        logger.warning("source_parsing: access denied user_id=%s", message.from_user.id)
        await bot.send_message(message.chat.id, "Access denied")
        return
        
    # Check for arguments (e.g. "/parse check")
    args = ""
    if message.text:
        parts = message.text.split(maxsplit=1)
        if len(parts) > 1:
            args = parts[1].strip().lower()
            
    if args == "check":
        keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
            [
                types.InlineKeyboardButton(text="Драмтеатр", callback_data="parse_check_dramteatr"),
                types.InlineKeyboardButton(text="Музтеатр", callback_data="parse_check_muzteatr"),
            ],
            [
                types.InlineKeyboardButton(text="Собор", callback_data="parse_check_sobor"),
                types.InlineKeyboardButton(text="Третьяковка", callback_data="parse_check_tretyakov"),
            ],
            [
                types.InlineKeyboardButton(text="Филармония", callback_data="parse_check_philharmonia"),
            ]
        ])
        await bot.send_message(
            message.chat.id,
            "🛠 **Диагностический режим**\nВыберите источник для тестового запуска (без сохранения в БД):",
            reply_markup=keyboard,
            parse_mode="Markdown"
        )
        return
    
    logger.info("source_parsing: starting parse for user_id=%s", message.from_user.id)

    if PARSE_LOCK.locked():
        await bot.send_message(
            message.chat.id,
            "⏳ Парсинг уже выполняется. Дождитесь завершения текущего запуска.",
        )
        return
    
    await bot.send_message(
        message.chat.id,
        "🔄 Запуск парсинга источников (Драмтеатр, Музтеатр, Кафедральный собор, Третьяковка)..."
    )

    async with PARSE_LOCK:
        try:
            result = await run_source_parsing(db, bot, chat_id=message.chat.id)
            report = format_parsing_report(result)
            
            try:
                await bot.send_message(
                    message.chat.id,
                    report,
                    parse_mode="Markdown",
                )
            except Exception as e:
                logger.warning("source_parsing: failed to send report markdown: %s", e)
                await bot.send_message(
                    message.chat.id,
                    report.replace("**", ""),
                )

            if getattr(result, "added_events", None):
                lines = _format_added_events_lines(result.added_events)
                for chunk in _chunk_lines(lines):
                    await bot.send_message(
                        message.chat.id,
                        chunk,
                        parse_mode="Markdown",
                    )
            else:
                await bot.send_message(
                    message.chat.id,
                    "ℹ️ Новых событий не добавлено.",
                )
            
            # Show updated events with Telegraph links
            if getattr(result, "updated_events", None):
                lines = _format_updated_events_lines(result.updated_events)
                for chunk in _chunk_lines(lines):
                    await bot.send_message(
                        message.chat.id,
                        chunk,
                        parse_mode="Markdown",
                    )
            
            # Send JSON files if available
            json_files_sent = 0
            if hasattr(result, 'json_file_paths') and result.json_file_paths:
                from aiogram.types import FSInputFile
                for json_path in result.json_file_paths:
                    try:
                        import os
                        if os.path.exists(json_path):
                            filename = os.path.basename(json_path)
                            json_file = FSInputFile(json_path, filename=filename)
                            await bot.send_document(message.chat.id, json_file)
                            logger.info("source_parsing: sent JSON file %s", filename)
                            json_files_sent += 1
                        else:
                            logger.warning("source_parsing: JSON file not found %s", json_path)
                    except Exception as e:
                        logger.warning("source_parsing: failed to send JSON %s: %s", json_path, e)
            
            # Warn if no JSON files were sent
            if json_files_sent == 0 and result.total_events > 0:
                await bot.send_message(
                    message.chat.id,
                    "⚠️ JSON файлы парсера недоступны (временные файлы удалены).",
                )
            
            # Send log file if available
            if result.log_file_path:
                try:
                    from aiogram.types import FSInputFile
                    import os
                    if os.path.exists(result.log_file_path):
                        log_file = FSInputFile(result.log_file_path, filename="kaggle_log.txt")
                        await bot.send_document(message.chat.id, log_file)
                except Exception as e:
                    logger.warning("source_parsing: failed to send log file: %s", e)
                    
        except Exception as e:
            logger.exception("source_parsing: failed")
            await bot.send_message(
                message.chat.id,
                f"❌ Ошибка парсинга: {e}"
            )


async def source_parsing_scheduler(db: Database, bot: Bot, *, run_id: str | None = None) -> None:
    """Scheduled job for source parsing (runs at 02:00 AM).
    
    Called by APScheduler when ENABLE_SOURCE_PARSING=1.
    """
    logger.info("source_parsing_scheduler started run_id=%s", run_id)
    
    try:
        result = await run_source_parsing(db, bot)

        should_update_guard = result.total_events > 0 or not result.errors
        if should_update_guard:
            await _update_source_parsing_guard()
        
        # Send report to admin chat if configured
        admin_chat_id = os.getenv("ADMIN_CHAT_ID")
        if admin_chat_id and (result.stats_by_source or result.errors):
            report = format_parsing_report(result)
            try:
                await bot.send_message(
                    int(admin_chat_id),
                    f"📊 Автоматический парсинг источников завершён\n\n{report}",
                    parse_mode="Markdown",
                )
            except Exception as e:
                logger.warning("source_parsing: failed to send report: %s", e)
        
        logger.info(
            "source_parsing_scheduler complete run_id=%s total=%d",
            run_id,
            result.total_events,
        )
    except Exception as e:
        logger.exception("source_parsing_scheduler failed run_id=%s", run_id)


async def source_parsing_scheduler_if_changed(
    db: Database,
    bot: Bot,
    *,
    run_id: str | None = None,
) -> None:
    """Scheduled job for source parsing with change guard.

    Skips Kaggle if source pages did not change since the last successful run.
    """
    logger.info("source_parsing_scheduler_if_changed started run_id=%s", run_id)
    try:
        signatures = await _collect_source_parsing_signatures()
        if not signatures:
            logger.info("source_parsing_guard: signatures unavailable, running parse")
        else:
            guard_state = _load_source_parsing_guard()
            if guard_state.get("signatures") == signatures:
                logger.info("source_parsing_guard: no changes, skipping parse")
                return

        result = await run_source_parsing(db, bot)
        should_update_guard = result.total_events > 0 or not result.errors
        if should_update_guard and signatures:
            await _update_source_parsing_guard(signatures)
        elif should_update_guard:
            await _update_source_parsing_guard()

        admin_chat_id = os.getenv("ADMIN_CHAT_ID")
        if admin_chat_id and (result.stats_by_source or result.errors):
            report = format_parsing_report(result)
            try:
                await bot.send_message(
                    int(admin_chat_id),
                    f"📊 Автоматический парсинг источников завершён\n\n{report}",
                    parse_mode="Markdown",
                )
            except Exception as e:
                logger.warning("source_parsing: failed to send report: %s", e)

        logger.info(
            "source_parsing_scheduler_if_changed complete run_id=%s total=%d",
            run_id,
            result.total_events,
        )
    except Exception:
        logger.exception("source_parsing_scheduler_if_changed failed run_id=%s", run_id)


def register_parse_command(dp, db: Database, bot: Bot) -> None:
    """Register /parse command handler with dispatcher.
    
    Call this from the main app setup to register the command.
    """
    parse_wrapper = partial(handle_parse_command, db=db, bot=bot)
    dp.message.register(parse_wrapper, Command("parse"))
    
    # Register diagnostic callbacks
    callback_wrapper = partial(handle_parse_check_callback, bot=bot)
    # Using lambda to filter callbacks
    dp.callback_query.register(callback_wrapper, lambda c: c.data and c.data.startswith("parse_check_"))
    
    logger.info("source_parsing: registered /parse command and diagnostic callbacks")
