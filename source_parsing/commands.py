"""Command handler for /parse command.

This module contains the Telegram command handler for source parsing.
Follows project architecture by keeping handlers as separate services.
"""

from __future__ import annotations

import logging
import os
from functools import partial

from aiogram import Bot, types
from aiogram.filters import Command

from db import Database
from models import User
from source_parsing.handlers import (
    run_source_parsing,
    format_parsing_report,
    escape_md,
)

logger = logging.getLogger(__name__)

MAX_TG_MESSAGE_LEN = 3800


def _format_added_events_lines(added_events) -> list[str]:
    source_labels = {
        "dramteatr": "Драмтеатр",
        "muzteatr": "Музтеатр",
        "sobor": "Собор",
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


async def handle_parse_command(message: types.Message, db: Database, bot: Bot) -> None:
    """Handle /parse command to manually trigger source parsing from theatres."""
    logger.info("source_parsing: /parse command received from user_id=%s", message.from_user.id)
    
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not (user and user.is_superadmin):
        logger.warning("source_parsing: access denied user_id=%s", message.from_user.id)
        await bot.send_message(message.chat.id, "Access denied")
        return
    
    logger.info("source_parsing: starting parse for user_id=%s", message.from_user.id)
    
    await bot.send_message(
        message.chat.id,
        "🔄 Запуск парсинга источников (Драмтеатр, Музтеатр, Кафедральный собор)..."
    )
    
    try:
        result = await run_source_parsing(db, bot, chat_id=message.chat.id)
        report = format_parsing_report(result)
        
        await bot.send_message(
            message.chat.id,
            report,
            parse_mode="Markdown",
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
        
        # Send JSON files if available
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
                except Exception as e:
                    logger.warning("source_parsing: failed to send JSON %s: %s", json_path, e)
        
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


def register_parse_command(dp, db: Database, bot: Bot) -> None:
    """Register /parse command handler with dispatcher.
    
    Call this from the main app setup to register the command.
    """
    parse_wrapper = partial(handle_parse_command, db=db, bot=bot)
    dp.message.register(parse_wrapper, Command("parse"))
    logger.info("source_parsing: registered /parse command")
