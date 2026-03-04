import logging
import asyncio
import shlex
import math
import os
import html
from datetime import datetime
import tempfile
from pathlib import Path
from aiogram import Router, types, Bot, F
from aiogram.filters import Command, CommandObject
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from db import Database
from models import TelegramScannedMessage, TelegramSource
from source_parsing.post_metrics import load_telegram_popularity_overview
from telegram_sources import normalize_tg_username
from telegram_sources_seed import seed_telegram_sources
from heavy_ops import heavy_operation
from .service import (
    find_latest_telegram_results_json,
    find_recent_telegram_results_json,
    preview_telegram_recreate_reimport,
    recreate_telegram_events_from_results,
    run_telegram_import_from_results,
    run_telegram_dev_recreate_reimport,
    run_telegram_monitor,
)

tg_router = Router()
logger = logging.getLogger(__name__)
tg_monitor_router = tg_router

# State management
# user_id -> mode ("add" | "loc:<source_id>" | "ticket:<source_id>" | "festival:<source_id>")
adding_source_sessions: dict[int, str] = {}
# user_id -> last computed list of local result paths (newest first)
recent_import_results: dict[int, list[str]] = {}
# user_id -> selected local telegram_results.json path for import mode selection
selected_import_result: dict[int, str] = {}
_monitor_lock = asyncio.Lock()

SOURCES_PAGE_SIZE = max(3, min(int((os.getenv("TG_SOURCES_PAGE_SIZE") or "8") or 8), 15))


def _is_dev_mode_enabled() -> bool:
    return (os.getenv("DEV_MODE") or "").strip() == "1"


def get_tg_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🚀 Запустить мониторинг", callback_data="tg:run")],
            [InlineKeyboardButton(text="♻️ Импорт из JSON", callback_data="tg:rerun_import")],
            [InlineKeyboardButton(text="➕ Добавить источник", callback_data="tg:add")],
            [InlineKeyboardButton(text="🧩 Синхронизировать источники", callback_data="tg:seed")],
            [InlineKeyboardButton(text="📋 Список источников", callback_data="tg:list")],
        ]
    )

@tg_router.message(Command("tg"))
async def cmd_tg(message: types.Message, command: CommandObject):
    """
    Handle /tg command with UI.
    """
    import main
    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return

    # Check admin
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return

    await message.answer(
        "🤖 <b>Telegram Monitor Control</b>",
        reply_markup=get_tg_keyboard(),
        parse_mode="HTML",
    )

@tg_router.callback_query(F.data.startswith("tg:"))
async def handle_tg_callback(callback: CallbackQuery):
    import main
    db = main.get_db()
    user_id = callback.from_user.id
    
    # Check admin again for safety
    if not main.has_admin_access(await _get_user(db, user_id)):
        await callback.answer("⛔ Access denied", show_alert=True)
        return

    parts = callback.data.split(":")
    action = parts[1] if len(parts) > 1 else ""

    def _parse_int(idx: int, default: int) -> int:
        if len(parts) <= idx:
            return default
        try:
            return int(parts[idx])
        except Exception:
            return default
    
    if action == "run":
        await callback.message.answer("🚀 Starting Telegram Monitor...")
        await callback.answer()
        # Run async
        asyncio.create_task(
            run_monitor_task(
                callback.bot,
                callback.message.chat.id,
                operator_id=callback.from_user.id,
            )
        )
    elif action == "rerun_import":
        # Show the most recent local results so operator can re-import any of them
        # without waiting for Kaggle.
        paths = find_recent_telegram_results_json(limit=4)
        if not paths:
            await callback.message.answer(
                (
                    "❌ Локальный telegram_results.json не найден.\n"
                    "Сначала выполните хотя бы один запуск мониторинга через Kaggle."
                )
            )
            await callback.answer()
            return

        recent_import_results[user_id] = [str(p) for p in paths]
        selected_import_result.pop(user_id, None)

        tmp_dir = Path(tempfile.gettempdir())
        buttons: list[list[InlineKeyboardButton]] = []
        for idx, p in enumerate(paths):
            label_tail = p.parent.name
            try:
                stat = p.stat()
                dt = datetime.fromtimestamp(float(stat.st_mtime))
                ts_txt = dt.strftime("%Y-%m-%d %H:%M")
                label = f"{idx + 1}. {ts_txt} — {label_tail}"
            except Exception:
                label = f"{idx + 1}. {label_tail}"
            # Try to include a short relative path as hint (best-effort).
            try:
                rel = str(p.relative_to(tmp_dir))
                if rel and len(rel) <= 42:
                    label = f"{label} ({rel})"
            except Exception:
                pass
            buttons.append([InlineKeyboardButton(text=label, callback_data=f"tg:rerun_import_pick:{idx}")])
        buttons.append([InlineKeyboardButton(text="❌ Отмена", callback_data="tg:rerun_import_cancel")])

        await callback.message.answer(
            "♻️ Выберите локальный результат для импорта (последние 4):",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
            disable_web_page_preview=True,
        )
        await callback.answer()
    elif action == "rerun_import_cancel":
        recent_import_results.pop(user_id, None)
        selected_import_result.pop(user_id, None)
        await callback.answer("Ок")
    elif action == "rerun_import_pick" and len(parts) >= 3:
        idx = _parse_int(2, -1)
        chosen: Path | None = None
        cached = recent_import_results.get(user_id) or []
        if 0 <= idx < len(cached):
            chosen = Path(cached[idx])
        if not chosen or not chosen.exists():
            # Cache might be stale; try to recompute and pick again.
            paths = find_recent_telegram_results_json(limit=4)
            if 0 <= idx < len(paths):
                chosen = paths[idx]
        if not chosen or not chosen.exists():
            await callback.answer("Список устарел. Нажмите «♻️ Импорт из JSON» ещё раз.", show_alert=True)
            return

        recent_import_results.pop(user_id, None)
        selected_import_result[user_id] = str(chosen)

        mode_buttons: list[list[InlineKeyboardButton]] = [
            [InlineKeyboardButton(text="Импорт (обычно)", callback_data="tg:rerun_import_mode:normal")]
        ]
        if _is_dev_mode_enabled():
            mode_buttons.append(
                [
                    InlineKeyboardButton(
                        text="DEV: Recreate + Reimport",
                        callback_data="tg:rerun_import_mode:dev_recreate",
                    )
                ]
            )
        mode_buttons.append([InlineKeyboardButton(text="❌ Отмена", callback_data="tg:rerun_import_cancel")])

        await callback.message.answer(
            (
                "♻️ Выбран локальный результат:\n"
                f"<code>{html.escape(str(chosen))}</code>\n\n"
                "Выберите режим запуска:"
            ),
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=mode_buttons),
            disable_web_page_preview=True,
        )
        await callback.answer()
    elif action == "rerun_import_mode" and len(parts) >= 3:
        mode = (parts[2] or "").strip().lower()
        chosen_raw = selected_import_result.get(user_id)
        chosen = Path(chosen_raw) if chosen_raw else None
        if not chosen or not chosen.exists():
            selected_import_result.pop(user_id, None)
            await callback.answer("Список устарел. Нажмите «♻️ Импорт из JSON» ещё раз.", show_alert=True)
            return

        if mode == "normal":
            selected_import_result.pop(user_id, None)
            await callback.message.answer(
                f"♻️ Запускаю импорт из локального результата:\n{chosen}",
                disable_web_page_preview=True,
            )
            await callback.answer()
            asyncio.create_task(
                rerun_import_task(
                    callback.bot,
                    callback.message.chat.id,
                    results_path=chosen,
                    operator_id=callback.from_user.id,
                )
            )
            return

        if mode == "dev_recreate":
            if not _is_dev_mode_enabled():
                await callback.answer("DEV-режим недоступен (DEV_MODE!=1).", show_alert=True)
                return
            try:
                preview = await preview_telegram_recreate_reimport(
                    db,
                    results_path=chosen,
                )
            except Exception as exc:
                logger.exception("tg dev recreate preview failed")
                await callback.message.answer(f"❌ Не удалось подготовить DEV preview: {exc}")
                await callback.answer()
                return
            confirm_buttons = [
                [
                    InlineKeyboardButton(
                        text="✅ Подтвердить DEV Recreate + Reimport",
                        callback_data="tg:rerun_import_dev_confirm",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="Импорт (обычно)",
                        callback_data="tg:rerun_import_mode:normal",
                    )
                ],
                [InlineKeyboardButton(text="❌ Отмена", callback_data="tg:rerun_import_cancel")],
            ]
            await callback.message.answer(
                (
                    "⚠️ <b>DEV: Recreate + Reimport</b>\n"
                    f"Файл: <code>{html.escape(str(chosen))}</code>\n\n"
                    f"Будет удалено событий: <b>{int(preview.event_ids_found)}</b>\n"
                    f"Будет очищено marks already_scanned: <b>{int(preview.scanned_matches_found)}</b>\n"
                    f"Ссылок source_url: {int(preview.source_links_total)}\n"
                    f"Пар (source,message_id): {int(preview.message_pairs_total)}\n\n"
                    "Подтвердите запуск."
                ),
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=confirm_buttons),
                disable_web_page_preview=True,
            )
            await callback.answer()
            return

        await callback.answer("Неизвестный режим импорта.", show_alert=True)
    elif action == "rerun_import_dev_confirm":
        if not _is_dev_mode_enabled():
            await callback.answer("DEV-режим недоступен (DEV_MODE!=1).", show_alert=True)
            return
        chosen_raw = selected_import_result.pop(user_id, None)
        chosen = Path(chosen_raw) if chosen_raw else None
        if not chosen or not chosen.exists():
            await callback.answer("Список устарел. Нажмите «♻️ Импорт из JSON» ещё раз.", show_alert=True)
            return

        await callback.message.answer(
            f"🧪 DEV: запускаю Recreate + Reimport из:\n{chosen}",
            disable_web_page_preview=True,
        )
        await callback.answer()
        asyncio.create_task(
            rerun_import_recreate_task(
                callback.bot,
                callback.message.chat.id,
                results_path=chosen,
                operator_id=callback.from_user.id,
            )
        )

    elif action == "add":
        adding_source_sessions[user_id] = "add"
        await callback.message.answer(
            "✍️ Пришлите @username или ссылку вида https://t.me/username.\n"
            "Примеры:\n"
            "  @meowafisha\n"
            "  https://t.me/meowafisha\n"
            "\n"
            "Уровень доверия и параметры источника можно настроить позже через «📋 Список источников».\n"
            "Отмена: /cancel"
        )
        await callback.answer()
        
    elif action == "list":
        page = _parse_int(2, 1)
        await list_sources(db, callback.message, page=page, edit=True)
        await callback.answer()
    elif action == "seed":
        await callback.message.answer("⏳ Синхронизирую канонические источники…")
        await callback.answer()
        try:
            async with db.raw_conn() as conn:
                await seed_telegram_sources(conn)
                await conn.commit()
            await callback.message.answer("✅ Источники синхронизированы.")
        except Exception as e:
            logger.exception("tg seed failed")
            await callback.message.answer(f"❌ Ошибка seed: {e}")
        await list_sources(db, callback.message, page=1, edit=False)
    elif action == "toggle" and len(parts) >= 3:
        await toggle_source(db, callback.message, int(parts[2]), page=_parse_int(3, 1))
        await callback.answer()
    elif action == "trust" and len(parts) >= 3:
        await cycle_trust(db, callback.message, int(parts[2]), page=_parse_int(3, 1))
        await callback.answer()
    elif action == "delete" and len(parts) >= 3:
        await delete_source(db, callback.message, int(parts[2]), page=_parse_int(3, 1))
        await callback.answer()
    elif action == "reset" and len(parts) >= 3:
        await reset_source_marks(db, callback.message, int(parts[2]), page=_parse_int(3, 1))
        await callback.answer()
    elif action == "loc" and len(parts) >= 3:
        source_id = int(parts[2])
        adding_source_sessions[user_id] = f"loc:{source_id}:{_parse_int(3, 1)}"
        await callback.message.answer(
            "📍 Пришлите default_location для источника (как будет сохраняться в событиях),\n"
            "например: Научная библиотека\n"
            "Чтобы очистить — отправьте `-`.\n"
            "Отмена: /cancel",
            parse_mode="Markdown",
        )
        await callback.answer()
    elif action == "ticket" and len(parts) >= 3:
        source_id = int(parts[2])
        adding_source_sessions[user_id] = f"ticket:{source_id}:{_parse_int(3, 1)}"
        await callback.message.answer(
            "🎟 Пришлите default_ticket_link (https://...), который будет подставляться, если в посте нет ссылки.\n"
            "Чтобы очистить — отправьте `-`.\n"
            "Отмена: /cancel",
            parse_mode="Markdown",
        )
        await callback.answer()
    elif action == "festival" and len(parts) >= 3:
        source_id = int(parts[2])
        adding_source_sessions[user_id] = f"festival:{source_id}:{_parse_int(3, 1)}"
        await callback.message.answer(
            "🎪 Пришлите название серии фестиваля, если канал относится к фестивалю.\n"
            "Например: Кантата, Открытое море, Гаражка.\n"
            "Чтобы очистить — отправьте `-`.\n"
            "Отмена: /cancel",
            parse_mode="Markdown",
        )
        await callback.answer()
    elif action == "accept_suggest" and len(parts) >= 3:
        await accept_source_festival_suggestion(
            db,
            callback.message,
            int(parts[2]),
            page=_parse_int(3, 1),
        )
        await callback.answer()

@tg_router.message(lambda m: m.from_user.id in adding_source_sessions)
async def handle_source_input(message: types.Message):
    """Handle source username input."""
    import main
    db = main.get_db()
    
    user_id = message.from_user.id
    text = (message.text or "").strip()
    
    if text.lower() == "/cancel":
        adding_source_sessions.pop(user_id, None)
        await message.answer("❌ Cancelled.")
        return

    try:
        mode = adding_source_sessions.get(user_id) or "add"
        if mode == "add":
            username, options = parse_source_input(text)
            await add_source(db, message, username, options)
            adding_source_sessions.pop(user_id, None)
            return
        if mode.startswith("loc:"):
            _, source_id, page = (mode.split(":", 2) + ["1"])[:3]
            await set_source_location(db, message, int(source_id), text, page=int(page or 1))
            adding_source_sessions.pop(user_id, None)
            return
        if mode.startswith("ticket:"):
            _, source_id, page = (mode.split(":", 2) + ["1"])[:3]
            await set_source_ticket(db, message, int(source_id), text, page=int(page or 1))
            adding_source_sessions.pop(user_id, None)
            return
        if mode.startswith("festival:"):
            _, source_id, page = (mode.split(":", 2) + ["1"])[:3]
            await set_source_festival(db, message, int(source_id), text, page=int(page or 1))
            adding_source_sessions.pop(user_id, None)
            return
        raise ValueError("unknown session mode")
    except Exception as e:
        logger.error(f"Error adding source: {e}")
        await message.answer(f"❌ Error: {e}")
        adding_source_sessions.pop(user_id, None)

async def run_monitor_task(bot: Bot, chat_id: int, *, operator_id: int | None = None):
    import main
    db = main.get_db()
    
    try:
        if _monitor_lock.locked():
            await bot.send_message(chat_id, "⏳ Мониторинг уже запущен, ждём завершения.")
            return
        async with _monitor_lock:
            # Prevent confusing parallel "heavy" operations in operator chat:
            # VK auto import and Telegram monitoring share one global semaphore.
            async with heavy_operation(
                kind="tg_monitoring",
                trigger="manual",
                operator_id=operator_id,
                chat_id=chat_id,
            ):
                await run_telegram_monitor(
                    db,
                    bot=bot,
                    chat_id=chat_id,
                    send_progress=True,
                    trigger="manual",
                    operator_id=operator_id,
                )
    except Exception as e:
        logger.exception("Manual monitor run failed")
        msg = str(e or "").strip()
        if "database is locked" in msg.lower():
            await bot.send_message(
                chat_id,
                (
                    "❌ Monitor run failed: database is locked\n"
                    "Импорт мог выполниться частично до точки сбоя. "
                    "Запустите /tg ещё раз: обработанные посты будут пропущены по idempotency."
                ),
            )
        elif "authkeyduplicatederror" in msg.lower():
            await bot.send_message(
                chat_id,
                (
                    "❌ Monitor run failed: AuthKeyDuplicatedError\n"
                    "Telegram session использовалась параллельно с разных IP и стала недействительной. "
                    "Перевыпустите отдельную session для этого окружения и повторите /tg."
                ),
            )
        else:
            await bot.send_message(chat_id, f"❌ Monitor run failed: {e}")


async def rerun_last_import_task(bot: Bot, chat_id: int, *, operator_id: int | None = None):
    # Backwards-compat stub kept for callers that still assume "latest-only".
    return await rerun_import_task(bot, chat_id, operator_id=operator_id)


async def rerun_import_task(
    bot: Bot,
    chat_id: int,
    *,
    results_path: Path | None = None,
    operator_id: int | None = None,
):
    import main
    db = main.get_db()

    try:
        if _monitor_lock.locked():
            await bot.send_message(chat_id, "⏳ Мониторинг/импорт уже запущен, ждём завершения.")
            return
        async with _monitor_lock:
            async with heavy_operation(
                kind="tg_monitoring",
                trigger="manual_import_only",
                operator_id=operator_id,
                chat_id=chat_id,
            ):
                if results_path is None:
                    results_path = find_latest_telegram_results_json()
                    await bot.send_message(
                        chat_id,
                        f"📂 Нашёл локальный результат: {results_path}",
                        disable_web_page_preview=True,
                    )
                await run_telegram_import_from_results(
                    db,
                    results_path=results_path,
                    bot=bot,
                    chat_id=chat_id,
                    send_progress=True,
                    trigger="manual_import_only",
                    operator_id=operator_id,
                )
    except FileNotFoundError:
        await bot.send_message(
            chat_id,
            (
                "❌ Локальный telegram_results.json не найден.\n"
                "Сначала выполните хотя бы один запуск мониторинга через Kaggle."
            ),
        )
    except Exception as e:
        logger.exception("Manual tg import rerun failed")
        msg = str(e or "").strip()
        if "database is locked" in msg.lower():
            await bot.send_message(
                chat_id,
                (
                    "❌ Import rerun failed: database is locked\n"
                    "Импорт мог выполниться частично до точки сбоя. "
                    "Повторите кнопку: обработанные посты будут пропущены по idempotency."
                ),
            )
        else:
            await bot.send_message(chat_id, f"❌ Import rerun failed: {e}")


async def rerun_import_recreate_task(
    bot: Bot,
    chat_id: int,
    *,
    results_path: Path | None = None,
    operator_id: int | None = None,
):
    import main
    db = main.get_db()

    if not _is_dev_mode_enabled():
        await bot.send_message(chat_id, "⛔ DEV Recreate + Reimport доступен только при DEV_MODE=1.")
        return

    try:
        if _monitor_lock.locked():
            await bot.send_message(chat_id, "⏳ Мониторинг/импорт уже запущен, ждём завершения.")
            return
        async with _monitor_lock:
            async with heavy_operation(
                kind="tg_monitoring",
                trigger="manual_import_dev_recreate",
                operator_id=operator_id,
                chat_id=chat_id,
            ):
                if results_path is None:
                    results_path = find_latest_telegram_results_json()
                    await bot.send_message(
                        chat_id,
                        f"📂 Нашёл локальный результат: {results_path}",
                        disable_web_page_preview=True,
                    )
                await run_telegram_dev_recreate_reimport(
                    db,
                    results_path=results_path,
                    bot=bot,
                    chat_id=chat_id,
                    send_progress=True,
                    trigger="manual_import_dev_recreate",
                    operator_id=operator_id,
                )
    except FileNotFoundError:
        await bot.send_message(
            chat_id,
            (
                "❌ Локальный telegram_results.json не найден.\n"
                "Сначала выполните хотя бы один запуск мониторинга через Kaggle."
            ),
        )
    except Exception as e:
        logger.exception("Manual tg dev recreate rerun failed")
        await bot.send_message(chat_id, f"❌ DEV Recreate + Reimport failed: {e}")

def normalize_source(raw: str) -> str:
    return normalize_tg_username(raw)


def parse_source_input(text: str) -> tuple[str, dict[str, str]]:
    parts = shlex.split(text)
    if not parts:
        raise ValueError("empty input")
    username = normalize_source(parts[0])
    if not username:
        raise ValueError("invalid source")
    options: dict[str, str] = {}
    for token in parts[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if not value:
            continue
        options[key] = value
    return username, options


async def add_source(
    db: Database,
    message: types.Message,
    username: str,
    options: dict[str, str] | None = None,
) -> None:
    options = options or {}
    trust = options.get("trust")
    if trust and trust.lower() not in {"high", "medium", "low"}:
        await message.answer("⚠️ trust должен быть high|medium|low")
        trust = None
    default_location = options.get("location")
    default_ticket = options.get("ticket")
    festival_series = options.get("festival") or options.get("series")

    async with db.get_session() as session:
        result = await session.execute(
            select(TelegramSource).where(TelegramSource.username == username)
        )
        source = result.scalar_one_or_none()
        if source:
            source.enabled = True
            if trust:
                source.trust_level = trust.lower()
            if default_location:
                source.default_location = default_location
            if default_ticket:
                source.default_ticket_link = default_ticket
            if festival_series is not None:
                clean_series = festival_series.strip()
                source.festival_series = clean_series or None
                source.festival_source = bool(clean_series)
            await session.commit()
            await message.answer(f"✅ Источник @{username} обновлён и включён.")
            return
        try:
            source = TelegramSource(
                username=username,
                enabled=True,
                trust_level=trust.lower() if trust else None,
                default_location=default_location,
                default_ticket_link=default_ticket,
                festival_series=(festival_series.strip() if festival_series else None),
                festival_source=bool(festival_series and festival_series.strip()),
            )
            session.add(source)
            await session.commit()
            await message.answer(f"✅ Источник @{username} добавлен.")
        except IntegrityError:
            await session.rollback()
            await message.answer("❌ Ошибка добавления источника.")


async def list_sources(db: Database, message: types.Message, *, page: int = 1, edit: bool = False):
    page = max(1, int(page or 1))
    async with db.get_session() as session:
        total = await session.scalar(select(func.count()).select_from(TelegramSource))
        total = int(total or 0)
        pages = max(1, int(math.ceil(total / SOURCES_PAGE_SIZE)) if total else 1)
        page = max(1, min(page, pages))
        offset = (page - 1) * SOURCES_PAGE_SIZE

        result = await session.execute(
            select(TelegramSource)
            .order_by(TelegramSource.username.asc())
            .offset(offset)
            .limit(SOURCES_PAGE_SIZE)
        )
        sources = result.scalars().all()

        source_ids = [s.id for s in sources if s.id is not None]
        stats_map = {}
        if source_ids:
            stats_rows = await session.execute(
                select(
                    TelegramScannedMessage.source_id,
                    func.count(TelegramScannedMessage.message_id),
                    func.sum(TelegramScannedMessage.events_extracted),
                    func.sum(TelegramScannedMessage.events_imported),
                    func.max(TelegramScannedMessage.processed_at),
                )
                .where(TelegramScannedMessage.source_id.in_(source_ids))
                .group_by(TelegramScannedMessage.source_id)
            )
            stats_map = {row[0]: row for row in stats_rows}
    if not sources:
        await message.answer("Источники не настроены.")
        return
    lines = [f"<b>Telegram Sources</b> — page {page}/{pages} (total={total})"]
    keyboard = []

    def _fmt_dt(value: datetime | None) -> str:
        if not value:
            return "—"
        return value.strftime("%Y-%m-%d %H:%M:%S")

    def _fmt_confidence(value: float | None) -> str:
        if value is None:
            return "—"
        try:
            val = float(value)
        except Exception:
            return "—"
        val = max(0.0, min(1.0, val))
        return f"{val:.2f}"

    def _fmt_int(value: int | None) -> str:
        if not isinstance(value, int) or value < 0:
            return "—"
        return str(int(value))

    for src in sources:
        status = "✅" if src.enabled else "⛔"
        trust = src.trust_level or "low"
        title = (src.title or "").strip()
        if title:
            lines.append(
                f"{status} <b>{html.escape(title)}</b> (@{html.escape(src.username)}; trust={html.escape(trust)})"
            )
        else:
            lines.append(f"{status} @{html.escape(src.username)} (trust={html.escape(trust)})")
        stats = stats_map.get(src.id)
        if stats:
            _source_id, scanned_total, extracted_total, imported_total, last_processed = stats
            scanned_total = int(scanned_total or 0)
            extracted_total = int(extracted_total or 0)
            imported_total = int(imported_total or 0)
            lines.append(
                "  ↳ "
                f"last_scan: {_fmt_dt(src.last_scan_at)}; "
                f"last_msg_id: {src.last_scanned_message_id or '—'}; "
                f"scanned: {scanned_total}; "
                f"events: {imported_total}/{extracted_total}; "
                f"last_processed: {_fmt_dt(last_processed)}"
            )
        else:
            lines.append(
                "  ↳ "
                f"last_scan: {_fmt_dt(src.last_scan_at)}; "
                f"last_msg_id: {src.last_scanned_message_id or '—'}; "
                "scanned: 0; events: 0/0"
            )

        try:
            overview = await load_telegram_popularity_overview(
                db,
                source_id=int(src.id),
                age_day=0,
            )
            base = overview.baseline
            lines.append(
                "  ↳ "
                f"median_{int(overview.horizon_days)}d: "
                f"views={_fmt_int(base.median_views)} "
                f"likes={_fmt_int(base.median_likes)}; "
                f"days={int(overview.days_covered)}/{int(overview.horizon_days)}; "
                f"posts={int(base.sample)}"
                + ("; fallback=1" if overview.used_fallback else "")
            )
        except Exception:
            logger.debug(
                "tg.list_sources popularity overview failed source_id=%s",
                src.id,
                exc_info=True,
            )

        fest_label = (src.festival_series or "—").strip()
        if len(fest_label) > 36:
            fest_label = fest_label[:35] + "…"
        lines.append(f"  ↳ festival: {fest_label}")
        suggested_series = (src.suggested_festival_series or "").strip()
        if suggested_series and not (src.festival_series or "").strip():
            if len(suggested_series) > 48:
                suggested_series = suggested_series[:47] + "…"
            lines.append(
                f"  ↳ suggested: {html.escape(suggested_series)} (confidence {html.escape(_fmt_confidence(src.suggestion_confidence))})"
            )
            suggested_site = (src.suggested_website_url or "").strip()
            if suggested_site:
                lines.append(f"  ↳ site: {html.escape(suggested_site)}")
        keyboard.append(
            [
                InlineKeyboardButton(
                    text=f"{'Disable' if src.enabled else 'Enable'} @{src.username}",
                    callback_data=f"tg:toggle:{src.id}:{page}",
                ),
                InlineKeyboardButton(
                    text=f"Trust → {trust}",
                    callback_data=f"tg:trust:{src.id}:{page}",
                ),
            ]
        )
        loc_label = (src.default_location or "—").strip()
        if len(loc_label) > 24:
            loc_label = loc_label[:23] + "…"
        ticket_label = (src.default_ticket_link or "—").strip()
        if len(ticket_label) > 24:
            ticket_label = ticket_label[:23] + "…"
        keyboard.append(
            [
                InlineKeyboardButton(
                    text=f"📍 Локация → {loc_label}",
                    callback_data=f"tg:loc:{src.id}:{page}",
                ),
                InlineKeyboardButton(
                    text=f"🎟 Ticket → {ticket_label}",
                    callback_data=f"tg:ticket:{src.id}:{page}",
                ),
            ]
        )
        keyboard.append(
            [
                InlineKeyboardButton(
                    text=f"🎪 Фестиваль → {fest_label}",
                    callback_data=f"tg:festival:{src.id}:{page}",
                )
            ]
        )
        if (src.suggested_festival_series or "").strip() and not (src.festival_series or "").strip():
            keyboard.append(
                [
                    InlineKeyboardButton(
                        text="✅ Принять подсказку",
                        callback_data=f"tg:accept_suggest:{src.id}:{page}",
                    )
                ]
            )
        suggested_site = (src.suggested_website_url or "").strip()
        if suggested_site.lower().startswith(("http://", "https://")):
            keyboard.append(
                [
                    InlineKeyboardButton(
                        text="🌐 Suggested website",
                        url=suggested_site,
                    )
                ]
            )
        keyboard.append(
            [
                InlineKeyboardButton(
                    text=f"♻️ Сбросить отметки @{src.username}",
                    callback_data=f"tg:reset:{src.id}:{page}",
                )
            ]
        )
        keyboard.append(
            [
                InlineKeyboardButton(
                    text=f"🗑️ Удалить @{src.username}",
                    callback_data=f"tg:delete:{src.id}:{page}",
                )
            ]
        )

    if pages > 1:
        nav_row = []
        if page > 1:
            nav_row.append(
                InlineKeyboardButton(text="⬅️ Назад", callback_data=f"tg:list:{page-1}")
            )
        if page < pages:
            nav_row.append(
                InlineKeyboardButton(text="➡️ Далее", callback_data=f"tg:list:{page+1}")
            )
        if nav_row:
            keyboard.append(nav_row)

    text = "\n".join(lines)
    markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
    if edit:
        try:
            await message.edit_text(text, reply_markup=markup, parse_mode="HTML")
            return
        except Exception:
            pass
    await message.answer(text, reply_markup=markup, parse_mode="HTML")


async def toggle_source(db: Database, message: types.Message, source_id: int, *, page: int = 1) -> None:
    async with db.get_session() as session:
        src = await session.get(TelegramSource, source_id)
        if not src:
            await message.answer("Источник не найден")
            return
        src.enabled = not src.enabled
        await session.commit()
    await list_sources(db, message, page=page)


async def cycle_trust(db: Database, message: types.Message, source_id: int, *, page: int = 1) -> None:
    order = ["low", "medium", "high"]
    async with db.get_session() as session:
        src = await session.get(TelegramSource, source_id)
        if not src:
            await message.answer("Источник не найден")
            return
        current = (src.trust_level or "low").lower()
        try:
            idx = order.index(current)
        except ValueError:
            idx = 0
        src.trust_level = order[(idx + 1) % len(order)]
        await session.commit()
    await list_sources(db, message, page=page)

async def delete_source(db: Database, message: types.Message, source_id: int, *, page: int = 1) -> None:
    async with db.get_session() as session:
        src = await session.get(TelegramSource, source_id)
        if not src:
            await message.answer("Источник не найден")
            return
        username = src.username
        await session.execute(
            delete(TelegramScannedMessage).where(
                TelegramScannedMessage.source_id == source_id
            )
        )
        await session.delete(src)
        await session.commit()
    await message.answer(f"🗑️ Источник @{username} удалён.")
    await list_sources(db, message, page=page)

async def set_source_location(db: Database, message: types.Message, source_id: int, value: str, *, page: int = 1) -> None:
    raw = (value or "").strip()
    new_val = None if raw in {"", "-"} else raw
    async with db.get_session() as session:
        src = await session.get(TelegramSource, source_id)
        if not src:
            await message.answer("Источник не найден")
            return
        src.default_location = new_val
        await session.commit()
    await message.answer(f"✅ default_location обновлён для @{src.username}: {new_val or '—'}")
    await list_sources(db, message, page=page)

async def set_source_ticket(db: Database, message: types.Message, source_id: int, value: str, *, page: int = 1) -> None:
    raw = (value or "").strip()
    new_val = None if raw in {"", "-"} else raw
    if new_val and not new_val.lower().startswith(("http://", "https://")):
        await message.answer("⚠️ Ticket link должен быть http(s) URL или '-' для очистки.")
        return
    async with db.get_session() as session:
        src = await session.get(TelegramSource, source_id)
        if not src:
            await message.answer("Источник не найден")
            return
        src.default_ticket_link = new_val
        await session.commit()
    await message.answer(f"✅ default_ticket_link обновлён для @{src.username}: {new_val or '—'}")
    await list_sources(db, message, page=page)

async def set_source_festival(db: Database, message: types.Message, source_id: int, value: str, *, page: int = 1) -> None:
    raw = (value or "").strip()
    new_val = None if raw in {"", "-"} else raw
    async with db.get_session() as session:
        src = await session.get(TelegramSource, source_id)
        if not src:
            await message.answer("Источник не найден")
            return
        src.festival_series = new_val
        src.festival_source = bool(new_val)
        await session.commit()
    await message.answer(f"✅ festival_series обновлён для @{src.username}: {new_val or '—'}")
    await list_sources(db, message, page=page)


async def accept_source_festival_suggestion(
    db: Database,
    message: types.Message,
    source_id: int,
    *,
    page: int = 1,
) -> None:
    async with db.get_session() as session:
        src = await session.get(TelegramSource, source_id)
        if not src:
            await message.answer("Источник не найден")
            return
        manual_series = (src.festival_series or "").strip()
        suggested_series = (src.suggested_festival_series or "").strip()
        if manual_series:
            await message.answer(f"ℹ️ Для @{src.username} festival_series уже задан вручную.")
            return
        if not suggested_series:
            await message.answer(f"ℹ️ Для @{src.username} нет подсказки festival_series.")
            return
        src.festival_series = suggested_series
        src.festival_source = True
        await session.commit()
    await message.answer(f"✅ Принята подсказка для @{src.username}: {suggested_series}")
    await list_sources(db, message, page=page)

async def reset_source_marks(db: Database, message: types.Message, source_id: int, *, page: int = 1) -> None:
    """Reset last scanned message id and scanned marks for a Telegram source (operator UI)."""
    async with db.get_session() as session:
        src = await session.get(TelegramSource, source_id)
        if not src:
            await message.answer("Источник не найден")
            return
        await session.execute(
            delete(TelegramScannedMessage).where(
                TelegramScannedMessage.source_id == source_id
            )
        )
        src.last_scanned_message_id = None
        src.last_scan_at = None
        await session.commit()
    await message.answer(f"♻️ Отметки мониторинга для @{src.username} сброшены.")
    await list_sources(db, message, page=page)

async def _get_user(db: Database, user_id: int):
    from models import User
    async with db.get_session() as session:
        return await session.get(User, user_id)
