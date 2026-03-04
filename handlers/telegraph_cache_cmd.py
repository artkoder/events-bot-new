from __future__ import annotations

import logging
from typing import Any

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from runtime import require_main_attr
from telegraph_cache_sanitizer import format_probe_stats_text, run_telegraph_cache_sanitizer

logger = logging.getLogger(__name__)

telegraph_cache_router = Router(name="telegraph_cache")


async def _require_superadmin(db, user_id: int) -> bool:  # noqa: ANN001 - db is runtime-injected
    from models import User

    async with db.get_session() as session:
        user = await session.get(User, int(user_id))
        return bool(user and not user.blocked and user.is_superadmin)


def _parse_args(text: str) -> dict[str, Any]:
    parts = (text or "").strip().split()
    args = parts[1:]
    out: dict[str, Any] = {}
    for p in args:
        p = str(p or "").strip()
        if not p:
            continue
        if p in {"--no-enqueue", "--no_queue", "--no-queue"}:
            out["enqueue_regen"] = False
            continue
        if p in {"--enqueue", "--queue"}:
            out["enqueue_regen"] = True
            continue
        if p in {"--no-months"}:
            out["include_month_pages"] = False
            continue
        if p in {"--no-weekends"}:
            out["include_weekend_pages"] = False
            continue
        if p in {"--no-festivals"}:
            out["include_festival_pages"] = False
            continue
        if p in {"--no-index"}:
            out["include_festivals_index"] = False
            continue
        if p.startswith("--events=") or p.startswith("--limit="):
            try:
                out["limit_events"] = int(p.split("=", 1)[1])
            except Exception:
                pass
            continue
        if p.startswith("--festivals="):
            try:
                out["limit_festivals"] = int(p.split("=", 1)[1])
            except Exception:
                pass
            continue
        if p.startswith("--back=") or p.startswith("--days-back="):
            try:
                out["days_back"] = int(p.split("=", 1)[1])
            except Exception:
                pass
            continue
        if p.startswith("--forward=") or p.startswith("--days-forward="):
            try:
                out["days_forward"] = int(p.split("=", 1)[1])
            except Exception:
                pass
            continue
        if p.startswith("--timeout-min="):
            try:
                out["kaggle_timeout_minutes"] = int(p.split("=", 1)[1])
            except Exception:
                pass
            continue
        if p.startswith("--regen-after="):
            try:
                out["regen_min_consecutive_failures"] = int(p.split("=", 1)[1])
            except Exception:
                pass
            continue
    return out


@telegraph_cache_router.message(Command("telegraph_cache_stats"))
async def cmd_telegraph_cache_stats(message: Message) -> None:
    get_db = require_main_attr("get_db")
    db = get_db()
    if db is None:
        await message.answer("❌ База данных ещё не инициализирована. Попробуйте позже.")
        return
    try:
        user_id = int(message.from_user.id)
    except Exception:
        await message.answer("❌ Не удалось определить пользователя.")
        return
    if not await _require_superadmin(db, user_id):
        await message.answer("❌ Команда доступна только администраторам.")
        return

    kind = None
    try:
        parts = (message.text or "").strip().split()
        if len(parts) >= 2:
            kind = parts[1].strip()
    except Exception:
        kind = None

    try:
        text = await format_probe_stats_text(db, kind=kind)
        await message.answer(text, disable_web_page_preview=True)
    except Exception:
        logger.exception("telegraph_cache_stats failed")
        await message.answer("❌ Не удалось собрать статистику. Проверьте логи.")


@telegraph_cache_router.message(Command("telegraph_cache_sanitize"))
async def cmd_telegraph_cache_sanitize(message: Message) -> None:
    get_db = require_main_attr("get_db")
    get_bot = require_main_attr("get_bot")
    db = get_db()
    bot = get_bot()
    if db is None:
        await message.answer("❌ База данных ещё не инициализирована. Попробуйте позже.")
        return
    try:
        user_id = int(message.from_user.id)
    except Exception:
        await message.answer("❌ Не удалось определить пользователя.")
        return
    if not await _require_superadmin(db, user_id):
        await message.answer("❌ Команда доступна только администраторам.")
        return

    opts = _parse_args(message.text or "")
    status_msg = await message.answer(
        "🧼 Запускаю Telegraph cache sanitizer (Kaggle/Telethon)…\n"
        "Это может занять несколько минут.",
        disable_web_page_preview=True,
    )
    try:
        res = await run_telegraph_cache_sanitizer(
            db,
            bot=bot,
            chat_id=int(message.chat.id),
            operator_id=int(user_id),
            trigger="manual",
            status_message_id=getattr(status_msg, "message_id", None),
            **opts,
        )
    except Exception as exc:
        logger.exception("telegraph_cache_sanitize failed")
        await message.answer(
            f"❌ Sanitizer завершился с ошибкой: {exc}",
            disable_web_page_preview=True,
        )
        return

    imported = res.get("imported") or {}
    regen = res.get("regen") or {}
    meta = res.get("targets_meta") or {}
    lines = [
        "✅ Telegraph cache sanitizer: готово",
        f"run_id: {res.get('run_id')}",
        (
            f"events_in_window: eligible={meta.get('eligible_event_pages', 0)} "
            f"selected={meta.get('selected_event_pages', 0)} "
            f"limit={meta.get('limit_events', 0)}"
        ),
        (
            f"targets: {imported.get('total', 0)} "
            f"cached_page_ok={imported.get('ok', 0)} "
            f"no_cached_page={imported.get('fail', 0)} "
            f"warn_no_photo={imported.get('warn_no_photo', 0)}"
        ),
    ]
    if regen:
        lines.append(
            "regen enqueued: "
            + ", ".join(f"{k}={int(v)}" for k, v in regen.items() if int(v or 0) > 0)
        )
    lines.append("Подробности: /telegraph_cache_stats")
    await message.answer("\n".join(lines), disable_web_page_preview=True)
