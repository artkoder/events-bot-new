from __future__ import annotations

import html
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from sqlalchemy import or_, select

from runtime import require_main_attr

if TYPE_CHECKING:
    from db import Database

logger = logging.getLogger(__name__)

recent_imports_router = Router(name="recent_imports")

_MAX_TG_MESSAGE_LEN = 3800
_SOURCE_LABEL_ORDER = {
    "Telegram": 0,
    "VK": 1,
    "/parse": 2,
}


@dataclass
class _RecentImportItem:
    event_id: int
    title: str
    telegraph_url: str | None
    event_date: str
    event_time: str
    added_at: datetime | None
    latest_imported_at: datetime | None
    is_created: bool
    source_labels: set[str] = field(default_factory=set)


def _chunk_lines(lines: list[str], *, max_len: int = _MAX_TG_MESSAGE_LEN) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        line = str(line or "")
        line_len = len(line) + 1
        if current and current_len + line_len > max_len:
            chunks.append("\n".join(current).strip())
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _coerce_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_hours_arg(text: str | None) -> int:
    parts = (text or "").strip().split(maxsplit=1)
    if len(parts) <= 1:
        return 24
    raw = parts[1].strip()
    if not raw.isdigit():
        raise ValueError("usage")
    hours = int(raw)
    if not (1 <= hours <= 720):
        raise ValueError("range")
    return hours


def _telegraph_url_for_event(event: Any) -> str | None:
    raw_url = str(getattr(event, "telegraph_url", "") or "").strip()
    if raw_url.startswith(("http://", "https://")):
        return raw_url
    raw_path = str(getattr(event, "telegraph_path", "") or "").strip().lstrip("/")
    if raw_path:
        return f"https://telegra.ph/{raw_path}"
    return None


def _source_label(source_type: str | None) -> str | None:
    cleaned = str(source_type or "").strip().lower()
    if not cleaned:
        return None
    if cleaned == "telegram":
        return "Telegram"
    if cleaned.startswith("vk"):
        return "VK"
    if cleaned.startswith("parser:"):
        return "/parse"
    return None


def _format_window_stamp(value: datetime | None, tz: Any) -> str:
    dt = _coerce_utc(value)
    if dt is None:
        return ""
    return dt.astimezone(tz).strftime("%d.%m %H:%M")


def _format_event_when(date_raw: str | None, time_raw: str | None) -> str:
    raw_date = str(date_raw or "").strip()
    if not raw_date:
        return ""
    display_date = raw_date
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            display_date = datetime.strptime(raw_date, fmt).strftime("%d.%m.%Y")
            break
        except Exception:
            continue
    raw_time = str(time_raw or "").strip()
    return f"{display_date} {raw_time}".strip()


async def _load_recent_import_items(
    db: Database,
    *,
    hours: int,
    now_utc: datetime | None = None,
) -> tuple[list[_RecentImportItem], datetime, datetime]:
    end_utc = _coerce_utc(now_utc) or datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(hours=int(hours))

    from models import Event, EventSource

    async with db.get_session() as session:
        rows = (
            await session.execute(
                select(Event, EventSource)
                .join(EventSource, EventSource.event_id == Event.id)
                .where(EventSource.imported_at >= start_utc)
                .where(EventSource.imported_at < end_utc)
                .where(
                    or_(
                        EventSource.source_type == "telegram",
                        EventSource.source_type.like("vk%"),
                        EventSource.source_type.like("parser:%"),
                    )
                )
                .order_by(EventSource.imported_at.desc(), EventSource.id.desc())
            )
        ).all()

    items_by_event: dict[int, _RecentImportItem] = {}
    min_dt = datetime.min.replace(tzinfo=timezone.utc)
    for event, source in rows:
        label = _source_label(getattr(source, "source_type", None))
        if label is None:
            continue
        event_id = int(getattr(event, "id", 0) or 0)
        if event_id <= 0:
            continue
        imported_at = _coerce_utc(getattr(source, "imported_at", None))
        added_at = _coerce_utc(getattr(event, "added_at", None))
        item = items_by_event.get(event_id)
        if item is None:
            item = _RecentImportItem(
                event_id=event_id,
                title=str(getattr(event, "title", "") or "").strip() or "событие",
                telegraph_url=_telegraph_url_for_event(event),
                event_date=str(getattr(event, "date", "") or "").strip(),
                event_time=str(getattr(event, "time", "") or "").strip(),
                added_at=added_at,
                latest_imported_at=imported_at,
                is_created=bool(added_at and start_utc <= added_at < end_utc),
            )
            items_by_event[event_id] = item
        if imported_at and (item.latest_imported_at or min_dt) < imported_at:
            item.latest_imported_at = imported_at
        if added_at and start_utc <= added_at < end_utc:
            item.is_created = True
        item.source_labels.add(label)

    items = sorted(
        items_by_event.values(),
        key=lambda item: (item.latest_imported_at or min_dt, item.event_id),
        reverse=True,
    )
    return items, start_utc, end_utc


def _render_recent_imports_lines(
    items: list[_RecentImportItem],
    *,
    hours: int,
    start_utc: datetime,
    end_utc: datetime,
    tz: Any,
) -> list[str]:
    created = sum(1 for item in items if item.is_created)
    updated = max(0, len(items) - created)
    lines = [
        "📥 <b>Недавние импорты событий</b>",
        (
            f"Окно: {_format_window_stamp(start_utc, tz)} - {_format_window_stamp(end_utc, tz)} "
            f"({int(hours)} ч, {html.escape(str(tz))})."
        ),
        "Источники: Telegram, VK, /parse.",
        f"Событий: {len(items)}. Статусы: ✅ создано={created}, 🔄 обновлено={updated}.",
    ]
    if not items:
        lines.append("")
        lines.append(
            f"Нет событий, созданных или обновлённых из Telegram/VK//parse за последние {int(hours)} ч."
        )
        return lines

    lines.append("")
    for item in items:
        title_html = html.escape(item.title)
        if item.telegraph_url:
            title_html = (
                f'<a href="{html.escape(item.telegraph_url, quote=True)}">{title_html}</a>'
            )
        meta: list[str] = []
        event_when = _format_event_when(item.event_date, item.event_time)
        if event_when:
            meta.append(event_when)
        if item.source_labels:
            source_text = ", ".join(
                sorted(
                    item.source_labels,
                    key=lambda label: (_SOURCE_LABEL_ORDER.get(label, 99), label),
                )
            )
            meta.append(source_text)
        imported_at = _format_window_stamp(item.latest_imported_at, tz)
        if imported_at:
            meta.append(f"импорт {imported_at}")
        row = f"{item.event_id} {'✅' if item.is_created else '🔄'} {title_html}"
        if meta:
            row += " | " + " | ".join(html.escape(part) for part in meta)
        lines.append(row)
    return lines


async def _require_superadmin(db: Database, user_id: int) -> bool:
    from models import User

    async with db.get_session() as session:
        user = await session.get(User, user_id)
    return bool(require_main_attr("has_admin_access")(user))


async def _send_recent_imports_report(
    message: Message,
    db: Database,
    *,
    hours: int,
) -> None:
    tz = require_main_attr("LOCAL_TZ")
    items, start_utc, end_utc = await _load_recent_import_items(db, hours=hours)
    lines = _render_recent_imports_lines(
        items,
        hours=hours,
        start_utc=start_utc,
        end_utc=end_utc,
        tz=tz,
    )
    for chunk in _chunk_lines(lines):
        await message.answer(
            chunk,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )


@recent_imports_router.message(Command("recent_imports"))
async def cmd_recent_imports(message: Message) -> None:
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
    try:
        hours = _parse_hours_arg(message.text)
    except ValueError as exc:
        if str(exc) == "range":
            await message.answer("❌ Укажите окно от 1 до 720 часов. Пример: /recent_imports 48")
        else:
            await message.answer("❌ Использование: /recent_imports [hours]")
        return

    try:
        await _send_recent_imports_report(message, db, hours=hours)
    except Exception:
        logger.exception("recent_imports: failed")
        await message.answer("❌ Не удалось построить отчёт. Проверьте логи.")
