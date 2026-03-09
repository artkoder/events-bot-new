from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from sqlalchemy import and_, case, func, or_, select

from runtime import require_main_attr

if TYPE_CHECKING:
    from db import Database

logger = logging.getLogger(__name__)

recent_imports_router = Router(name="recent_imports")

_MAX_TG_MESSAGE_LEN = 3600
_DEFAULT_HOURS = 24
_MAX_HOURS = 168

_SOURCE_GROUP_META: tuple[tuple[str, str], ...] = (
    ("telegram", "Telegram"),
    ("vk", "VK"),
    ("parse", "/parse"),
)


@dataclass(slots=True, frozen=True)
class _RecentImportRow:
    event_id: int
    title: str
    telegraph_url: str | None
    status: str
    imported_at: datetime | None


def _chunk_lines(lines: list[str], *, max_len: int = _MAX_TG_MESSAGE_LEN) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        text = str(line or "")
        line_len = len(text) + 1
        if current and current_len + line_len > max_len:
            chunks.append("\n".join(current).strip())
            current = [text]
            current_len = line_len
        else:
            current.append(text)
            current_len += line_len
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _parse_hours_arg(text: str | None) -> int:
    raw = str(text or "").strip()
    if not raw:
        return _DEFAULT_HOURS
    parts = raw.split(maxsplit=1)
    if len(parts) < 2:
        return _DEFAULT_HOURS
    candidate = parts[1].strip()
    if not candidate:
        return _DEFAULT_HOURS
    if not candidate.isdigit():
        raise ValueError("hours must be an integer")
    hours = int(candidate)
    if hours < 1 or hours > _MAX_HOURS:
        raise ValueError(f"hours must be between 1 and {_MAX_HOURS}")
    return hours


async def _require_superadmin(db: Database, user_id: int) -> bool:
    from models import User

    async with db.get_session() as session:
        user = await session.get(User, int(user_id))
        return bool(user and not user.blocked and user.is_superadmin)


def _source_group_filter(source_key: str):
    from models import EventSource

    if source_key == "telegram":
        return or_(EventSource.source_type.like("telegram%"), EventSource.source_type == "tg")
    if source_key == "vk":
        return EventSource.source_type.like("vk%")
    if source_key == "parse":
        return EventSource.source_type.like("parser:%")
    raise ValueError(f"unknown source_key: {source_key}")


def _event_telegraph_url(*, telegraph_url: str | None, telegraph_path: str | None) -> str | None:
    url = str(telegraph_url or "").strip()
    if url.startswith(("http://", "https://")):
        return url
    path = str(telegraph_path or "").strip().lstrip("/")
    if path:
        return f"https://telegra.ph/{path}"
    return None


def _fmt_window(dt: datetime, tz) -> str:  # noqa: ANN001 - tz is runtime timezone object
    return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M")


async def _load_recent_import_rows(
    db: Database,
    *,
    source_key: str,
    start_utc: datetime,
    end_utc: datetime,
) -> list[_RecentImportRow]:
    from models import Event, EventSource

    created_expr = func.max(
        case(
            (
                and_(
                    Event.added_at >= start_utc,
                    Event.added_at < end_utc,
                ),
                1,
            ),
            else_=0,
        )
    ).label("is_created")
    last_imported_expr = func.max(EventSource.imported_at).label("last_imported_at")

    stmt = (
        select(
            Event.id,
            Event.title,
            Event.telegraph_url,
            Event.telegraph_path,
            last_imported_expr,
            created_expr,
        )
        .join(EventSource, EventSource.event_id == Event.id)
        .where(
            _source_group_filter(source_key),
            EventSource.imported_at >= start_utc,
            EventSource.imported_at < end_utc,
        )
        .group_by(
            Event.id,
            Event.title,
            Event.telegraph_url,
            Event.telegraph_path,
        )
        .order_by(last_imported_expr.desc(), Event.id.desc())
    )

    async with db.get_session() as session:
        rows = (await session.execute(stmt)).all()

    out: list[_RecentImportRow] = []
    for event_id, title, telegraph_url, telegraph_path, imported_at, is_created in rows:
        try:
            ev_id = int(event_id)
        except Exception:
            continue
        out.append(
            _RecentImportRow(
                event_id=ev_id,
                title=str(title or "").strip() or "событие",
                telegraph_url=_event_telegraph_url(
                    telegraph_url=str(telegraph_url or "").strip() or None,
                    telegraph_path=str(telegraph_path or "").strip() or None,
                ),
                status="created" if int(is_created or 0) > 0 else "updated",
                imported_at=imported_at,
            )
        )
    return out


def _render_report_pages(
    *,
    hours: int,
    start_local: datetime,
    end_local: datetime,
    tz_label: str,
    sections: dict[str, list[_RecentImportRow]],
) -> list[str]:
    body_lines: list[str] = [
        f"Окно: {_fmt_window(start_local, start_local.tzinfo)} → {_fmt_window(end_local, end_local.tzinfo)} ({html.escape(tz_label)})",
        "",
    ]

    for source_key, source_label in _SOURCE_GROUP_META:
        rows = list(sections.get(source_key) or [])
        created_count = sum(1 for row in rows if row.status == "created")
        updated_count = sum(1 for row in rows if row.status == "updated")
        body_lines.append(
            f"<b>{html.escape(source_label)}</b> — {len(rows)} "
            f"(создано {created_count}, обновлено {updated_count})"
        )
        if not rows:
            body_lines.append("—")
            body_lines.append("")
            continue
        for idx, row in enumerate(rows, start=1):
            label = "создано" if row.status == "created" else "обновлено"
            title = html.escape(row.title)
            if row.telegraph_url:
                title_part = f'<a href="{html.escape(row.telegraph_url)}">{title}</a>'
            else:
                title_part = title
            body_lines.append(f"{idx}. {title_part} (id={row.event_id}, {label})")
        body_lines.append("")

    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    header = f"🧾 <b>События из Telegram / VK / /parse за последние {hours} ч.</b>"
    body_max_len = max(120, int(_MAX_TG_MESSAGE_LEN) - 220)
    body_chunks = _chunk_lines(body_lines, max_len=body_max_len)
    if not body_chunks:
        body_chunks = ["Нет данных."]
    if len(body_chunks) == 1:
        return [f"{header}\n{body_chunks[0]}".strip()]
    pages: list[str] = []
    total = len(body_chunks)
    for idx, chunk in enumerate(body_chunks, start=1):
        pages.append(f"{header}\nСтраница {idx}/{total}\n\n{chunk}".strip())
    return pages


async def _send_recent_imports_report(message: Message, db: Database, *, hours: int) -> None:
    tz = require_main_attr("LOCAL_TZ")
    end_utc = datetime.now(timezone.utc).replace(microsecond=0)
    start_utc = end_utc - timedelta(hours=max(1, int(hours)))
    sections = {
        source_key: await _load_recent_import_rows(
            db,
            source_key=source_key,
            start_utc=start_utc,
            end_utc=end_utc,
        )
        for source_key, _source_label in _SOURCE_GROUP_META
    }
    pages = _render_report_pages(
        hours=hours,
        start_local=start_utc.astimezone(tz),
        end_local=end_utc.astimezone(tz),
        tz_label=tz.tzname(None) or "UTC",
        sections=sections,
    )
    for page in pages:
        await message.answer(page, parse_mode="HTML", disable_web_page_preview=True)


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
        await message.answer(f"❌ {exc}. Использование: /recent_imports [hours]")
        return

    try:
        await _send_recent_imports_report(message, db, hours=hours)
    except Exception:
        logger.exception("recent_imports: failed")
        await message.answer("❌ Не удалось построить отчёт. Проверьте логи.")
